# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch

from fairseq2.assets import default_asset_store
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.instruction import load_instruction_dataset
from fairseq2.generation import (
    BeamSearchSequenceGenerator,
    Sampler,
    SamplingSequenceGenerator,
    SequenceGenerator,
    StandardBeamSearchAlgorithm,
    TopKSampler,
    TopPSampler,
)
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.decoder import DecoderModel
from fairseq2.recipes.lm.text_completer import TextCompleter
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, setup_gangs
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class SamplingConfig:
    """Holds the configuration for sequence generation based on sampling.

    See :class:`SamplingSequenceGenerator` for more info.
    """

    sampler: Literal["top-p", "top-k"] = "top-p"
    """The sampling algorithm."""

    top_p: float = 0.9
    """The cumulative probability threshold for top-p sampling."""

    top_k = 10
    """The number of top candidates to select from for top-k sampling."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int = 512
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = False
    """If ``True``, returns generated sequences with prompts appended."""

    compute_scores: bool = False
    """If ``True``, computes scores of generated sequences."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 0.6
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


@dataclass
class BeamSearchConfig:
    """Holds the configuration for sequence generation based on beam search.

    See :class:`BeamSearchSequenceGenerator` for more info.
    """

    algorithm: Literal["standard"] = "standard"
    """The beam search algorithm."""

    beam_size: int = 5
    """The beam size."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: int = 512
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
    """The maximum sequence length including prompt."""

    echo_prompt: bool = False
    """If ``True``, returns generated sequences with prompts appended."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 1.0
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


@dataclass
class TextCompleteConfig:
    """Holds the configuration of a text completion recipe."""

    dataset_name: str = "oa2_gsm8k_safety"  # TODO: change!
    """The dataset to generate with."""

    tokenizer_name: str = "llama3_instruct"
    """The tokenizer to use."""

    batch_size: int = 1
    """The input batch size."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model_name: str = "llama3_8b_instruct"
    """The name of the model to finetune."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.bfloat16
    """The data type of the model."""

    tensor_parallel_size: int = 1
    """The size of Megatron-style tensor parallelism."""

    # Generation
    mode: Literal["sampling", "beam_search"] = "sampling"
    """The mode of sequence generation."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    """The configuration for sequence generation based on sampling."""

    beam_search: BeamSearchConfig = field(default_factory=BeamSearchConfig)
    """The configuration for sequence generation based on beam search."""

    seed: int = 2
    """The random number generator seed for sequence generation."""


text_complete_presets = ConfigRegistry[TextCompleteConfig]()

text_complete_preset = text_complete_presets.decorator


@text_complete_preset("llama3_8b_instruct")
def _llama3_8b_instruct() -> TextCompleteConfig:
    return TextCompleteConfig()


@text_complete_preset("llama3_70b_instruct")
def _llama3_70b_instruct() -> TextCompleteConfig:
    config = _llama3_8b_instruct()

    config.model_name = "llama3_70b_instruct"
    config.tensor_parallel_size = 8

    return config


@text_complete_preset("llama2_7b_chat")
def _llama2_7b_chat() -> TextCompleteConfig:
    config = _llama3_8b_instruct()

    config.model_name = "llama2_7b_chat"
    config.tokenizer_name = "llama2"

    return config


@text_complete_preset("llama2_70b_chat")
def _llama2_70b_chat() -> TextCompleteConfig:
    config = _llama2_7b_chat()

    config.model_name = "llama2_70b_chat"
    config.tensor_parallel_size = 8

    return config


def load_text_completer(config: TextCompleteConfig, output_dir: Path) -> TextCompleter:
    """Load a :class:`TextCompleter`."""
    wall_watch = Stopwatch(start=True)

    root_gang, gangs = setup_gangs(log, tp_size=config.tensor_parallel_size)

    dp_gang = gangs["dp"]  # data
    tp_gang = gangs["tp"]  # tensor

    log.info("Loading {} tokenizer.", config.tokenizer_name)

    tokenizer = load_text_tokenizer(config.tokenizer_name)

    log.info("Tokenizer loaded.")

    log.info("Loading {} dataset.", config.dataset_name)

    dataset = load_instruction_dataset(config.dataset_name)

    data_reader = dataset.create_prompt_reader(
        split="test",
        tokenizer=tokenizer,
        gang=dp_gang,
        batch_size=config.batch_size,
        num_prefetch=config.num_prefetch,
    )

    log.info("Dataset loaded.")

    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.checkpoint_dir)
        )

    log.info("Loading {} model on data parallel rank 0 (per shard).", config.model_name)

    if dp_gang.rank == 0:
        init_device = dp_gang.device
    else:
        init_device = META

    model = load_model(
        config.model_name, gangs=gangs, device=init_device, dtype=config.dtype
    )

    root_gang.barrier()

    log.info("Model loaded on data parallel rank 0.")

    if not isinstance(model, DecoderModel):
        raise ValueError("`config.model_name` must specify a decoder model.")

    if dp_gang.size != 1:
        broadcast_model(model, dp_gang, log)

    log_model(model, log)

    # Initialize the sequence generator.
    generator: SequenceGenerator

    if config.mode == "sampling":
        sampler: Sampler

        if config.sampling.sampler == "top-p":
            sampler = TopPSampler(config.sampling.top_p)
        elif config.sampling.sampler == "top-k":
            sampler = TopKSampler(config.sampling.top_k)
        else:
            raise ValueError(
                f"`config.sampling.sampler` must be 'top-p' or 'top-k', but is '{config.sampling.sampler}' instead."
            )

        generator = SamplingSequenceGenerator(
            model,
            sampler,
            min_gen_len=config.sampling.min_gen_len,
            max_gen_len=config.sampling.max_gen_len,
            max_seq_len=config.sampling.max_seq_len,
            echo_prompt=config.sampling.echo_prompt,
            compute_scores=config.sampling.compute_scores,
            normalize_scores=config.sampling.normalize_scores,
            temperature=config.sampling.temperature,
            unk_penalty=config.sampling.unk_penalty,
            len_penalty=config.sampling.len_penalty,
            prefill_chunk_size=config.sampling.prefill_chunk_size,
            decode_capacity_increment=config.sampling.decode_capacity_increment,
        )
    elif config.mode == "beam_search":
        if config.beam_search.algorithm == "standard":
            algorithm = StandardBeamSearchAlgorithm()
        else:
            raise ValueError(
                f"`config.beam_search.algorithm` must be 'standard', but is '{config.beam_search.algorithm}' instead."
            )

        generator = BeamSearchSequenceGenerator(
            model,
            algorithm=algorithm,
            beam_size=config.beam_search.beam_size,
            min_gen_len=config.beam_search.min_gen_len,
            max_gen_len=config.beam_search.max_gen_len,
            echo_prompt=config.beam_search.echo_prompt,
            normalize_scores=config.beam_search.normalize_scores,
            temperature=config.beam_search.temperature,
            unk_penalty=config.beam_search.unk_penalty,
            len_penalty=config.beam_search.len_penalty,
            prefill_chunk_size=config.beam_search.prefill_chunk_size,
            decode_capacity_increment=config.beam_search.decode_capacity_increment,
        )
    else:
        raise ValueError(
            f"`config.mode` must be 'sampling' or 'beam_search', but is '{config.model}' instead."
        )

    output_file = output_dir.joinpath(f"output_{dp_gang.rank}.txt")

    return TextCompleter(
        generator=generator,
        tokenizer=tokenizer,
        gang=root_gang,
        dp_gang=dp_gang,
        tp_gang=tp_gang,
        data_reader=data_reader,
        output_file=output_file,
        seed=config.seed,
        wall_watch=wall_watch,
    )
