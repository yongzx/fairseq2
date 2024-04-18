# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from pathlib import Path
from typing import List, Optional, TextIO, final

from fairseq2.data.text import TextTokenDecoder, TextTokenizer
from fairseq2.datasets import DataReader
from fairseq2.gang import FakeGang, Gang
from fairseq2.generation import SequenceGenerator
from fairseq2.logging import get_log_writer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.utils.cli import create_rich_progress
from fairseq2.typing import CPU, override
from fairseq2.utils.profiler import Stopwatch
from fairseq2.utils.rng import RngBag

log = get_log_writer(__name__)


@final
class TextCompleter:
    _generator: SequenceGenerator
    _text_decoder: TextTokenDecoder
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _data_reader: DataReader[SequenceBatch]
    _output_file: Path
    _seed: int
    _step_nr: int
    _wall_watch: Stopwatch
    _run: bool

    def __init__(
        self,
        generator: SequenceGenerator,
        tokenizer: TextTokenizer,
        gang: Gang,
        data_reader: DataReader[SequenceBatch],
        output_file: Path,
        wall_watch: Stopwatch,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        seed: int = 2,
    ) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The text tokenizer.
        :param gang:
            The gang to use for distributed evaluation.
        :param data_reader:
            The data reader of the eval split.
        :param output_file:
            The output file.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param dp_gang:
            The data parallel gang. If ``None``, ``gang`` will be used.
        :param tp_gang:
            The tensor parallel gang. Only required for tensor parallel models
            such as LLaMA 70B.
        :param seed:
            The random number generator seed for sequence generation.
        """
        self._generator = generator

        self._text_decoder = tokenizer.create_decoder()

        self._root_gang = gang

        if dp_gang is not None and tp_gang is not None:
            self._dp_gang = dp_gang
            self._tp_gang = tp_gang
        elif dp_gang is None and tp_gang is None:
            self._dp_gang = gang
            self._tp_gang = FakeGang(device=gang.device)
        else:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

        self._data_reader = data_reader

        self._output_file = output_file

        self._seed = seed

        self._step_nr = 0

        self._wall_watch = wall_watch

        self._run = False

    @override
    def __call__(self) -> None:
        if self._run:
            raise RuntimeError("The text completer can only be run once.")

        self._run = True

        log.info("Running text completion on {} device(s).", self._root_gang.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Text completion terminated at step {}!", self._step_nr)

            raise

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Text completion complete in {:,} seconds after {} steps!", int(elapsed_time), self._step_nr)  # fmt: skip

    def _do_run(self) -> None:
        rng_bag = RngBag.from_device_defaults(CPU, self._root_gang.device)

        # Set the seed for sequence generation.
        rng_bag.manual_seed(self._seed)

        try:
            fp = self._output_file.open("w")
        except OSError as ex:
            raise RuntimeError(
                f"The output file ({output_file}) cannot be opened. See nested exception for details."
            ) from ex

        with fp, create_rich_progress() as progress:
            task = progress.add_task("complete", total=None)

            for step_nr in count(start=1):
                self._step_nr = step_nr

                try:
                    batches = next(self._data_reader)
                except StopIteration:
                    break

                progress.update(task, advance=1)

                log.debug("Running step {}.", step_nr)

                for batch in batches:
                    self._run_generator(batch, fp)

                self._root_gang.barrier()

    def _run_generator(self, batch: SequenceBatch, fp: TextIO) -> None:
        output = self._generator(batch.seqs, batch.padding_mask)

        if self._tp_gang.rank == 0:
            prompts = batch.example["prompt"]

            responses = []

            scores = []

            step_scores = []

            for idx, hypotheses in enumerate(output.hypotheses):
                if len(hypotheses) == 0:
                    raise RuntimeError(
                        f"The sequence generator returned no hypothesis at index {idx}. Please file a bug report."
                    )

                hypothesis = hypotheses[0]

                response = self._text_decoder(hypothesis.seq)

                responses.append(response)

                scores.append(hypothesis.score)

                step_scores.append(hypothesis.step_scores)

            for idx, (prompt, response) in enumerate(zip(prompts, responses)):
                fp.write("<<<<< PROMPT >>>>>\n")
                fp.write(prompt)

                fp.write("\n\n\n<<<<< RESPONSE >>>>>\n")
                fp.write(response)

                if scores[idx] is not None:
                    fp.write("\n\n\n<<<<< SCORE >>>>>\n")

                    fp.write(f"{float(scores[idx]):.8f}\n")

                    fp.write(", ".join(f"{s:.8f}" for s in step_scores[idx].tolist()))

                fp.write("\n\n\n============================\n\n\n")

                fp.flush()
