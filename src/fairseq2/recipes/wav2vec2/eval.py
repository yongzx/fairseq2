# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, TextIO, final

from torch import Tensor
from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.recipes.evaluator import AbstractEvalUnit
from fairseq2.recipes.utils.setup import check_model_type
from fairseq2.recipes.wav2vec2.common import Wav2Vec2MetricBag
from fairseq2.typing import override

log = get_log_writer(__name__)


@final
class Wav2Vec2EvalUnit(AbstractEvalUnit[Tensor]):
    """Represents the evaluation unit of a wav2vec 2.0 model."""

    def __init__(
        self,
        model: Module,
        gang: Gang,
        *,
        output_stream: Optional[TextIO] = None,
    ) -> None:
        """
        :param model:
            The wav2vec 2.0 model. Might be wrapped with DDP or FSDP.
        :param gang:
            The gang for distributed evaluation.
        :param output_stream:
            The output stream to dump evaluation output.
        """
        super().__init__(model)

        check_model_type(model, Wav2Vec2Model)

        self._metric_bag = Wav2Vec2MetricBag(gang)

    @override
    def __call__(self, batch: Tensor) -> None:
        input_batch = SequenceBatch(batch, None)

        output = self._model(input_batch)

        loss = output.compute_loss()

        self._metric_bag.update_losses(input_batch, loss.detach())

        self._metric_bag.update_batch_metrics(input_batch)

    @property
    @override
    def metric_bag(self) -> Wav2Vec2MetricBag:
        return self._metric_bag

    @property
    @override
    def throughput_metric_name(self) -> str:
        return "num_source_elements"
