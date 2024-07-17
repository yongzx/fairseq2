# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch

from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag
from fairseq2.metrics.aggregation import Mean, Sum
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Loss


class Wav2Vec2MetricBag(MetricBag):
    """Holds the training metrics of a wav2vec 2.0 model."""

    _loss: Mean
    _contrastive_loss: Mean
    _diversity_loss: Mean
    _feature_penalty: Mean
    _batch_size: Mean
    _elements_per_batch: Mean
    _num_examples: Sum
    _num_source_elements: Sum
    _total_num_examples: Sum
    _total_num_source_elements: Sum

    def __init__(self, gang: Gang) -> None:
        """
        :param gang:
            The gang over which to sync metrics.
        """
        super().__init__(gang)

        d = gang.device

        self.register_metric("_loss", Mean(device=d), persistent=False)

        self.register_metric("_contrastive_loss", Mean(device=d), persistent=False)

        self.register_metric("_diversity_loss", Mean(device=d), persistent=False)

        self.register_metric("_feature_penalty", Mean(device=d), persistent=False)

        self.register_metric("_batch_size", Mean(device=d), persistent=False)

        self.register_metric("_elements_per_batch", Mean(device=d), persistent=False)

        self.register_metric("_num_examples", Sum(device=d), persistent=False)

        self.register_metric("_num_source_elements", Sum(device=d), persistent=False)

        self._total_num_examples = Sum(device=d)

        self._total_num_source_elements = Sum(device=d)

    @torch.inference_mode()
    def update_losses(self, batch: SequenceBatch, loss: Wav2Vec2Loss) -> None:
        """Update the loss metrics.

        :param batch:
            The batch processed by the model.
        :param loss:
            The loss of ``batch``.
        """
        self._loss.update(
            loss.total / batch.batch_size / math.log(2), weight=batch.batch_size
        )

        self._contrastive_loss.update(
            loss.contrastive / batch.batch_size / math.log(2), weight=batch.batch_size
        )

        self._diversity_loss.update(
            loss.diversity / batch.batch_size / math.log(2), weight=batch.batch_size
        )

        self._feature_penalty.update(
            loss.feature_penalty / batch.batch_size / math.log(2),
            weight=batch.batch_size,
        )

    @torch.inference_mode()
    def update_batch_metrics(self, batch: SequenceBatch) -> None:
        """Update the batch metrics.

        :param seqs:
            The batch of seqs processed by the model.
        """
        batch_size = batch.batch_size

        num_source_elements = batch.num_elements()

        self._batch_size.update(batch_size * self._gang.size)

        self._elements_per_batch.update(num_source_elements * self._gang.size)

        self._num_examples.update(batch_size)

        self._num_source_elements.update(num_source_elements)

        self._total_num_examples.update(batch_size)

        self._total_num_source_elements.update(num_source_elements)
