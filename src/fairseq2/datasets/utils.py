# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq2.gang import Gang
from fairseq2.logging import LogWriter


def _reduce_num_batches(num_batches: int, gang: Gang, log: LogWriter) -> int:
    all_num_batches = torch.zeros((gang.size,), device=gang.device, dtype=torch.int64)

    num_batches_ = torch.tensor(num_batches, device=gang.device)

    gang.all_gather(all_num_batches, num_batches_)

    min_num_batches = int(all_num_batches.min())
    if min_num_batches != 0:
        return min_num_batches

    # If not all processes have reached end of data, report the ones that have
    # reached for debugging purposes.
    if log.is_enabled_for(logging.DEBUG) and all_num_batches.sum() > 0:
        ranks = all_num_batches.bool().logical_not_().nonzero().squeeze(-1).tolist()

        s = ", ".join(str(r) for r in ranks)

        log.debug("End of data reached at rank(s) {}.", s)

    return 0


def batch_by_size_vec(
    indices, num_tokens_vec, max_tokens: int, max_sentences: int, bsz_mult: int
):
    if indices.size == 0:
        return []

    assert (
        max_tokens <= 0 or np.max(num_tokens_vec) <= max_tokens
    ), f"Sentences lengths should not exceed max_tokens={max_tokens}"

    indices_len = len(indices)
    batches_ends = np.zeros(indices_len, dtype=np.int32)
    pos = 0
    new_batch_end = 0
    new_batch_max_tokens = 0
    new_batch_sentences = 0
    new_batch_num_tokens = 0
    overflow = False
    size_matches_with_bsz_mult = False
    batches_count = 0
    batch_start = 0
    tail_max_tokens = 0
    batch_max_tokens = 0

    for pos in range(indices_len):
        tail_max_tokens = max(tail_max_tokens, num_tokens_vec[pos])
        new_batch_end = pos + 1
        new_batch_max_tokens = max(batch_max_tokens, tail_max_tokens)
        new_batch_sentences = new_batch_end - batch_start
        new_batch_num_tokens = new_batch_sentences * new_batch_max_tokens
        overflow = (
            new_batch_sentences > max_sentences > 0
            or new_batch_num_tokens > max_tokens > 0
        )
        size_matches_with_bsz_mult = (
            new_batch_sentences < bsz_mult or new_batch_sentences % bsz_mult == 0
        )
        if overflow:
            tail_num_tokens = tail_max_tokens * (
                new_batch_end - batches_ends[batches_count]
            )
            tail_overflow = tail_num_tokens > max_tokens > 0
            if tail_overflow:
                batches_count += 1
                batches_ends[batches_count] = pos
                tail_max_tokens = num_tokens_vec[pos]
            batch_start = batches_ends[batches_count]
            batches_count += 1
            new_batch_max_tokens = tail_max_tokens
        if overflow or size_matches_with_bsz_mult:
            batches_ends[batches_count] = new_batch_end
            batch_max_tokens = new_batch_max_tokens
            tail_max_tokens = 0

    if batches_ends[batches_count] != indices_len:
        batches_count += 1

    return np.split(indices, batches_ends[:batches_count])


def batch_by_size_fn(
    indices, num_tokens_fn, max_tokens: int, max_sentences: int, bsz_mult: int
):
    indices_len = len(indices)
    num_tokens_vec = np.zeros(indices_len, dtype=np.int64)
    for pos in range(indices_len):
        num_tokens_vec[pos] = num_tokens_fn(indices[pos])
    return batch_by_size_vec(
        indices, num_tokens_vec, max_tokens, max_sentences, bsz_mult
    )
