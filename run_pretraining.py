# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from pathlib import Path

from fairseq2 import setup_extensions
from fairseq2.logging import get_log_writer
from fairseq2.recipes.logging import setup_basic_logging, setup_logging
from fairseq2.recipes.utils.log import exception_logger, log_config
from fairseq2.recipes.wav2vec2.train import (
    load_wav2vec2_trainer,
    wav2vec2_train_presets,
)
from fairseq2.utils.cluster import Cluster, ClusterConfig

log = get_log_writer(__name__)
USER = os.getenv("USER")


def train_wav2vec2_model(config, output_dir: Path) -> None:
    """Run wav2vec2 pretraining.

    :param config:
        The job configuration.
    :param output_dir:
        The output directory to store checkpoints and logs.
    """

    with exception_logger(log):
        setup_extensions()
        setup_basic_logging(debug=False)
        log_file = output_dir.expanduser().joinpath("logs/rank_{rank}.log").resolve()
        setup_logging(log_file, debug=False)
        log_config(config, log, output_dir.joinpath("config.yaml"))
        trainer = load_wav2vec2_trainer(config, output_dir)
        trainer()


def main(args: Namespace) -> None:
    num_gpus = 64
    preset = args.preset
    config = wav2vec2_train_presets.get(preset)
    config.seed = args.seed
    output_dir = Path(f"/checkpoint/{USER}/wav2vec2_train/{preset}_seed{config.seed}")

    cluster_config = ClusterConfig(
        cluster="slurm",
        parallelism=1,
        partition="nllb,ust,devaccel,learnaccel",
        num_nodes=(num_gpus + 7) // 8,
        num_gpus_per_node=min(num_gpus, 8),
        cpus_per_task=10,
        log_dir=output_dir.joinpath("submitit"),
        timeout=timedelta(minutes=4000),
    )

    cluster = Cluster(cluster_config)

    cluster.run_job(train_wav2vec2_model, config, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(prog="train_mms")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--preset", type=str, default="base_960h_fs1_masking_5epoch")
    args = parser.parse_args()
    main(args)
