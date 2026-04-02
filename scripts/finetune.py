#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning PLAME for MSA generation."""

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from transformers import (
    HfArgumentParser,
    T5Config,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import random_split

from plame.data.msadata import Alphabet, MSABatchConverter, MSADataSet_v3
from plame.models.model import MSA_AUGMENTOR

warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_PROJECT", "PLAME")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
os.environ.setdefault("WANDB_SILENT", "true")

check_min_version("4.16.0.dev0")
require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model/config/tokenizer selection."""

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments for training/eval data configuration."""

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to training data directory."},
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to evaluation data file."},
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "Path to test data file."},
    )
    num_alignments: int = field(
        default=32, metadata={"help": "Number of sequences per MSA sample."},
    )
    threshold: int = field(
        default=1536, metadata={"help": "Maximum sequence length in one batch."},
    )
    max_source_length: Optional[int] = field(
        default=1024, metadata={"help": "Max input sequence length after tokenization."},
    )
    max_target_length: Optional[int] = field(
        default=1024, metadata={"help": "Max target sequence length after tokenization."},
    )
    val_max_target_length: Optional[int] = field(
        default=None, metadata={"help": "Max target length for validation. Defaults to max_target_length."},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=100, metadata={"help": "Truncate evaluation examples to this value if set."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of processes for preprocessing."},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Ignore padded tokens in loss computation."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


class GradientClippingOptimizerWrapper(Optimizer):
    """Wraps an optimizer to add gradient clipping before each step."""

    def __init__(self, optimizer, max_grad_norm):
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        super().__init__(optimizer.param_groups, optimizer.defaults)

    def step(self, closure=None):
        if self.max_grad_norm > 0:
            for group in self.param_groups:
                clip_grad_norm_(group["params"], self.max_grad_norm)
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none)


class GradientClippingTrainer(Trainer):
    """Trainer with gradient clipping and auxiliary loss logging."""

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        self.optimizer = GradientClippingOptimizerWrapper(optimizer, self.args.max_grad_norm)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        if hasattr(model, "loss_fct_chq"):
            lm_logits = outputs.logits
            labels = inputs["labels"]
            _, weighted_ce, ce_loss, diversity_loss = model.loss_fct_chq(lm_logits, labels)
            self.log({
                "weight_ce_loss": weighted_ce.detach().item(),
                "ce_loss": ce_loss.detach().item(),
                "diversity_loss": diversity_loss.detach().item(),
            })

        return (loss, outputs) if return_outputs else loss


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # FSDP configuration
    training_args.fsdp = "full_shard auto_wrap"
    training_args.fsdp_transformer_layer_cls_to_wrap = "T5Block"
    training_args.fsdp_min_num_params = 0
    training_args.max_grad_norm = 1.0
    training_args.clip_grad_norm = 1.0
    training_args.safe_serialization = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, bf16 training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Tokenizer and dataset
    tokenizer = Alphabet.from_architecture(name="msa_transformer")
    data_args.tokenizer = tokenizer

    train_dataset = MSADataSet_v3(data_args, num_alignments=data_args.num_alignments, threshold=data_args.threshold)

    valid_ratio = 0.05
    test_ratio = 0.05
    valid_size = int(valid_ratio * len(train_dataset))
    test_size = int(test_ratio * len(train_dataset))
    train_size = len(train_dataset) - valid_size - test_size
    train_dataset, valid_dataset, test_dataset = random_split(
        train_dataset, [train_size, valid_size, test_size]
    )

    # Model
    config = T5Config.from_pretrained("./config")
    config.seq_per_msa = data_args.num_alignments
    config.vocab_size = len(tokenizer)
    config.torch_dtype = torch.bfloat16

    model = MSA_AUGMENTOR(config=config)
    model.gradient_checkpointing_enable()

    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(torch.bfloat16)

    msadata_collator = MSABatchConverter(tokenizer)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Trainer
    trainer = GradientClippingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=msadata_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    trainer.eval_dataset = test_dataset
    test_results = trainer.evaluate()
    logger.info(f"Perplexity: {math.exp(test_results['eval_loss']):.2f}")


if __name__ == "__main__":
    main()
