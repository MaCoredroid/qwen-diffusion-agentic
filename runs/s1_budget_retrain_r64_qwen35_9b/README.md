---
library_name: peft
base_model: /home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init
tags:
- base_model:adapter:/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init
- lora
- transformers
datasets:
- customized
pipeline_tag: text-generation
model-index:
- name: s1_budget_retrain_r64_qwen35_9b
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# s1_budget_retrain_r64_qwen35_9b

This model is a fine-tuned version of [/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init](https://huggingface.co//home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init) on the customized dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 71101
- gradient_accumulation_steps: 2
- total_train_batch_size: 2
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: warmup_stable_decay
- lr_scheduler_warmup_steps: 100
- training_steps: 2000

### Training results



### Framework versions

- PEFT 0.19.1
- Transformers 4.53.1
- Pytorch 2.12.1+cu130
- Datasets 2.14.6
- Tokenizers 0.21.4