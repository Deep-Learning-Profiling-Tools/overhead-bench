from dataclasses import dataclass

import datasets
import torch
from torch import bfloat16
import transformers

# from callback import EfficiencyCallback
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer
import time

from liger_kernel.transformers import AutoLigerKernelForCausalLM
import triton.profiler as proton
import torch.profiler

@dataclass
class CustomArguments:
    model_name: str = "Qwen/Qwen2-1.5B"
    dataset: str = "tatsu-lab/alpaca"
    max_seq_length: int = 256
    use_liger: bool = False
    profile_torch: bool = False


def formatting_prompts_func(example):
    return example["text"]


def train():
    parser = transformers.HfArgumentParser((transformers.TrainingArguments, CustomArguments))
    training_args, custom_args = parser.parse_args_into_dataclasses()
    training_args.bf16 = True
    training_args.bf16_full_eval = True
    training_args.use_liger_kernel = custom_args.use_liger
    training_args.max_seq_length = custom_args.max_seq_length
    training_args.max_steps = 250

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        custom_args.model_name,
        padding_side="left",
        truncation_side="left",
        cache_dir="/scratch/jlee436/liger/model"
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset(custom_args.dataset, cache_dir="/scratch/jlee436/liger/data")["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    response_prompt = tokenizer.encode("### Response:\n", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_prompt,
        pad_to_multiple_of=4,
    )

    if custom_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            custom_args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=bfloat16,
            cache_dir="/scratch/jlee436/liger/model",
            # These args will get passed to the appropriate apply_liger_kernel_to_* function
            # to override the default settings
            # cross_entropy=True,
            # fused_linear_cross_entropy=False,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            custom_args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=bfloat16,
            cache_dir="/scratch/jlee436/liger/model",
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        # max_seq_length=custom_args.max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        # callbacks=[EfficiencyCallback()],
    )
    if custom_args.profile_torch:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with torch.profiler.record_function("trainer.train"):
                print(f"START_PROFILE: {time.time()}")
                trainer.train()
                print(f"END_PROFILE: {time.time()}")
        prof.export_stacks(f"pt_trace.json")
    else:
        with proton.scope("trainer"):
            print(f"START_PROFILE: {time.time()}")
            trainer.train()
            print(f"END_PROFILE: {time.time()}")


if __name__ == "__main__":
    train()
