import os
import time
import torch
from unsloth import FastModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from unsloth.chat_templates import standardize_sharegpt
import pandas as pd
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import triton.profiler as proton

model_map = {
    "phi-4": "unsloth/Phi-4",
    "gemma-3-4b": "unsloth/gemma-3-4b-it",
    "llama-3.1-8b": "unsloth/Llama-3.1-8B",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B",
    "qwen3-14b": "unsloth/Qwen3-14B",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "llama-4-scout": "unsloth/Llama-4-Scout-17B-16E-unsloth-bnb-4bit"
}
def format_dataset_llama(dataset, tokenizer):

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    return dataset.map(formatting_prompts_func, batched = True,)

def main(profiling_mode, model_name):
    # Initialize model and tokenizer
    print(f"Initializing model and tokenizer for {model_name}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_map[model_name],
        max_seq_length = 2048,
        load_in_4bit = False,
        load_in_8bit = False,
        full_finetuning = False,
        cache_dir = "/scratch/jlee436/unsloth/model"
    )

    # Add LoRA adapters
    print(f"Adding LoRA adapters for {model_name}...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        r = 8,
        lora_alpha = 8,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )

    # Load and prepare dataset
    print("Loading and preparing dataset...")



    dataset = load_dataset("mlabonne/FineTome-100k", split="train", cache_dir="/scratch/jlee436/unsloth/data")
    dataset = standardize_sharegpt(dataset, tokenizer)

    if "llama" in model_name.lower():
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
    elif "mistral" in model_name.lower():
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "mistral",
        )


    dataset = tokenizer.apply_chat_template(
        dataset["conversations"],
        tokenize = False,
    )

    data = pd.Series(dataset)[:1000]
    data.name = "text"
    final_dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Initialize trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = final_dataset,
        eval_dataset = None,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 1,  # Change this for longer training
            learning_rate = 2e-4,
            logging_steps = 15,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none",
        ),
    )

    print(f"START_PROFILE: {time.time()}")
    if profiling_mode == "proton":
        session_id = proton.start(name=f"unsloth_{model_name}", context="shadow")
        trainer.train()
        proton.finalize(session_id)
    elif profiling_mode == "torch":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            trainer.train()
        prof.export_chrome_trace(f"unsloth_trace_{model_name}.json")
    else:
        trainer.train()
    print(f"END_PROFILE: {time.time()}")



if __name__ == "__main__":
    # parse arguments to check profiling mode, which is a string "proton", "torch", or "none"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiling", type=str, default="none")
    parser.add_argument("--model", type=str, default="phi-4")
    args = parser.parse_args()

    main(args.profiling, args.model)