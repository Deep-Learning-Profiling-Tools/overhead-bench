# Original code from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb
import argparse
import torch


from datasets import load_dataset
import triton.profiler as proton

SUPPORTS_BFLOAT16 = torch.cuda.get_device_capability()[0] >= 8


def format_dataset(dataset, tokenizer):

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


    
def train_unsloth(args):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        cache_dir = "/scratch/jlee436/unsloth/model",
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    dataset = load_dataset("yahma/alpaca-cleaned", split = "train", cache_dir="/scratch/jlee436/unsloth/data")
    dataset = format_dataset(dataset, tokenizer)

    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not SUPPORTS_BFLOAT16,
            bf16 = SUPPORTS_BFLOAT16,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    if args.profiling == "proton":
        session_id = proton.start(name="profile_name", context="shadow")
        trainer_stats = trainer.train()
        proton.finalize(session_id)
    elif args.profiling == "torch":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            trainer_stats = trainer.train()
    else:
        trainer_stats = trainer.train()


def train_native():
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model
    from peft import LoraConfig, TaskType




    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = "unsloth/Meta-Llama-3.1-8B",
        # max_seq_length = max_seq_length,
        # dtype = dtype,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    ), AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0,
        use_rslora=False,
        # loftq_config=None,
        bias="none",
    )

    # lora_model = get_peft_model(model, lora_config)

    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    dataset = format_dataset(dataset, tokenizer)



    trainer = SFTTrainer(
        model = model,
        # tokenizer = tokenizer,
        train_dataset = dataset,
        # dataset_text_field = "text",
        # max_length = max_seq_length,
        # dataset_num_proc = 2,
        # packing = False, 
        peft_config=lora_config,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not SUPPORTS_BFLOAT16,
            bf16 = SUPPORTS_BFLOAT16,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth")
    parser.add_argument("--profiling", type=str, default="none")
    args = parser.parse_args()

    # use args.model to determine which model to train
    if args.model == "unsloth":
        train_unsloth(args)
    elif args.model == "native":
        train_native()
