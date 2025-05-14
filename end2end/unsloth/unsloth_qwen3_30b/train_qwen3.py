from unsloth import FastModel
import transformers

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from torch.profiler import profile, ProfilerActivity


def load_model(model_name: str = "unsloth/Qwen3-30B-A3B",
               max_seq_length: int = 2048,
               load_in_4bit: bool = True,
               load_in_8bit: bool = False,
               full_finetuning: bool = True):
    """
    Load the Qwen3 MOE model with quantization.
    """
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=full_finetuning,
        cache_dir="/scratch/jlee436/unsloth/model",

    )
    return model, tokenizer


def get_peft_config(r: int = 16,
                    lora_alpha: int = 16,
                    target_modules: list = ["q_proj", "v_proj"],
                    bias: str = "none"):
    """
    Prepare LoRA configuration for QLoRA fine-tuning.
    """
    return LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        bias=bias,
    )


def load_data(dataset_name: str = "yahma/alpaca-cleaned",
              cache_dir: str = "/scratch/jlee436/unsloth/data"):
    """
    Load and return the training split of a HuggingFace dataset.
    """
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset["train"]


def get_training_args(output_dir: str = "./qwen3_30B_finetuned",
                      per_device_batch_size: int = 2,
                      gradient_accumulation_steps: int = 8,
                      num_train_epochs: int = 3,
                      learning_rate: float = 2e-5,
                      fp16: bool = True,
                      logging_steps: int = 10):
    """
    Create and return HuggingFace TrainingArguments.
    """
    return transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        # cache_dir="/scratch/jlee436/unsloth/model",
    )


def get_data_collator(tokenizer):
    """
    Prepare a data collator for causal language modeling.
    """
    return transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        # cache_dir="/scratch/jlee436/unsloth/model",
    )


def create_trainer(model, tokenizer, train_dataset, peft_config, training_args, data_collator):
    """
    Initialize and return an SFTTrainer for fine-tuning.
    """
    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )


def profile_and_train(trainer, trace_file: str = "cuda_profile_qwen3.json", top_k: int = 10):
    """
    Profile CUDA activities during training and export trace.
    """
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        trainer.train()

    # Print top CUDA kernels
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=top_k))
    # Export trace for Chrome
    prof.export_chrome_trace(trace_file)


def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    # Prepare LoRA config
    peft_config = get_peft_config()
    # Load training data
    train_dataset = load_data()
    # Prepare training args and data collator
    training_args = get_training_args()
    data_collator = get_data_collator(tokenizer)
    # Initialize trainer
    trainer = create_trainer(model, tokenizer, train_dataset,
                             peft_config, training_args, data_collator)
    # Run training with profiling
    profile_and_train(trainer)
    # Save the fine-tuned model
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
