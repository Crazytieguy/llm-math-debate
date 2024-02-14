"""
Based mostly on https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

import torch
import typer
from peft import LoraConfig  # type: ignore
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from ..data_processing.load_dataset import load_and_split_dataset
from ..data_processing.pydantic_models import Solution

tqdm.pandas()


def format_prompt(example_raw: dict) -> dict:
    solution = Solution.model_validate(example_raw)
    return {
        "text": f"""\
{solution.problem}

###

{solution.solution}

Correct: {"yes" if solution.solution_error is None else "no"}"""
    }


def main(
    base_model: str = "meta-llama/Llama-2-70b-hf",
    output_dir: str = "llama-70b-solution-classifier",
    dataset_path: str = "amps/mathematica/solution_dataset.jsonl",
):
    dataset_dict = load_and_split_dataset(dataset_path)
    dataset = dataset_dict["train"].map(format_prompt, batched=False)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # critical for memory usage
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        num_train_epochs=1,
        max_steps=-1,
        report_to="none",  # type: ignore
        save_steps=100,
        save_total_limit=10,
    )

    peft_config = LoraConfig(
        r=128,
        lora_alpha=8,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)

    # For weird reasons, this is required in order for the model to learn to output eos.
    # See https://github.com/huggingface/transformers/issues/22794
    # I don't expect to need the token '♥'
    tokenizer.add_special_tokens({"pad_token": "♥"})

    response_template = "\n\n###\n\n"
    # skip '<s>', '▁' and '</s>'
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:-1]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,  # type: ignore
        dataset_text_field="text",
        max_seq_length=1024,
        peft_config=peft_config,
    )

    trainer.train()  # type: ignore
    trainer.save_model(output_dir)


if __name__ == "__main__":
    typer.run(main)
