import json
from pathlib import Path
from typing import Optional

import torch
import typer
from peft import PeftModel  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ..data_processing.load_dataset import load_and_split_dataset
from .sft import format_prompt


def main(
    base_model: str = "meta-llama/Llama-2-70b-hf",
    model_dir: str = "llama-70b-solution-classifier",
    dataset_path: str = "amps/mathematica/solution_dataset.jsonl",
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = Path(f"{model_dir}-test.jsonl")
    dataset_dict = load_and_split_dataset(dataset_path)
    dataset = dataset_dict["test"]
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    data = []
    for sample in dataset:
        assert isinstance(sample, dict)
        formatted = format_prompt(sample)
        prompt = formatted["text"].removesuffix(" yes").removesuffix(" no")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=10)
        output_text = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(),  # type: ignore
            skip_special_tokens=True,
        )[0]
        correct = formatted["text"].split(" ")[-1]
        inferred = output_text.split(" ")[-1]
        is_correct = correct == inferred
        sample["inferred"] = inferred
        sample["is_correct"] = is_correct
        data.append(sample)
    output_path.write_text("\n".join(json.dumps(sample) for sample in data))


if __name__ == "__main__":
    typer.run(main)
