from collections import Counter, defaultdict
from pprint import pprint

import torch
import typer
from accelerate import Accelerator
from peft import PeftModel  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import is_xpu_available

from ..data_processing.load_dataset import load_and_split_dataset
from ..data_processing.pydantic_models import Solution
from .sft import format_prompt


def main(
    base_model: str = "meta-llama/Llama-2-13b-hf",
    model_dir: str = "llama-13b-solution-classifier",
    dataset_path: str = "amps/mathematica/solution_dataset.jsonl",
):
    dataset_dict = load_and_split_dataset(dataset_path)
    dataset = dataset_dict["test"]
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    stats_by_domain_class = defaultdict(Counter)
    stats_by_number_of_steps = defaultdict(Counter)
    for sample in dataset:
        solution = Solution.model_validate(sample)
        formatted = format_prompt(sample)  # type: ignore
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
        is_correct_str = "correct" if is_correct else "incorrect"
        stats_by_domain_class[f"{solution.domain}/{solution.problem_class}"][
            is_correct_str
        ] += 1
        stats_by_number_of_steps[solution.steps][is_correct_str] += 1
    print("Stats by domain class:")
    pprint(dict(stats_by_domain_class))
    print("Stats by number of steps:")
    pprint(dict(stats_by_number_of_steps))


if __name__ == "__main__":
    typer.run(main)
