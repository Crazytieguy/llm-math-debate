from pathlib import Path

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .error_insertion_prompt import format_error_insertion_prompt

client = OpenAI()


def get_few_shot_messages(
    domain: str, problem_class: str
) -> list[ChatCompletionMessageParam]:
    if problem_class.endswith("_w_steps"):
        problem_class = problem_class.removesuffix("_w_steps")
    dir = Path("amps/mathematica/error_insertion_examples") / domain / problem_class
    if not dir.exists():
        raise ValueError(f"Directory {dir} does not exist")
    examples = list(dir.glob("?.txt"))  # There should always be less than 10 examples
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a math teacher creating a dataset of wrong solutions for educational purposes.",
        }
    ]
    for example in examples:
        example_answer = dir / (example.stem + ".wrong.txt")
        messages.extend(
            [
                {"role": "user", "content": example.read_text()},
                {"role": "assistant", "content": example_answer.read_text()},
            ]
        )
    return messages


def get_prompt(
    error_insertion_prompt: str, domain: str, problem_class: str
) -> list[ChatCompletionMessageParam]:
    messages = get_few_shot_messages(domain, problem_class)
    messages.append({"role": "user", "content": error_insertion_prompt})
    return messages


def generate_wrong_solution(
    error_insertion_prompt: str, domain: str, problem_class: str
) -> str | None:
    messages = get_prompt(error_insertion_prompt, domain, problem_class)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )
    return response.choices[0].message.content


def main(domain: str, problem_class: str, problem_file_name: str):
    if not problem_class.endswith("_w_steps"):
        problem_class += "_w_steps"
    problem_file = (
        Path("amps/mathematica/formatted") / domain / problem_class / problem_file_name
    )
    problem = problem_file.read_text()
    error_insertion_prompt = format_error_insertion_prompt(problem)
    wrong_solution = generate_wrong_solution(
        error_insertion_prompt, domain, problem_class
    )
    print(error_insertion_prompt)
    print(wrong_solution)


if __name__ == "__main__":
    typer.run(main)
