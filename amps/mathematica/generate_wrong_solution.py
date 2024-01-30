import asyncio
from pathlib import Path

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .compare_answers import AnswerComparison, compare_answers
from .error_insertion_prompt import format_error_insertion_prompt

client = OpenAI()
app = typer.Typer()


async def generate_wrong_solution(
    error_insertion_prompt: str, domain: str, problem_class: str, retries_left: int = 1
) -> str:
    messages = get_prompt(error_insertion_prompt, domain, problem_class)
    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-4-1106-preview", messages=messages
        )
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("No completion content")
    try:
        await validate_error_insertion(error_insertion_prompt, content)
    except ValueError as e:
        if retries_left <= 0:
            raise ValueError(
                f"Could not generate valid error insertion\n\n{error_insertion_prompt}\n\n{content}"
            ) from e
        return await generate_wrong_solution(
            error_insertion_prompt, domain, problem_class, retries_left - 1
        )
    return content


async def validate_error_insertion(
    error_insertion_prompt: str, error_insertion: str
) -> None:
    original_solution = error_insertion_prompt.split("```")[-2]
    wrong_solution = error_insertion.split("```")[-2]
    requested_step = int(error_insertion_prompt.strip("\n.")[-2:])
    original_before_step, original_after_step = original_solution.split(
        f"\n{requested_step}. "
    )
    wrong_before_step, wrong_after_step = wrong_solution.split(f"\n{requested_step}. ")
    if original_before_step.strip() != wrong_before_step.strip():
        raise ValueError("Wrong solution does not have the same steps before the error")
    original_step = original_after_step.split(f"\n{requested_step + 1}. ")[0]
    wrong_step = wrong_after_step.split(f"\n{requested_step + 1}. ")[0]
    if original_step.strip() == wrong_step.strip():
        raise ValueError("Wrong solution does not have an error on the right step")
    wrong_sol_lower = wrong_solution.lower()
    for w in ["wrong", "error", "incorrect", "mistake"]:
        if w in wrong_sol_lower:
            raise ValueError(f"Wrong solution has the word {w} in it")
    answer_comparison = await compare_answers(original_solution, wrong_solution)
    if answer_comparison == AnswerComparison.SAME:
        raise ValueError("Wrong solution has the same answer as the original")


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


@app.command()
def generate_one(domain: str, problem_class: str, problem_file_name: str):
    if not problem_class.endswith("_w_steps"):
        problem_class += "_w_steps"
    problem_file = (
        Path("amps/mathematica/formatted") / domain / problem_class / problem_file_name
    )
    problem = problem_file.read_text()
    error_insertion_prompt = format_error_insertion_prompt(problem)
    wrong_solution = asyncio.run(
        generate_wrong_solution(error_insertion_prompt, domain, problem_class)
    )
    print(error_insertion_prompt)
    print(wrong_solution)


async def gather(tasks):
    return await asyncio.gather(*tasks, return_exceptions=True)


@app.command()
def generate_many(domain: str, problem_class: str, n: int = 10):
    output_dir = Path("amps/mathematica/error_insertions") / domain / problem_class
    output_dir.mkdir(parents=True, exist_ok=True)
    if not problem_class.endswith("_w_steps"):
        problem_class += "_w_steps"
    tasks = []
    for i in range(n):
        problem_file = (
            Path("amps/mathematica/formatted") / domain / problem_class / f"{i}.txt"
        )
        problem = problem_file.read_text()
        error_insertion_prompt = format_error_insertion_prompt(problem)
        tasks.append(
            generate_wrong_solution(error_insertion_prompt, domain, problem_class)
        )
    results = asyncio.run(gather(tasks))
    for i, result in enumerate(results):
        output_file = output_dir / f"{i}.txt"
        output_file.write_text(str(result))


if __name__ == "__main__":
    app()
