import asyncio
import json
import logging
import random
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from toolz.itertoolz import count, take

from .compare_answers import AnswerComparison, compare_answers
from .error_insertion_prompt import format_error_insertion_prompt
from .pydantic_models import Solution, SolutionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()

insertion_failures_queue: Queue[dict] = Queue()


def insert_errors(final_error_count: int = 4):
    dataset_path = Path("amps/mathematica/solution_dataset.jsonl")
    data = [
        Solution.model_validate_json(line)
        for line in dataset_path.read_text().splitlines()
    ]
    task_results = []
    solutions = iter(data)
    done_event = Event()
    insertion_failures_thread = Thread(
        target=write_insertion_failures, args=(done_event,)
    )
    insertion_failures_thread.start()
    while True:
        num_errors = count(filter(lambda s: s.solution_error is not None, data))
        insertions_remaining = final_error_count - num_errors
        if insertions_remaining <= 0:
            break
        tasks = []
        for solution in take(min(insertions_remaining, 8), solutions):
            if solution.solution_error is None:
                tasks.append(generate_wrong_solution(solution))
            else:
                tasks.append(identity(solution))
        task_results.extend(asyncio.run(gather(tasks)))
        for i, result in enumerate(task_results):
            if isinstance(result, Solution):
                data[i] = result
    dataset_path.write_text("".join(s.model_dump_json() + "\n" for s in data))
    done_event.set()


def write_insertion_failures(done_event: Event):
    insertion_failures_file = Path("amps/mathematica/error_insertion_failures.jsonl")
    with insertion_failures_file.open("a") as f:
        while not done_event.is_set():
            try:
                insertion_failure = insertion_failures_queue.get(timeout=1)
            except Empty:
                continue
            f.write(json.dumps(insertion_failure) + "\n")


async def generate_wrong_solution(
    solution: Solution, retries_left: int = 2
) -> Solution:
    problem_identifier = (
        f"{solution.domain}/{solution.problem_class}/{solution.source_file}"
    )
    logger.info(f"Generating error insertion for {problem_identifier}")
    messages = get_few_shot_messages(solution.domain, solution.problem_class)
    error_step = random.randint(1, solution.steps)
    error_insertion_prompt = format_error_insertion_prompt(solution, error_step)
    messages.append({"role": "user", "content": error_insertion_prompt})
    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-4-1106-preview", messages=messages
        )
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("No completion content")
    try:
        modified_solution = validate_error_insertion(solution, error_step, content)
        comparison = await compare_answers(
            solution.solution, modified_solution.solution
        )
        if comparison == AnswerComparison.SAME:
            raise ValueError("Final answer is the same as the original")
        logger.info(f"Generated error insertion for {problem_identifier}")
        return modified_solution
    except ValueError as e:
        insertion_failure = solution.model_dump()
        insertion_failure["reason"] = str(e)
        insertion_failure["error_step"] = error_step
        insertion_failure["content"] = content
        insertion_failures_queue.put_nowait(insertion_failure)
        if retries_left <= 0:
            msg = f"Failed generating error insertion for {problem_identifier} due to {e}"
            logger.warning(msg)
            e.add_note(msg)
            raise e
        logger.warning(f"Retrying error insertion for {problem_identifier} due to {e}")
        return await generate_wrong_solution(solution, retries_left - 1)


def validate_error_insertion(
    original_solution: Solution, requested_step: int, error_insertion: str
) -> Solution:
    try:
        wrong_solution = error_insertion.split("```")[-2].strip()
    except IndexError:
        raise ValueError("No '```' found in completion")
    original_before_step, original_after_step = split_on_step(
        original_solution.solution, requested_step
    )
    try:
        wrong_before_step, wrong_after_step = split_on_step(
            wrong_solution, requested_step
        )
    except ValueError:
        raise ValueError(f"Wrong solution does not have step {requested_step}")
    if original_before_step != wrong_before_step:
        raise ValueError("Wrong solution does not have the same steps before the error")
    original_step = original_after_step.split(f"\n{requested_step + 1}. ")[0]
    wrong_step = wrong_after_step.split(f"\n{requested_step + 1}. ")[0]
    if original_step.strip() == wrong_step.strip():
        raise ValueError("Wrong solution does not have an error on the right step")
    wrong_sol_lower = wrong_solution.lower()
    for w in ["wrong", "error", "incorrect", "mistake"]:
        if w in wrong_sol_lower:
            raise ValueError(f"Wrong solution has the word {w} in it")
    before_solution = error_insertion.split("Modified solution:")[0].strip()
    explanation = None
    note = None
    try:
        before_note, after_note = before_solution.split("Note:")
        note = after_note.strip()
        if before_note.startswith("Explanation:"):
            explanation = before_note.removeprefix("Explanation:").strip()
    except ValueError:
        if before_solution.startswith("Explanation:"):
            explanation = before_solution.removeprefix("Explanation:").strip()

    return Solution(
        domain=original_solution.domain,
        problem_class=original_solution.problem_class,
        source_file=original_solution.source_file,
        problem=original_solution.problem,
        solution=wrong_solution,
        steps=original_solution.steps,
        solution_error=SolutionError(
            explanation=explanation,
            note=note,
            step=requested_step,
            original_solution=original_solution.solution,
        ),
    )


def split_on_step(solution: str, step: int) -> list[str]:
    if step == 1:
        return ["", solution.removeprefix("1. ").strip()]
    return [s.strip() for s in solution.split(f"\n{step}. ")]


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
            "content": """\
You are a math teacher creating a dataset of wrong solutions for educational purposes. \
You always follow the dataset's format: Explain the error you are about to insert, \
then make a note on how to propagate it to subsequent steps correctly, and finally, \
provide the wrong solution. You NEVER point out or explain the error inside the modified solution body.""",
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


async def gather(tasks):
    return await asyncio.gather(*tasks, return_exceptions=True)


async def identity(x):
    return x


if __name__ == "__main__":
    typer.run(insert_errors)
