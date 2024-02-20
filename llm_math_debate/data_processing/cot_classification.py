import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import typer

from .pydantic_models import Solution

client = openai.Client()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model: str = "gpt-3.5-turbo"):
    raw_solutions = (
        Path("amps/mathematica/solution_dataset.jsonl").read_text().splitlines()
    )
    solutions = [Solution.model_validate_json(line) for line in raw_solutions]
    out_file = Path(f"amps/mathematica/{model}-classification.jsonl")
    with ThreadPoolExecutor(max_workers=8) as executor, out_file.open("w") as f:
        futures = [
            executor.submit(process_solution, solution, model) for solution in solutions
        ]
        for future in as_completed(futures):
            result = future.result()
            logger.info(
                f"Classified {result['domain']}/{result['problem_class']}/{result['source_file']}: {result['classification_correct']}"
            )
            f.write(json.dumps(result) + "\n")


def process_solution(solution: Solution, model: str, retries_left=3) -> dict:
    prompt = get_prompt(solution)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful math teacher."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise Exception("OpenAI returned no content")
    base_response = solution.model_dump() | {"chain_of_thought": content}
    last_line = content.splitlines()[-1]
    solution_correct = solution.solution_error is None
    if "INCORRECT" in last_line:
        return base_response | {"classification_correct": not solution_correct}
    if "CORRECT" in last_line:
        return base_response | {"classification_correct": solution_correct}
    if retries_left > 0:
        return process_solution(solution, model, retries_left - 1)
    return base_response | {"classification_correct": None}


def get_prompt(solution: Solution) -> str:
    return f"""Given the following problem:

```
{solution.problem}
```

Is the following solution correct?

```
{solution.solution}
```

Verify each step individually, then give your final answer (CORRECT / INCORRECT)
"""


if __name__ == "__main__":
    typer.run(main)
