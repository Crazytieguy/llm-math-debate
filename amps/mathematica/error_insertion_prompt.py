import random
from pathlib import Path

import typer

from .pydantic_models import Solution, parse_solution_file


def main(domain: str, problem_class: str):
    prompt = sample_error_prompt(domain, problem_class)
    print(prompt, end="")


def sample_error_prompt(domain: str, problem_class: str):
    problem_file = randomly_select_problem(domain, problem_class)
    solution = parse_solution_file(domain, problem_class, problem_file)
    return format_error_insertion_prompt(solution)


def format_error_insertion_prompt(solution: Solution, step: int | None = None) -> str:
    if step is None:
        step = random.randint(1, solution.steps)

    return f"""Given the following problem:

```

{solution.problem}

```

And correct solution:

```

{solution.solution}

```

Insert an error into step {step}.
"""


def randomly_select_problem(domain: str, problem_class: str) -> Path:
    if not problem_class.endswith("_w_steps"):
        problem_class += "_w_steps"
    dir = Path("amps/mathematica/formatted") / domain / problem_class
    if not dir.exists():
        raise ValueError(f"Directory {dir} does not exist")
    files = list(dir.iterdir())
    problem_file = random.choice(files)
    return problem_file


if __name__ == "__main__":
    typer.run(main)
