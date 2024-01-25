import random
from pathlib import Path

import typer


def main(domain: str, problem_class: str):
    prompt = sample_error_prompt(domain, problem_class)
    print(prompt)


def sample_error_prompt(domain: str, problem_class: str):
    problem_file = randomly_select_problem(domain, problem_class)
    return format_error_insertion_prompt(problem_file.read_text())


def format_error_insertion_prompt(problem: str) -> str:
    problem, solution = problem.split("\n\n\\hrule\n\n")
    step = random.choice(
        [
            step
            for line in solution.splitlines()
            if (step := safe_int(line.split(". ")[0])) is not None
        ]
    )

    return f"""Given the following problem:
```

{problem}

```

And correct solution:

```

{solution}

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


def safe_int(x: str) -> int | None:
    try:
        return int(x)
    except ValueError:
        return None


if __name__ == "__main__":
    typer.run(main)
