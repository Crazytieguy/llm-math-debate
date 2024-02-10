from pathlib import Path

from pydantic import BaseModel


class SolutionError(BaseModel):
    explanation: str | None
    note: str | None
    step: int
    original_solution: str


class Solution(BaseModel):
    domain: str
    problem_class: str
    source_file: str
    problem: str
    solution: str
    steps: int
    solution_error: SolutionError | None = None


def parse_solution_file(domain: str, problem_class: str, source_file: Path) -> Solution:
    file_content = source_file.read_text()
    problem, solution = file_content.split("\n\n\\hrule\n\n")
    steps = [
        step
        for line in solution.splitlines()
        if (step := safe_int(line.split(". ")[0])) is not None
    ][-1]
    return Solution(
        domain=domain,
        problem_class=problem_class,
        source_file=source_file.name,
        problem=problem,
        solution=solution,
        steps=steps,
    )


def safe_int(x: str) -> int | None:
    try:
        return int(x)
    except ValueError:
        return None
