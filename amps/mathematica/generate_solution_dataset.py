from pathlib import Path

import typer

from .pydantic_models import Solution, parse_solution_file


def generate_solution_dataset(amount_per_class: int = 100, append: bool = True):
    relevant_classes_dir = Path("amps/mathematica/error_insertion_examples")
    domain_classes = [
        (domain.name, problem_class.name)
        for domain in relevant_classes_dir.iterdir()
        for problem_class in domain.iterdir()
    ]
    dataset_path = Path("amps/mathematica/solution_dataset.jsonl")
    solutions_by_domain_class = {
        (domain, problem_class): [] for domain, problem_class in domain_classes
    }
    if append and dataset_path.exists():
        data = [
            Solution.model_validate_json(line)
            for line in dataset_path.read_text().splitlines()
        ]
        for solution in data:
            try:
                solutions_by_domain_class[
                    (solution.domain, solution.problem_class)
                ].append(solution)
            except KeyError as e:
                e.add_note("dataset already has solutions for an irrelevant class")
                raise e
    for (domain, problem_class), solutions in solutions_by_domain_class.items():
        skipped = 0
        while len(solutions) < amount_per_class:
            source_file = Path(
                f"amps/mathematica/formatted/{domain}/{problem_class}_w_steps/{len(solutions) + skipped}.txt"
            )
            if not source_file.exists():
                skipped += 1
                continue
            solutions.append(parse_solution_file(domain, problem_class, source_file))
    with dataset_path.open("w") as f:
        for one_of_each in zip(*solutions_by_domain_class.values()):
            for solution in one_of_each:
                f.write(solution.model_dump_json() + "\n")


if __name__ == "__main__":
    typer.run(generate_solution_dataset)
