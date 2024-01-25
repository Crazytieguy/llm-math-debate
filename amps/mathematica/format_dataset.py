from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from simplify_mathematica_tex import simplify_tex  # type: ignore


def main():
    formatted_dir = Path("amps/mathematica/formatted")
    formatted_dir.mkdir(exist_ok=True)
    for domain in Path("amps/mathematica/original").iterdir():
        if not domain.is_dir():
            continue
        for problem_class in domain.iterdir():
            if not problem_class.name.endswith("w_steps"):
                continue
            print(domain.name, problem_class.name)
            out_dir = formatted_dir / domain.name / problem_class.name
            out_dir.mkdir(exist_ok=True, parents=True)

            def handle_file(file: Path):
                problem_text = file.read_text()
                try:
                    simplified = simplify_tex(problem_text)
                except ValueError:
                    return
                out_file = out_dir / file.name
                out_file.write_text(simplified)

            with ThreadPoolExecutor(max_workers=16) as executor:
                executor.map(handle_file, problem_class.glob("*.t??"))


if __name__ == "__main__":
    main()
