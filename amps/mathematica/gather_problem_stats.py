import csv
from pathlib import Path

import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4")  # just to get an estimate

csv_file = open("amps/mathematica/problem_stats.csv", "w")
csv_writer = csv.DictWriter(
    csv_file,
    fieldnames=["domain", "problem_class", "problem", "num_steps", "num_tokens"],
)
csv_writer.writeheader()

# Note - this only works before formatting
identify_step = """\\end{array}
 \\\\
 """

for domain in Path("amps/mathematica").iterdir():
    if not domain.is_dir():
        continue
    for problem_class in domain.iterdir():
        if not problem_class.name.endswith("w_steps"):
            continue
        print(domain.name, problem_class.name)
        for problem in problem_class.glob("*.txt"):
            problem_text = problem.read_text()
            num_steps = problem_text.count(identify_step) + 1
            num_tokens = len(tokenizer.encode(problem_text))
            csv_writer.writerow(
                {
                    "domain": domain.name,
                    "problem_class": problem_class.name.removesuffix("_w_steps"),
                    "problem": problem.name.removesuffix(".txt"),
                    "num_steps": num_steps,
                    "num_tokens": num_tokens,
                }
            )

csv_file.close()
