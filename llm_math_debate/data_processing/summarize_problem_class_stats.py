import pandas as pd

df = pd.read_csv("amps/mathematica/problem_stats.csv")

df.drop(columns=["problem"]).groupby(["domain", "problem_class"]).aggregate(
    mean_steps=("num_steps", "mean"),
    max_steps=("num_steps", "max"),
    mean_tokens=("num_tokens", "mean"),
    max_tokens=("num_tokens", "max"),
).to_csv("amps/mathematica/problem_class_stats.csv")
