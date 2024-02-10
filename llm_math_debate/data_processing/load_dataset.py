from datasets import DatasetDict, load_dataset


def load_and_split_dataset(path: str) -> DatasetDict:
    return load_dataset("json", data_files=path, split="train").train_test_split(  # type: ignore
        test_size=0.2, seed=42
    )
