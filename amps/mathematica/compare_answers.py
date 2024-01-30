import os
import re
from enum import Enum, auto

import httpx

WOLFRAMALPHA_APP_ID = os.environ["WOLFRAMALPHA_APP_ID"]
CLIENT = httpx.AsyncClient(timeout=6.5)


class AnswerComparison(Enum):
    """Enum representing the result of comparing two answers."""

    SAME = auto()
    SAME_VALUE_SIMPLIFIED_DIFFERENTLY = auto()
    DIFFERENT = auto()


async def compare_answers(a: str, b: str) -> AnswerComparison:
    """Compares two answers."""
    a = extract_answer(a)
    b = extract_answer(b)
    if not await answers_same_wolframalpha(a, b):
        return AnswerComparison.DIFFERENT
    if have_same_integers(a, b):
        return AnswerComparison.SAME
    return AnswerComparison.SAME_VALUE_SIMPLIFIED_DIFFERENTLY


def extract_answer(solution: str) -> str:
    """Extracts the answer from a solution string."""
    try:
        answer = solution.split("\n\nAnswer: ")[1].strip()
    except IndexError:
        raise ValueError("No 'Answer' found in solution string.")
    if answer[0] != "$":
        raise ValueError("Answer does not start with '$'.")
    if answer[-1] != "$":
        raise ValueError("Answer does not end with '$'.")
    answer = answer[1:-1]
    answer = answer.split("\\approx")[0].strip()
    if "=" in answer:
        answer = answer.split("=")[1].strip()
    return answer


INTEGER_RE = re.compile(r"\d+")


def have_same_integers(a: str, b: str) -> bool:
    """Returns True if a and b have the same integers.
    Proxy for the solutions not having different simplifications."""
    a_integers = set(INTEGER_RE.findall(a))
    b_integers = set(INTEGER_RE.findall(b))
    return a_integers == b_integers


async def answers_same_wolframalpha(a: str, b: str) -> bool:
    """Returns True if a and b are the same according to WolframAlpha."""
    params = {
        "appid": WOLFRAMALPHA_APP_ID,
        "input": f"{a} = {b}",
        "output": "json",
    }
    response = await CLIENT.get("https://api.wolframalpha.com/v2/query", params=params)
    response.raise_for_status()
    data = response.json()
    queryresult = data["queryresult"]
    if not queryresult["success"]:
        raise ValueError("WolframAlpha query failed.")
    pods = queryresult["pods"]
    try:
        result_pod = [pod for pod in pods if pod["id"] == "Result"][0]
    except IndexError:
        raise ValueError("No 'Result' pod found.")
    subpods = result_pod["subpods"]
    try:
        subpod = subpods[0]
    except IndexError:
        raise ValueError("No subpod found.")
    plaintext = subpod["plaintext"]
    if plaintext == "True":
        return True
    if plaintext == "False":
        return False
    raise ValueError(f"Unexpected plaintext from WolframAlpha: {plaintext}")
