"""utilities.py"""

import re
import json
from datetime import datetime
from typing import List, Tuple

from core.config import (
    THINK_START,
    THINK_END,
    scores,
)


def parse_reasoning_from_text(input_string: str) -> str:
    """
    Extracts the reasoning text between THINK_START and THINK_END markers.

    Args:
        input_string (str): The input string containing reasoning.

    Returns:
        str: The extracted reasoning text.
    """
    result = []
    capturing = False
    i = 0
    while i < len(input_string):
        if input_string[i : i + len(THINK_START)] == THINK_START:
            capturing = True
            result.append(THINK_START)
            i += len(THINK_START)
        elif input_string[i : i + len(THINK_END)] == THINK_END:
            capturing = False
            result.append(THINK_END)
            i += len(THINK_END)
        elif capturing:
            result.append(input_string[i])
            i += 1
        else:
            i += 1
    cleaned_output = "".join(result)
    return cleaned_output


def remove_think_text(input_string: str) -> str:
    """
    Removes text between THINK_START and THINK_END markers.

    Args:
        input_string (str): The input string to process.

    Returns:
        str: The string with think text removed.
    """
    retval = input_string
    if THINK_START in input_string:
        result = []
        skip = False
        i = 0
        while i < len(input_string):
            if input_string[i : i + len(THINK_START)] == THINK_START:
                skip = True
                i += len(THINK_START)
            elif input_string[i : i + len(THINK_END)] == THINK_END:
                skip = False
                i += len(THINK_END)
            elif not skip:
                result.append(input_string[i])
                i += 1
            else:
                i += 1
        retval = "".join(result).lstrip()
    return retval


def add_score(score: float) -> None:
    """
    Appends a new score to the global scores list and prints scores.

    Args:
        score (float): The score to add.
    """
    scores.append(score)
    print_scores()


def print_scores() -> None:
    """
    Prints the current scores and logs them to a file.
    """
    score_text = ", ".join(map(str, scores))
    print("Scores:", score_text)


def fix_json(json_str: str) -> str:
    """
    Attempts to fix common JSON issues like unbalanced parentheses.

    Args:
        json_str (str): The input JSON string.

    Returns:
        str: The fixed JSON string.
    """

    def balance_parentheses(s: str) -> str:
        open_parens = s.count("(")
        close_parens = s.count(")")
        while open_parens > close_parens:
            s += ")"
            close_parens += 1
        return s

    def fix_dict(d: dict) -> None:
        for key, value in d.items():
            if isinstance(value, str):
                d[key] = balance_parentheses(value)
            elif isinstance(value, list):
                d[key] = [
                    balance_parentheses(item) if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                fix_dict(value)

    # For this implementation, we simply return the input after attempted fixes.
    return json_str


def replace_inner_quotes_in_json_strings(json_string: str) -> str:
    """
    Replaces inner double quotes within JSON strings with single quotes.

    Args:
        json_string (str): The JSON string to process.

    Returns:
        str: The modified JSON string.
    """

    def replace_quotes(match):
        return match.group(0).replace('"', "'")

    modified_json = re.sub(
        r"(?<=: \[.*?)[\[].*?[]](?=.*?\])", replace_quotes, json_string
    )
    return modified_json


def escape_newlines_in_json_strings(json_string: str) -> str:
    """
    Escapes newlines within JSON strings.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with escaped newlines.
    """

    def replace_newlines(match):
        return match.group(0).replace("\n", "\\n")

    pattern = r'(?:"(?:\\.|[^"\\])*")'
    json_string = re.sub(pattern, replace_newlines, json_string)
    return json_string


def analyze_scores(scores2: List[float]) -> Tuple[bool, int]:
    """analyze_scores"""
    if len(scores2) < 2:
        return False, 0
    is_score_worse = scores2[-1] < scores2[-2]
    streak_count = 1
    index = len(scores2) - 2
    while index >= 0 and scores2[index] == scores2[-1]:
        streak_count += 1
        index -= 1
    return is_score_worse, streak_count


def convert_invalid_json_to_valid(input_str: str) -> str:
    """
    Attempts to convert an invalid JSON string into a valid JSON string.

    Args:
        input_str (str): The invalid JSON string.

    Returns:
        str: The valid JSON string or error message.
    """
    input_str = re.sub(r"```json\s*", "", input_str, flags=re.IGNORECASE)
    input_str = re.sub(r"```\s*", "", input_str)
    try:
        input_str = escape_newlines_in_json_strings(input_str)
        input_str = replace_inner_quotes_in_json_strings(input_str)
        input_str = fix_json(input_str)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"convert_invalid_json_to_valid: {e}")
    input_str = input_str.strip()
    try:
        if not input_str.startswith("{"):
            input_str = "{" + input_str
        if not input_str.endswith("}"):
            input_str = input_str + "}"
        input_str = re.sub(
            r'"Critical_Evaluation":\s*', '"Critical_Evaluation": {', input_str, count=1
        )
        input_str = input_str + "}"
        data = json.loads(input_str)
        return json.dumps(data, indent=4)
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"


def get_current_datetime() -> str:
    """
    Returns the current date and time as a formatted string.

    Returns:
        str: The current date and time.
    """
    now = datetime.now()
    formatted_time = now.strftime("%A, %B %d, %Y, %H:%M:%S")
    return f"Current date and time:{formatted_time}"


def debug_json(data, header: str = "JSON Debug Output:") -> None:
    """
    Prints formatted JSON for debugging.

    Args:
        data (str or dict): The data to be debugged.
        header (str): Header message.
    """
    print("~" * 80)
    print(f"\n{header}\n")
    try:
        print(json.dumps(data, indent=4))
    except (TypeError, ValueError):
        try:
            json_obj = json.loads(data)
            print(json.dumps(json_obj, indent=4))
        except (TypeError, json.JSONDecodeError):
            print(str(data).replace("{'role'", "\n\n\n{'role'"))
    print("\n" + "~" * 80 + "\n")
