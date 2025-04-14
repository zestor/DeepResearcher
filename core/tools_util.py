"""tools_util.py"""

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import (
    MAX_TOOL_PARALLEL_THREADS,
    lock,
)
from core.web_services import web_search, call_web_content_retriever
from core.llm_helpers import call_openai


def parse_tool_calls_from_text(assistant_content: str):
    """
    Parses tool call instructions from assistant's content. Expects JSON blocks.

    Args:
        assistant_content (str): The content of the assistant message.

    Returns:
        list: A list of dictionaries representing tool calls.
    """
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    blocks = re.findall(pattern, assistant_content, flags=re.DOTALL)
    tool_calls = []
    for block in blocks:
        try:
            data = json.loads(block)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                tool_calls.append(
                    {
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data["arguments"]),
                        }
                    }
                )
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return tool_calls


def compress_messages_to_single_user_message(messages: list) -> list:
    """
    Compresses multiple messages into a single user message string.

    Args:
        messages (list): A list of message dictionaries.

    Returns:
        list: A list with a single message dict.
    """
    formatted_output = ""
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        formatted_output += f"\n=====\n[{role.upper()}]:\n=====\n{content}\n\n"
    return [{"role": "user", "content": formatted_output}]


def process_single_tool_call(tc: dict, model_version: str) -> dict:
    """
    Processes a single tool call and dispatches to the correct function.

    Args:
        tc (dict): The tool call dictionary.
        model_version (str): The model version identifier.

    Returns:
        dict: The resulting message from the tool call.
    """
    func_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
    arguments_json = (
        tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments
    )
    print(f"***** TOOL CALL ***** Tool name: {func_name} Arguments: {arguments_json}")
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        arguments = {}
    if func_name == "web_search":
        query = arguments.get("query", "")
        result = web_search(query)
        result = f"Tool Response to query '{query}': {result}"
    elif func_name == "call_web_content_retriever":
        url = arguments.get("url", "")
        result = call_web_content_retriever(url)
    elif func_name == "call_research_professional":
        subprompt = arguments.get("prompt", "")
        result = call_openai(subprompt)
    elif func_name == "call_openai":
        subprompt = arguments.get("prompt", "")
        result = call_openai(subprompt)
    else:
        result = f"Tool {func_name} is not implemented."
    tool_role = (
        "tool" if model_version in [] else "user"
    )  # MODELS_WITH_TOOL_USAGE handled in llm_helpers
    tool_result_message = {"role": tool_role, "content": result}
    if model_version in []:  # if needed add extra keys
        tool_result_message["tool_response"] = func_name
        if isinstance(tc, dict) and "id" in tc:
            tool_result_message["tool_call_id"] = tc["id"]
    with lock:
        try:
            with open(
                "deep_research_intermediate.txt", "a", encoding="utf-8"
            ) as output_file:
                output_file.write(f"{result}\n")
                output_file.write("=" * 80 + "\n")
        except IOError:
            print("An error occurred while writing to the file.")
    return tool_result_message


def process_tool_calls(messages: list, tool_calls: list, model_version: str) -> list:
    """
    Processes multiple tool calls concurrently.

    Args:
        messages (list): List of current messages.
        tool_calls (list): List of tool call dictionaries.
        model_version (str): The model version identifier.

    Returns:
        list: Updated list of messages with tool call responses.
    """
    with ThreadPoolExecutor(max_workers=MAX_TOOL_PARALLEL_THREADS) as executor:
        future_to_tool_call = {
            executor.submit(process_single_tool_call, tc, model_version): tc
            for tc in tool_calls
        }
        for future in as_completed(future_to_tool_call):
            tool_result_message = future.result()
            with lock:
                messages.append(tool_result_message)
    return messages
