"""llm_helpers.py"""

import time
from typing import Tuple

from openai import OpenAI

from core.client_factory import get_client
from core.utilities import remove_think_text, get_current_datetime
from core.pricing import get_model_pricing
from core.config import (
    lock,
    Service,
    LLM_RETRY_WAIT_TIME,
    LLM_RETRY_COUNT,
    GRAND_TOTAL_COST,
    OPENAI_API_KEY,
    OPENAI_GPT_4O,
)


def process_and_store_message(tmp_messages, retval_text):
    """
    Processes the input and output text, prints them to the console,
    and writes them to a file atomically using a lock.

    Args:
    - tmp_messages (dict): Messages to be processed.
    - retval_text (str): The output text to be processed.
    """

    last_message = str(tmp_messages.copy().pop()) if len(tmp_messages) > 0 else None

    # Print input and output to console
    print("INPUT")
    print(">" * 100)
    print(last_message)
    print(">" * 100)
    print("OUTPUT")
    print("<" * 100)
    print(retval_text)
    print("<" * 100)

    # Write input and output to file
    with lock:
        try:
            with open(
                "deep_research_intermediate.txt", "a", encoding="utf-8"
            ) as output_file:
                output_file.write("INPUT")
                output_file.write(">" * 100)
                output_file.write(last_message)
                output_file.write(">" * 100)
                output_file.write("OUTPUT")
                output_file.write("<" * 100)
                output_file.write(retval_text)
                output_file.write("<" * 100)
        except IOError:
            print("An error occurred while writing to the file.")


def call_llm_api_with_retry(
    client,
    args: dict,
    retry_count: int = LLM_RETRY_COUNT,
    retry_wait_time: int = LLM_RETRY_WAIT_TIME,
):
    """
    Calls the LLM API with retries if necessary.

    Args:
        client: The LLM client instance.
        args (dict): Arguments for the API call.
        retry_count (int): Maximum number of retries.
        retry_wait_time (int): Wait time between retries.

    Returns:
        Response object from the API.
    """
    llm_call_count = 0
    while llm_call_count < retry_count:
        try:
            llm_call_count += 1
            if llm_call_count > 1:
                print(f"LLM CALL TRY {llm_call_count} of {retry_count}")
            response = client.chat.completions.create(**args)
            return response
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error calling LLM, waiting {retry_wait_time} seconds: {e}")
            if llm_call_count < retry_count:
                time.sleep(retry_wait_time)
            else:
                raise


def call_llm(
    prompt: str,
    model: str,
    USE_SERVICE: Service,  # pylint:disable=invalid-name
    message_prefix: str = None,
    messages: list = None,
) -> str:
    """
    Calls an LLM with the given prompt and messages.

    Args:
        prompt (str): The prompt text.
        model (str): The model identifier.
        USE_SERVICE (Service): The service to use.
        message_prefix (str, optional): A prefix message for logging.
        messages (list, optional): A list of message dicts.

    Returns:
        str: The assistant's response text.
    """
    tmp_messages = []
    retval_text = ""

    if messages is None:
        tmp_messages = [
            {"role": "user", "content": get_current_datetime() + "\n" + prompt}
        ]
    else:
        tmp_messages = messages.copy()
        tmp_messages.append({"role": "user", "content": prompt})
    client = get_client(USE_SERVICE)
    args = {
        "model": model,
        "messages": tmp_messages,
    }

    try:
        retval = call_llm_api_with_retry(client, args)
        if message_prefix:
            print(f"{message_prefix} {USE_SERVICE} {model}")
        print_token_usage_details(retval, USE_SERVICE, model)
        retval_text = retval.choices[0].message.content
    except Exception as e:  # pylint: disable=broad-exception-caught
        retval_text = f"Error calling LLM model='{model}': {str(e)}"

    process_and_store_message(tmp_messages, retval_text)
    return retval_text


def call_openai(prompt: str, model: str = None, messages: list = None) -> str:
    """
    Calls OpenAI's chat API.

    Args:
        prompt (str): The prompt text.
        model (str, optional): The model to use.
        messages (list, optional): List of message dicts.

    Returns:
        str: The assistant's response text.
    """

    tmp_messages = []
    retval_text = ""

    if messages is None:
        tmp_messages = [
            {"role": "user", "content": get_current_datetime() + "\n" + prompt}
        ]
    else:
        tmp_messages = messages.copy()
        tmp_messages.append({"role": "user", "content": prompt})

    try:
        openai_client = OpenAI()
        openai_client.api_key = OPENAI_API_KEY
        completion = openai_client.chat.completions.create(
            model=model if model else OPENAI_GPT_4O, messages=tmp_messages
        )
        print_token_usage_details(
            completion, Service.OPENAI, model if model else OPENAI_GPT_4O
        )
        retval_text = completion.choices[0].message.content
        retval_text = remove_think_text(retval_text)
    except Exception as e:  # pylint: disable=broad-exception-caught
        retval_text = f"Error calling LLM model='{model}': {str(e)}"

    process_and_store_message(tmp_messages, retval_text)
    return retval_text


def print_token_usage_details(
    response,
    service: Service,
    model: str,
    perplexity_content_size: str = None,
    print_row: bool = True,
) -> None:
    """
    Logs token usage details and pricing information.

    Args:
        response: The LLM response object.
        service (Service): The service used.
        model (str): The model identifier.
        perplexity_content_size (str, optional): Perplexity content size.
        print_row (bool, optional): If True, prints a summary row.
    """
    usage = getattr(response, "usage", None)
    input_price, output_price, additional = get_model_pricing(
        service, model, perplexity_content_size
    )
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", "N/A")
        completion_tokens = getattr(usage, "completion_tokens", "N/A")
        input_cost = (
            (prompt_tokens / 1000000) * input_price
            if isinstance(prompt_tokens, (int, float))
            else 0
        )
        output_cost = (
            (completion_tokens / 1000000) * output_price
            if isinstance(completion_tokens, (int, float))
            else 0
        )
        global GRAND_TOTAL_COST  # pylint: disable=global-statement
        GRAND_TOTAL_COST += input_cost + output_cost + additional
        if print_row:
            print(
                f"USAGE... Prompt {prompt_tokens} -> ${input_cost:.6f} "
                f"Completion {completion_tokens} -> ${output_cost:.6f} "
                f"Tools -> ${additional:.6f} "
                f"Grand_Total ${GRAND_TOTAL_COST:.6f}"
            )


def check_tokens_exceeded(
    is_final_answer: bool, messages: list, question: str
) -> Tuple[bool, list]:
    """
    Checks if token limit is exceeded and forces a final answer if needed.

    Args:
        is_final_answer (bool): Indicates if final answer is reached.
        messages (list): Current messages.
        question (str): The user's question.

    Returns:
        Tuple[bool, list]: Updated is_final_answer flag and messages list.
    """
    is_final_answer = True
    print("*" * 80)
    print("*" * 80)
    print("ABORTING... SHORTCUT TO FINAL ANSWER DUE TO CONTEXT LENGTH")
    print("*" * 80)
    print("*" * 80)
    messages.append(
        {
            "role": "user",
            "content": (
                f"""
            Write your long long long final answer to the user's question without missing any detail. 
            Response must be text, not JSON.\n\nUser's Question\n\n{question}
            """
            ),
        }
    )
    return is_final_answer, messages
