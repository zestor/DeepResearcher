"""reasoning.py"""

from typing import Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from core.tools_util import parse_tool_calls_from_text
from core.web_services import web_search
from core.llm_helpers import call_openai, call_llm
from core.utilities import remove_think_text, parse_reasoning_from_text
from core.config import (
    Service,
    DEEPSEEK_R1,
    GROQ_DEEPSEEK_R1_LLAMA,
    GROQ_QWEN_QWQ,
    GROQ_LLAMA_3_VERSATILE,
    TOGETHER_LLAMA_4_MAVERICK,
    FIREWORKS_LLAMA_4_MAVERICK,
    OPENAI_O3_MINI,
    OPENAI_O1_MINI,
    OPENAI_O1,
    MAX_TOOL_PARALLEL_THREADS,
    MAX_QUESTIONS_FOR_REASONING_EXPANSION,
    OPENAI_USE_MODEL_EXPANDED_REASONING,
    USE_SERVICE_EXPANDED_REASONING,
    LLM_USE_MODEL_EXPANDED_REASONING,
    research_questions,
)


def get_model_args(
    model_version: str, USE_SERVICE: Service, tools: list = None
):  # pylint: disable=invalid-name
    """
    Returns token limits and model arguments specific to the model.

    Args:
        model_version (str): The model identifier.
        USE_SERVICE (Service): The service being used.
        tools (list, optional): List of tool definitions.

    Returns:
        Tuple(int, dict): Maximum prompt tokens and the model arguments.
    """
    MAX_PROMPT_TOKENS = 60000
    model_args = {"model": model_version}
    if model_version == GROQ_DEEPSEEK_R1_LLAMA:
        if USE_SERVICE != Service.GROQ:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update({"temperature": 0.6, "max_completion_tokens": 131072})
    elif model_version == GROQ_LLAMA_3_VERSATILE:
        if USE_SERVICE != Service.GROQ:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update({"max_completion_tokens": 32768})
    elif model_version == "deepseek-ai/DeepSeek-V3":
        if USE_SERVICE != Service.TOGETHER:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update({"max_completion_tokens": 12288})
    elif model_version == FIREWORKS_LLAMA_4_MAVERICK:
        if USE_SERVICE != Service.TOGETHER:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update({"max_completion_tokens": 16384})
    elif model_version == "accounts/fireworks/models/llama4-scout-instruct-basic":
        if USE_SERVICE != Service.FIREWORKS:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update({"max_completion_tokens": 16384})
    elif model_version == TOGETHER_LLAMA_4_MAVERICK:
        if USE_SERVICE == Service.TOGETHER:
            model_args.update({"max_completion_tokens": 524000 - MAX_PROMPT_TOKENS})
        elif USE_SERVICE == Service.GROQ:
            model_args.update({"max_completion_tokens": 8192})
        else:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
    elif model_version == GROQ_QWEN_QWQ:
        if USE_SERVICE != Service.GROQ:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update({"max_completion_tokens": 32768, "temperature": 0.6})
    elif model_version in (OPENAI_O1, OPENAI_O3_MINI):
        if USE_SERVICE != Service.OPENAI:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update(
            {
                "tools": tools,
                "reasoning_effort": "high",
                "max_completion_tokens": 100000,
                "response_format": {"type": "text"},
            }
        )
    elif model_version == OPENAI_O1_MINI:
        if USE_SERVICE != Service.OPENAI:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update(
            {"max_completion_tokens": 65536, "response_format": {"type": "text"}}
        )
    elif model_version == DEEPSEEK_R1:
        if USE_SERVICE != Service.DEEPSEEK:
            raise ValueError(
                f"Unsupported service: {USE_SERVICE} model: {model_version}"
            )
        model_args.update({"max_tokens": 8192, "temperature": 1.5, "stream": False})
    else:
        raise ValueError(f"Unsupported service: {USE_SERVICE} model: {model_version}")
    return MAX_PROMPT_TOKENS, model_args


def get_reasoning_tools_and_messages(
    model_version: str, messages: list, msg, assistant_content: str, response
) -> Tuple[list, list, str, str]:
    """
    Extracts tool calls and reasoning content from the assistant's message.

    Args:
        model_version (str): The model identifier.
        messages (list): The message stack.
        msg: The message object.
        assistant_content (str): The assistant's content.
        response: The full response object.

    Returns:
        Tuple: (tool_calls, messages, assistant_content, reasoning_content)
    """
    reasoning_content = None

    def append_message(content, role="assistant"):
        if content:
            messages.append({"role": role, "content": content})

    if model_version in [OPENAI_O1, OPENAI_O3_MINI]:
        tool_calls = getattr(msg, "tool_calls", None)
        messages.append(msg)
    elif model_version == OPENAI_O1_MINI:
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(assistant_content)
    elif model_version == DEEPSEEK_R1:
        reasoning_content = getattr(
            response.choices[0].message, "reasoning_content", None
        )
        if not reasoning_content:

            reasoning_content = parse_reasoning_from_text(assistant_content)
            assistant_content = remove_think_text(assistant_content)
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(reasoning_content)
        append_message(assistant_content)
    elif model_version in [GROQ_QWEN_QWQ, GROQ_DEEPSEEK_R1_LLAMA]:

        reasoning_content = parse_reasoning_from_text(assistant_content)
        assistant_content = remove_think_text(assistant_content)
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(reasoning_content)
        append_message(assistant_content)
    else:
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(assistant_content)
    return tool_calls, messages, assistant_content, reasoning_content


def expand_reasoning(reasoning_content, tool_calls, messages):
    """expand_reasoning"""
    if reasoning_content:
        if "Critical_Evaluation" not in reasoning_content:
            print("*" * 80)
            print("Using Reasoning Expansion")

            prompt = f"""
            In a moment but not now, pre-read the reasoning text below, 
            from the reasoning text identify the most important unique 
            questions (no more than {MAX_QUESTIONS_FOR_REASONING_EXPANSION})
            which were obscurely asked but never answered. For each question 
            identified, respond on a single line with the question and it's 
            context identified with Context:. Response must not include additional 
            formatting, numbering, bullets, introduction, commentary, or conclusion. 
            All I need is a list of questions, one per line, with the associated 
            context on the same line.
            
            ```reasoning text
            {reasoning_content}
            ```
            """

            if USE_SERVICE_EXPANDED_REASONING == Service.OPENAI:
                llm_question_response = call_openai(
                    prompt, OPENAI_USE_MODEL_EXPANDED_REASONING
                )
            else:
                llm_question_response = call_llm(
                    prompt,
                    LLM_USE_MODEL_EXPANDED_REASONING,
                    USE_SERVICE_EXPANDED_REASONING,
                    "***** EXPANDED REASONING *****",
                )

            prompt = f"""
            In a moment but not now, pre-read the proposed_questions, tool_calls, and prior_questions below, then consolidate the questions which are similar in proposed_questions. Eliminate any question from proposed_questions which is similar to a question in tool_calls or prior_questions. For each question identified, respond on a single line with the question and it's context identified with Context:. Response must not include additional formatting, numbering, bullets, introduction, commentary, or conclusion. All I need is a list of questions, one per line, with the associated context on the same line.

            ```proposed_questions
            {llm_question_response}
            ```

            ```tool_calls
            {tool_calls}
            ```

            ```prior_questions
            {research_questions}
            ```
            """
            llm_question_response2 = call_openai(
                prompt, OPENAI_USE_MODEL_EXPANDED_REASONING
            )
            questions = llm_question_response2.splitlines()
            # eliminate empty questions
            questions = [s for s in questions if s]

            research_questions.extend(s for s in questions if s)

            print("*" * 80)
            print(f"Additional Questions:\n\n{llm_question_response2}")
            print("*" * 80)

            def process_question(question):
                question = question.strip()
                if question:
                    print("*" * 80)
                    print(f"Question: {question}")

                    additional_answer = web_search(question)
                    additiona_QnA = f"""
                    Trusted Research to Question:
                    {question}
                    
                    Answer:
                    {additional_answer}
                    """

                    print("*" * 80)
                    print(additiona_QnA)

                    return {"role": "user", "content": additiona_QnA}
                return None

            with ThreadPoolExecutor(max_workers=MAX_TOOL_PARALLEL_THREADS) as executor:
                future_to_question = {
                    executor.submit(process_question, question): question
                    for question in questions
                }
                for future in as_completed(future_to_question):
                    result = future.result()
                    if result:
                        messages.append(result)

    return messages
