"""research_professional.py"""

import json

from core.client_factory import get_client
from core.utilities import get_current_datetime, analyze_scores
from core.llm_helpers import (
    call_llm_api_with_retry,
    call_llm,
    call_openai,
    print_token_usage_details,
    check_tokens_exceeded,
)
from core.tools_util import process_tool_calls, compress_messages_to_single_user_message
from core.prompt_helpers import score_answer, get_manager_feedback
from core.reasoning import (
    get_model_args,
    get_reasoning_tools_and_messages,
    expand_reasoning,
)
from core.config import (
    Service,
    scores,
    USE_SERVICE_REASONING,
    DEEPSEEK_USE_MODEL,
    GROQ_USE_MODEL,
    OPENAI_USE_MODEL,
    TOGETHER_USE_MODEL,
    USE_SERVICE_SUMMARY,
    OPENAI_USE_MODEL_SUMMARY,
    LLM_USE_MODEL_SUMMARY,
    USE_MULTI_ROUND_TEST_TIME_SCALING,
    MAX_TRIES_FOR_TEST_TIME_SCALING,
    USE_REASONING_EXPANSION,
    MAX_TRIES_TO_INCREASE_SCORE,
    MODELS_WITH_TOOL_USAGE,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Use this to utilize a PhD grad student to perform research, "
                "they can only research one single intent question at a time, "
                "they have no context or prior knowledge of this conversation, "
                "you must give them the context and a single intention query. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A straight to the point concise succint question or search query to be sent to research assistant",  # pylint:disable=line-too-long
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def call_research_professional2(
    question: str, prompt: str, model_version: str = OPENAI_USE_MODEL
) -> str:
    """
    Conducts a full reasoning loop to generate a detailed investigative report.

    Args:
        question (str): The user's question.
        prompt (str): The initial prompt.
        model_version (str): The model version to use.

    Returns:
        str: The final answer or report.
    """
    if USE_SERVICE_REASONING == get_client.__globals__["Service"].DEEPSEEK:
        model_version = DEEPSEEK_USE_MODEL
    if USE_SERVICE_REASONING == get_client.__globals__["Service"].GROQ:
        model_version = GROQ_USE_MODEL
    if USE_SERVICE_REASONING == get_client.__globals__["Service"].TOGETHER:
        model_version = TOGETHER_USE_MODEL

    is_final_answer = False
    messages = []
    system_message = """
    I've upgraded your functionality post training data cutoff to access current data and use tools. 
    You can now request tool executions using JSON in your responses, and I'll run them and return the results in user messages. 
    You are able to call tools just by telling me you want to run a tool by responding with JSON as described below. 
    You are a helpful AI that can use the following tools by producing JSON in your message. 
    To call multiple tools (max 5), output multiple JSON blocks (in triple backticks, with a line ```json) in a single response.
    When you want to call a tool, output EXACTLY a JSON block (in triple backticks, with a line ```json) of the form:

    {
        "name": "<tool_name>",
        "arguments": { "name" : "value", "name":"value", ... }
    }

    The valid tools are:
    """
    system_message += f"""
    {json.dumps(tools, indent=4, ensure_ascii=False)}
    """
    system_message += """
    Return your final answer in plain text (no JSON) once you have all information you need.
    Do not output extraneous text outside or after a JSON block if calling a tool.
    """
    messages.append({"role": "user", "content": system_message})
    messages.append({"role": "user", "content": get_current_datetime() + "\n" + prompt})
    counter_for_multi_round_test_time_scaling = 0
    for _ in range(100):
        if USE_SERVICE_REASONING == get_client.__globals__["Service"].DEEPSEEK:
            base_args = {"messages": compress_messages_to_single_user_message(messages)}
        else:
            base_args = {"messages": messages}
        client = get_client(USE_SERVICE_REASONING)
        MAX_PROMPT_TOKENS, model_args = get_model_args(  # pylint: disable=invalid-name
            model_version, USE_SERVICE_REASONING, tools=[]
        )
        args = {**base_args, **model_args}
        print(f"***** REASONING LOOP ***** {USE_SERVICE_REASONING} {model_version}")
        response = call_llm_api_with_retry(client, args)
        msg = response.choices[0].message
        assistant_content = msg.content
        print("\n" + ">" * 100)
        print(assistant_content)
        print("<" * 100 + "\n")
        finish_reason = response.choices[0].finish_reason
        if response.usage.prompt_tokens > MAX_PROMPT_TOKENS and not is_final_answer:
            is_final_answer, messages = check_tokens_exceeded(
                is_final_answer, messages, question
            )
            continue
        tool_calls, messages, assistant_content, reasoning_content = (
            get_reasoning_tools_and_messages(
                model_version, messages, msg, assistant_content, response
            )
        )

        if USE_REASONING_EXPANSION:
            messages = expand_reasoning(reasoning_content, tool_calls, messages)

        # debug_json(messages,"Message Stack After:")

        print_token_usage_details(response, USE_SERVICE_REASONING, model_version)

        if tool_calls:
            messages = process_tool_calls(messages, tool_calls, model_version)
            continue
        if finish_reason == "stop":
            if USE_MULTI_ROUND_TEST_TIME_SCALING:
                if counter_for_multi_round_test_time_scaling == 0:
                    counter_for_multi_round_test_time_scaling += 1
                    messages.pop()
                    revise_prompt = f"""
                    Assistant’s previous answer is: 
                    <answer>{assistant_content}</answer>, and please re-answer.
                    """
                    messages.append({"role": "user", "content": revise_prompt})
                    continue
                elif (
                    counter_for_multi_round_test_time_scaling
                    <= MAX_TRIES_FOR_TEST_TIME_SCALING
                ):
                    counter_for_multi_round_test_time_scaling += 1
                    messages.pop()
                    messages.pop()
                    revise_prompt = f"""
                    The assistant’s previous answer is: 
                    <answer>{assistant_content}</answer>, and please re-answer.
                    """
                    messages.append({"role": "user", "content": revise_prompt})
                    continue
                else:
                    messages.pop(-2)
                    counter_for_multi_round_test_time_scaling = 0
            is_pass_threshold, scoring_pros_cons = score_answer(question, messages)
            if is_pass_threshold:
                prompt_final = f"""
                I conduct thorough research to create detailed investigative reports. 
                Based on our conversation, here is your final comprehensive narrative report.
                User's Question:\n{question}"""

                print("*" * 50)
                print("***** PRODUCING FINAL ANSWER *****")
                print("*" * 50)

                if USE_SERVICE_SUMMARY == get_client.__globals__["Service"].OPENAI:
                    final_answer = call_openai(
                        prompt_final, OPENAI_USE_MODEL_SUMMARY, messages
                    )
                else:
                    final_answer = call_llm(
                        prompt_final,
                        LLM_USE_MODEL_SUMMARY,
                        USE_SERVICE_SUMMARY,
                        None,
                        messages,
                    )
                return final_answer
            else:
                manager_feedback = get_manager_feedback(question, assistant_content)
                revise_prompt = f"""
                Use tools and revise response based on your Manager's feedback:{manager_feedback}
                {scoring_pros_cons}
                """
                messages.append({"role": "user", "content": revise_prompt})
                continue
        elif finish_reason in ["length", "max_tokens", "content_filter"]:
            print("The model's response ended due to finish_reason =", finish_reason)
            break
        if assistant_content.strip():
            print("\nAssistant:\n" + assistant_content)
            return assistant_content
    return "Lacked sufficient details to complete request."


def call_research_professional(
    question: str, prompt: str, model_version: str = OPENAI_USE_MODEL
) -> str:
    """
    Calls reasoning LLM
    """

    if USE_SERVICE_REASONING == get_client.__globals__["Service"].DEEPSEEK:
        model_version = DEEPSEEK_USE_MODEL
    if USE_SERVICE_REASONING == get_client.__globals__["Service"].GROQ:
        model_version = GROQ_USE_MODEL
    if USE_SERVICE_REASONING == get_client.__globals__["Service"].TOGETHER:
        model_version = TOGETHER_USE_MODEL

    is_final_answer = False

    messages = []

    if model_version not in MODELS_WITH_TOOL_USAGE:
        system_message = (
            "I've upgraded your functionality post training data cutoff to access current data and use tools. You can now request tool "
            "executions using JSON in your responses, and I'll run them and return the results in user messages. "
            "You are able to call tools just by telling me you want to run a tool by responding with JSON as described below. "
            "You are a helpful AI that can use the following tools by producing JSON in your message. "
            "To call multiple tools (max 10), output multiple JSON blocks (in triple backticks, with a line ```json) in a single response."
            "When you want to call a tool, output EXACTLY a JSON block (in triple backticks, with a line ```json) "
            "of the form:\n\n"
            "{\n"
            '  "name": "<tool_name>",\n'
            '  "arguments": { "name" : "value", "name":"value", ... }\n'
            "}\n\n"
            "The valid tools are:\n"
            f"{json.dumps(tools, indent=4, ensure_ascii=False)}"
            "Return your final answer in plain text (no JSON) once you have all information you need. "
            "Do not output extraneous text outside or after a JSON block if calling a tool."
        )
        messages.append({"role": "user", "content": system_message})
    messages.append({"role": "user", "content": get_current_datetime() + "\n" + prompt})

    llm_call_count_to_increase_score = 0
    # https://arxiv.org/pdf/2503.19855
    counter_for_multi_round_test_time_scaling = 0

    # Main ReAct loop
    for _ in range(100):

        # debug_json(messages,"Message Stack Before:")

        # for DeepSeek they don't support multiple messages
        # need to create a big string with user/assistant messages
        # and set as single user message
        if USE_SERVICE_REASONING == Service.DEEPSEEK:
            base_args = {
                "messages": compress_messages_to_single_user_message(messages),
            }
        else:
            base_args = {
                "messages": messages,
            }

        client = get_client(USE_SERVICE_REASONING)

        MAX_PROMPT_TOKENS, model_args = get_model_args(  # pylint: disable=invalid-name
            model_version, USE_SERVICE_REASONING, tools
        )

        # Merge common and model-specific settings
        args = {**base_args, **model_args}

        print(f"***** REASONING LOOP ***** {USE_SERVICE_REASONING} {model_version}")
        response = call_llm_api_with_retry(client, args)

        # debug_json(response, "Message Received")

        msg = response.choices[0].message

        assistant_content = msg.content

        print("\n\n")
        print(">" * 100)
        print(f"{assistant_content}")
        print("<" * 100)
        print("\n\n")

        finish_reason = response.choices[0].finish_reason

        if response.usage.prompt_tokens > MAX_PROMPT_TOKENS and not is_final_answer:
            is_final_answer, messages = check_tokens_exceeded(
                is_final_answer, messages, question
            )
            continue

        tool_calls, messages, assistant_content, reasoning_content = (
            get_reasoning_tools_and_messages(
                model_version, messages, msg, assistant_content, response
            )
        )

        if USE_REASONING_EXPANSION:
            messages = expand_reasoning(reasoning_content, tool_calls, messages)

        # debug_json(messages,"Message Stack After:")

        print_token_usage_details(response, USE_SERVICE_REASONING, model_version)

        # If there are tool calls, handle them
        if tool_calls:
            messages = process_tool_calls(messages, tool_calls, model_version)
            # After tool calls, continue loop so the model sees the new tool outputs
            continue

        # If no tool calls, check finish_reason
        if finish_reason == "stop":

            if USE_MULTI_ROUND_TEST_TIME_SCALING:
                #####################
                # Think Twice: Enhancing LLM Reasoning
                # by Scaling Multi-round Test-time Thinking
                # https://arxiv.org/pdf/2503.19855
                #####################
                if counter_for_multi_round_test_time_scaling == 0:
                    counter_for_multi_round_test_time_scaling += 1
                    # remove llm response with <think> and answer
                    messages.pop()
                    # replace it with just the answer
                    revise_prompt = f"The assistant’s previous answer is: <answer>{assistant_content}</answer>, and please re-answer."
                    messages.append({"role": "user", "content": revise_prompt})
                    continue
                elif (
                    counter_for_multi_round_test_time_scaling
                    <= MAX_TRIES_FOR_TEST_TIME_SCALING
                ):
                    counter_for_multi_round_test_time_scaling += 1
                    # remove re-answer msg from prior step
                    messages.pop()
                    # remove llm response with <think> and answer
                    messages.pop()
                    # replace it with re-anwer msg
                    revise_prompt = f"The assistant’s previous answer is: <answer>{assistant_content}</answer>, and please re-answer."
                    messages.append({"role": "user", "content": revise_prompt})
                    continue
                else:
                    # we have our answer, move on to scoring
                    # remove re-answer msg, but keep llm response with <think> and answer
                    messages.pop(-2)  # remove second to last msg
                    counter_for_multi_round_test_time_scaling = 0

            #####################
            # SCORING
            #####################
            is_pass_threshold, scoring_pros_cons = score_answer(question, messages)
            is_score_worse, streak_count = analyze_scores(scores)

            # score has been the same for streak_count
            if streak_count >= MAX_TRIES_TO_INCREASE_SCORE:
                is_pass_threshold = (
                    True  # just end it, it's not better in the last 3 attempts
                )

            # give it 1 more chance to get higher score
            if (
                is_score_worse or streak_count > 1
            ) and llm_call_count_to_increase_score <= 1:
                llm_call_count_to_increase_score += 1
                scores.pop()  # remove the last score
                messages.pop()  # remove the last answer
                continue  # go again
            else:
                llm_call_count_to_increase_score = 0

            # it still could not do it
            # TODO: take this out if you want to have it do manager feedback and keep trying
            if is_score_worse or streak_count > 1:
                is_pass_threshold = True

            if is_pass_threshold:
                #####################
                # FINAL ANSWER
                #####################
                prompt2 = f"""
**Instruction for Generating a Comprehensive Investigative Report**

You are tasked with creating an extensive and detailed investigative report based on the provided user question and our entire conversation. Your report should be thorough, balanced, and meticulously analyze all aspects of the user's question Ensure that no detail from the context is overlooked. The report should be accessible to users who may not be experts in the relevant domain, so explain all concepts clearly and informatively without implying any lack of expertise on the user's part.

**Requirements:**

1. **Depth and Detail:**
   - Explore every avenue related to the topic.
   - Include all pertinent information from the context.
   - Provide a nuanced and comprehensive narrative.

2. **Structure and Content:**
   - Address the following elements within the narrative: "Who," "What," "When," "Where," "How," or "Why"
   - Do not use "Who," "What," "When," "Where," "How," or "Why" as section titles. Instead, integrate these elements seamlessly into the text.

3. **Clarity and Accessibility:**
   - Explain all relevant concepts in a clear and informative manner.
   - Maintain sensitivity to the user's perspective, ensuring the report is understandable without assuming prior expertise.

4. **Source Citation:**
   - Cite all url sources with complete url

5. **Length and Comprehensiveness:**
   - Produce an exceptionally long and detailed report that thoroughly covers the topic.
   - Ensure the report maintains coherence and readability despite its length.
   
```User's Question
{question}
```
"""
                prompt = f"I conduct thorough research to create detailed and balanced long long long investigative reports. I explore every avenue to produce comprehensive narratives, considering that the user might not be an expert in the domain, class, or task. I explain concepts clearly and informatively, being sensitive to the user's perspective without highlighting any lack of expertise. I carefully analyze the entire conversation, ensuring no detail is overlooked. With this in mind, I will write a comprehensive narrative report that addresses the Who, What, When, Where, How, and Why, without using these as section titles, as a text response.\n\nUser's Question\n\n{question}"

                print("*" * 50)
                print("***** PRODUCING FINAL ANSWER *****")
                print("*" * 50)
                if USE_SERVICE_SUMMARY == Service.OPENAI:
                    final_answer = call_openai(
                        prompt, OPENAI_USE_MODEL_SUMMARY, messages
                    )
                else:
                    final_answer = call_llm(
                        prompt, LLM_USE_MODEL_SUMMARY, USE_SERVICE_SUMMARY, messages
                    )
                return final_answer
            else:
                #####################
                # MANAGER FEEDBACK
                #####################
                manager_feedback = get_manager_feedback(question, assistant_content)
                if (
                    is_score_worse
                ):  # get rid of the last attempt and try again, we only want winners
                    # scores.pop()
                    # messages.pop()
                    revise_prompt = f"Your work has not improved, this is worse than the last work. Use more tools and revise your response based on your Manager's feedback:{manager_feedback}\n\n{scoring_pros_cons}"
                    messages.append({"role": "user", "content": revise_prompt})
                    continue
                elif (MAX_TRIES_TO_INCREASE_SCORE > 0) and (streak_count > 1):
                    revise_prompt = f"Your work needs to improve, this was no improvement over the last work. Use more tools and revise your response based on your Manager's feedback:{manager_feedback}\n\n{scoring_pros_cons}"
                    messages.append({"role": "user", "content": revise_prompt})
                    continue
                else:
                    revise_prompt = f"Use more tools and revise your response based on your Manager's feedback:{manager_feedback}\n\n{scoring_pros_cons}"
                    messages.append({"role": "user", "content": revise_prompt})
                    continue

        elif finish_reason in ["length", "max_tokens", "content_filter"]:
            # The conversation got cut off or other forced stop
            print("The model's response ended due to finish_reason =", finish_reason)
            break

        # If we get here with no tool calls and not “stop,”
        # we can guess the model simply produced final text or there's no more to do
        if assistant_content.strip():
            print("\nAssistant:\n" + assistant_content)
            return assistant_content

    return "Lacked sufficient details to complete request."
