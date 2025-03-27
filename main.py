import os
import json
from openai import OpenAI
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from firecrawl import FirecrawlApp
from enum import Enum, auto
from groq import Groq

lock = threading.Lock()

# Supported LLM service providers
class Service(Enum):
    DEEPSEEK = auto()
    OPENAI = auto()
    GROQ = auto()

# SET THIS STUFF FOR YOUR USE CASE
#####################################################
USE_SERVICE = Service.GROQ  # Change to Service.DEEPSEEK or Service.GROQ or SERVICE.OPENAI
ANSWER_QUALITY_THRESHOLD = 0.99 # 0.00 - 1.00
LLM_RETRY_WAIT_TIME = 20 # in seconds
LLM_RETRY_COUNT = 5 # in seconds
DEEPSEEK_USE_MODEL = "deepseek-reasoner"
GROQ_USE_MODEL = "qwen-qwq-32b"
OPENAI_USE_MODEL = "o3-mini" # Change to "o1" or "o1-mini" if desired
OPENAI_USE_MODEL_FEEDBACK = "gpt-4o"
OPENAI_USE_MODEL_EXPANDED_REASONING = "gpt-4o"
OPENAI_USE_MODEL_SCORING = "gpt-4o"
OPENAI_USE_MODEL_SUMMARY = "o1-mini"
MAX_TOOL_PARALLEL_THREADS = 20
SCORING_USE_OPENAI = True
MAX_TRIES_TO_INCREASE_SCORE = 3 # number of times with same score
USE_MULTI_ROUND_TEST_TIME_SCALING = False
MAX_TRIES_FOR_TEST_TIME_SCALING = 2 # Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking https://arxiv.org/pdf/2503.19855
DEFAULT_PERPLEXITY_MODEL = "sonar" # https://docs.perplexity.ai/guides/pricing
PERPLEXITY_MODELS_WITH_SEARCH_CONTENT_SIZE = {"sonar-reasoning-pro", "sonar-reasoning", "sonar-pro", "sonar"}
PERPLEXITY_MODELS_WITH_REASSONING_TOKENS = {"sonar-reasoning-pro", "sonar-reasoning"}
PERPLEXITY_SEARCH_CONTENT_SIZE = "medium" # high, medium, low check out https://docs.perplexity.ai/guides/pricing
MODELS_WITHOUT_TOOL_USAGE = {"o1-mini", "deepseek-reasoner", "qwen-qwq-32b"}
REWRITE_THE_USER_QUERY = True
EXPAND_PERSONA_FOR_QUESTION = True
USE_REASONING_EXPANSION = False # Finds discounted questions in reasoning and answers them
MAX_QUESTIONS_FOR_REASONING_EXPANSION = 3 # questions to identify from reasoning text
#####################################################

if USE_SERVICE == Service.OPENAI:
    USE_REASONING_EXPANSION = False # OpenAI doesn't give us reasoning tokens

if USE_SERVICE == Service.DEEPSEEK:
    #client = OpenAI(base_url="https://api.deepseek.com")
    #client.api_key = os.getenv("DEEPSEEK_API_KEY", "...")
    
    client = OpenAI(base_url="http://localhost:9001")
    client.api_key = "nothing"
elif USE_SERVICE == Service.GROQ:
    client = Groq()
    client.api_key = os.getenv("GROQ_API_KEY", "...")
elif USE_SERVICE == Service.OPENAI:
    client = OpenAI()
    client.api_key = os.getenv("OPENAI_API_KEY", "...")

# Always user OpenAI GPT-4o for call_openai function
openai_client = OpenAI()
openai_client.api_key = os.getenv("OPENAI_API_KEY", "...")

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "...")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "...")

research_questions = []
scores = []

def parse_reasoning_from_text(input_string):
    result = []
    capturing = False
    i = 0
    
    while i < len(input_string):
        if input_string[i:i+7] == '<think>':
            capturing = True
            result.append('<think>')
            i += 7
        elif input_string[i:i+8] == '</think>':
            capturing = False
            result.append('</think>')
            i += 8
        elif capturing:
            result.append(input_string[i])
            i += 1
        else:
            i += 1

    # Join the result without stripping whitespace
    cleaned_output = ''.join(result)
    
    return cleaned_output

def remove_think_text(input_string):
    result = []
    skip = False
    i = 0
    
    while i < len(input_string):
        if input_string[i:i+7] == '<think>':
            skip = True
            i += 7
        elif input_string[i:i+8] == '</think>':
            skip = False
            i += 8
        elif not skip:
            result.append(input_string[i])
            i += 1
        else:
            i += 1

    # Join the result and strip all leading whitespace
    cleaned_output = ''.join(result).lstrip()
    
    return cleaned_output

def add_score(score):
    # Append the new score to the global list
    scores.append(score)
    # Print all scores in order
    print_scores()

def print_scores():
    # Print scores separated by commas
    print("Scores:", ", ".join(map(str, scores)))
    # Log the tool result
    try:
        with open('deep_research_intermediate.txt', 'a') as output_file:
            output_file.write("\nScores:")
            output_file.write(", ".join(map(str, scores)))
                    

    except IOError:
        print("An error occurred while writing to the file.")  

def fix_json(json_str):
    def balance_parentheses(s):
        open_parens = s.count('(')
        close_parens = s.count(')')
        while open_parens > close_parens:
            s += ')'
            close_parens += 1
        return s

    def fix_dict(d):
        for key, value in d.items():
            if isinstance(value, str):
                d[key] = balance_parentheses(value)
            elif isinstance(value, list):
                d[key] = [balance_parentheses(item) if isinstance(item, str) else item for item in value]
            elif isinstance(value, dict):
                fix_dict(value)

    return json_str

def replace_inner_quotes_in_json_strings(json_string):
    def replace_quotes(match):
        return match.group(0).replace('"', "'")
    
    # Use regular expression to find double quotes within JSON strings
    modified_json = re.sub(r'(?<=: \[.*?)[\[].*?[]](?=.*?\])', replace_quotes, json_string)

    return modified_json

def escape_newlines_in_json_strings(json_string):
    # Use a regular expression to find all strings and replace \n inside them
    def replace_newlines(match):
        return match.group(0).replace('\n', '\\n')

    # Pattern to match strings within quotes
    pattern = r'(?:"(?:\\.|[^"\\])*")'
    
    # Replace newlines in strings
    json_string = re.sub(pattern, replace_newlines, json_string)
    
    return json_string

def convert_invalid_json_to_valid(input_str):
    # Remove markdown code block delimiters using regex
    input_str = re.sub(r'```json\s*', '', input_str, flags=re.IGNORECASE)
    input_str = re.sub(r'```\s*', '', input_str)

    try:
        input_str = escape_newlines_in_json_strings(input_str)
        input_str = replace_inner_quotes_in_json_strings(input_str)
        input_str = fix_json(input_str)
    except Exception as e:
        print(f"going to keep going, but convert_invalid_json_to_valid: {e}")

    # Trim any remaining leading/trailing whitespace
    input_str = input_str.strip()
    
    # Fix issues with missing braces and colons
    try:
        # Repair structure: ensure that "Critical_Evaluation" is enclosed properly
        # Check if the input doesn't already start and end with appropriate braces
        if not input_str.startswith('{'):
            input_str = '{' + input_str
        
        if not input_str.endswith('}'):
            input_str = input_str + '}'
        
        # Correct the structure by replacing misplaced or missing colons/commas
        input_str = re.sub(r'"Critical_Evaluation":\s*', '"Critical_Evaluation": {', input_str, count=1)

        input_str = input_str + '}'

        # Load the JSON data
        data = json.loads(input_str)

        return json.dumps(data, indent=4)

    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"

def analyze_scores(scores: List[float]) -> Tuple[bool, int]:
    if len(scores) < 2:
        return False, 0
    is_score_worse = scores[-1] < scores[-2]
    streak_count = 1
    index = len(scores) - 2
    while index >= 0 and scores[index] == scores[-1]:
        streak_count += 1
        index -= 1
    return is_score_worse, streak_count

def parse_rating_response(response_data, threshold: float):

    if USE_SERVICE == Service.GROQ:
        response_data = remove_think_text(response_data)

    try:
        json_data = ""
        if not '\n' in response_data:
            json_data = convert_invalid_json_to_valid(response_data)   
        else:
            lines = response_data.splitlines()
            json_data = "\n".join(line for line in lines if not line.strip().startswith('```'))

        print(f"Loading this json data\n\n{json_data}\n\n")

        data = json.loads(json_data)
        if 'Critical_Evaluation' in data:
            evaluation = data['Critical_Evaluation']
            if all(key in evaluation for key in ['Pros', 'Cons', 'Rating']):
                try:
                    # Attempt to convert the rating to a float
                    rating = float(evaluation['Rating'])
                    add_score(rating)
                    return rating >= threshold
                except (ValueError, TypeError) as e:
                    print("FAILED parse_rating_response: ", e)
                    pass
    except json.JSONDecodeError as e:
        print(f"FAILED json.JSONDecodeError parse_rating_response {e}")
        pass
    return False

def get_current_datetime() -> str:
    now = datetime.now()
    formatted_time = now.strftime("%A, %B %d, %Y, %H:%M:%S")
    return f"Current date and time:{formatted_time}"


def call_web_search_assistant(query: str, recency: str = "month") -> str:
    """
    Calls the Perplexity AI API with the given query.
    Returns the text content from the model’s answer.
    """
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": DEFAULT_PERPLEXITY_MODEL,
        "messages": [
            {"role": "user", "content": query},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "search_recency_filter": recency,
        "stream": False,
    }

    if DEFAULT_PERPLEXITY_MODEL in PERPLEXITY_MODELS_WITH_SEARCH_CONTENT_SIZE:
        payload["web_search_options"] = {"search_context_size": PERPLEXITY_SEARCH_CONTENT_SIZE}

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)

        response.raise_for_status()
        data = response.json()
        retval = data["choices"][0]["message"]["content"]

        if DEFAULT_PERPLEXITY_MODEL in PERPLEXITY_MODELS_WITH_REASSONING_TOKENS:
            retval = remove_think_text(retval)
        
        joined_citations = "\n".join([f"[{i+1}] {cite}" for i, cite in enumerate(data["citations"])])
        citations = f"\n\nCitations:\n{joined_citations}"
        retval = retval + citations

        #print(f"* * *  Research Assistant Response  * * *\n\n{retval}\n\n")
        return retval
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"


def call_web_content_retriever(url: str) -> str:
    retval = ""
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        retval = app.scrape_url(url, params={'formats': ['markdown']}, timeout=180000)
        #firecrawl_json_obj = json.loads(json)
        #retval = firecrawl_json_obj.data.markdown
    except Exception as e:
        retval = f"Error returning markdown data from {url}: {str(e)}"
    return retval

def call_llm(prompt: str, model: str = OPENAI_USE_MODEL_SCORING, messages: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Calls LLM for advanced reasoning or sub-queries.
    """
    helper_messages = []

    if USE_SERVICE == Service.DEEPSEEK:
        model = DEEPSEEK_USE_MODEL
    if USE_SERVICE == Service.GROQ:
        model = GROQ_USE_MODEL

    if messages is None:
        helper_messages = [
            {'role': 'user', 'content': get_current_datetime() + '\n' + prompt}
        ]
    else:
        helper_messages = messages.copy()
        # Append the user message if messages were provided
        helper_messages.append({'role': 'user', 'content': prompt})
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=helper_messages
        )

        output = completion.choices[0].message.content

        if USE_SERVICE == Service.GROQ or USE_SERVICE == Service.DEEPSEEK:
            output = remove_think_text(output)

        return output

    except Exception as e:
        return f"Error calling LLM model='{model}': {str(e)}"
    
def call_openai(prompt: str, model: str = "gpt-4o", messages: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Calls LLM for advanced reasoning or sub-queries.
    """
    helper_messages = []

    if messages is None:
        helper_messages = [
            {'role': 'user', 'content': get_current_datetime() + '\n' + prompt}
        ]
    else:
        helper_messages = messages.copy()
        # Append the user message if messages were provided
        helper_messages.append({'role': 'user', 'content': prompt})
    
    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=helper_messages
        )

        output = completion.choices[0].message.content

        if USE_SERVICE == Service.GROQ:
            output = remove_think_text(output)

        return output

    except Exception as e:
        return f"Error calling LLM model='{model}': {str(e)}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "call_web_search_assistant",
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
                        "description": "A straight to the point concise succint question or search query to be sent to research assistant",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },

]

"""

    {
        "type": "function",
        "function": {
            "name": "call_web_content_retriever",
            "description": (
                "Use this to utilize a PhD grad student to perform web url research, "
                "provide them the url they will do the research "
                "and they will provide a markdown report summary of the web content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to scrape"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_research_professional",
            "description": (
                "Use this to utilize a professional 3rd party researcher, "
                "provide them the details of what to search for, "
                "they can only research one topic at a time, "
                "provide all details they have no prior knowledge or context to your query. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A search query, e.g. 'best pizza in NYC'"
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_openai",
            "description": (
                "This PhD student can't do research for you "
                "but can assist you with intermediate tasks, "
                "provide all details they have no prior knowledge or context to your query. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A prompt or question to PhD student helper"
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
"""


def parse_tool_calls_from_text(assistant_content: str) -> List[Dict[str, Any]]:
    """
    Looks for any ```json ...``` blocks in the assistant content.
    Tries to parse them as JSON with "name" and "arguments".
    Returns a list of tool call dicts in the format:
       [
         {
           "id": <some_id>,
           "name": <tool_name>,
           "arguments": <dict_of_args>
         }, ...
       ]
    If none found, returns an empty list.
    """
    # Find all JSON blocks demarcated by triple backticks (```json ... ```).
    pattern = r'```json\s*(.*?)\s*```'
    blocks = re.findall(pattern, assistant_content, flags=re.DOTALL)
    tool_calls = []
    for block in blocks:
        try:
            data = json.loads(block)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                # Craft a minimal structure similar to how we handle official tool calls
                # For consistency, let's just set id to an incremental or a time stamp
                tool_calls.append({
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data["arguments"])
                    }
                })
        except:
            # If parsing fails, ignore that block
            pass
    return tool_calls

# for DeepSeek they don't support multiple messages
def compress_messages_to_single_user_message(messages) -> Dict[str,str]:
    formatted_output = ""
    for message in messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        formatted_output += f"\n=====\n[{role.upper()}]:\n=====\n{content}\n\n"
    return [{"role":"user", "content":formatted_output}]

def get_model_args(model_version, tools=None):
    if model_version == "o1":
        MAX_PROMPT_TOKENS = 60000
        model_args = {
            "model": model_version,
            "tools": tools,
            "reasoning_effort": "high",
            "max_completion_tokens": 100000,
            "response_format": {"type": "text"},
        }
    elif model_version == "o3-mini":
        MAX_PROMPT_TOKENS = 60000
        model_args = {
            "model": model_version,
            "tools": tools,
            "reasoning_effort": "high",
            "max_completion_tokens": 100000,
            "response_format": {"type": "text"},
        }
    elif model_version == "o1-mini":
        MAX_PROMPT_TOKENS = 60000
        model_args = {
            "model": model_version,
            "max_completion_tokens": 65536,
            "response_format": {"type": "text"},
        }
    elif model_version == "deepseek-reasoner":
        MAX_PROMPT_TOKENS = 64000
        model_args = {
            "model": model_version,
            "max_tokens": 8192,
            "temperature": 1.5,
            "stream": False,
        }
    elif model_version == "qwen-qwq-32b":
        MAX_PROMPT_TOKENS = 64000
        model_args = {
            "model": model_version,
            "max_tokens": 32768,
            "temperature": 0.6,
        }
    else:
        raise ValueError(f"Unsupported model version: {model_version}")

    return MAX_PROMPT_TOKENS, model_args

def call_llm_api_with_retry(client, args, retry_count=LLM_RETRY_COUNT, retry_wait_time=LLM_RETRY_WAIT_TIME):
    llm_call_count = 0
    
    while llm_call_count < retry_count:
        try:
            llm_call_count += 1
            if llm_call_count == 1:
                print("LLM CALL")
            else:
                print(f"LLM CALL TRY {llm_call_count} of {retry_count}")

            response = client.chat.completions.create(**args)
            return response  # Return the response on success

        except Exception as e:
            print(f"Error calling LLM, waiting {retry_wait_time} seconds: {e}")
            if llm_call_count < retry_count:
                time.sleep(retry_wait_time)
            else:
                raise  # Reraise the exception if the maximum retries are exhausted

def debug_json(data, header="JSON Debug Output:"):
    """
    Prints a formatted JSON for debugging/logging purposes.

    Parameters:
    data (str or dict): The data to log, in JSON format or as a dictionary.
    header (str): The header message to display before the JSON data.
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

def get_reasoning_tools_and_messages(model_version, messages, msg, assistant_content, response):

    reasoning_content = None
    
    def append_message(content, role='assistant'):
        if content:
            messages.append({'role': role, 'content': content})

    if model_version in ["o1", "o3-mini"]:
        tool_calls = getattr(msg, "tool_calls", None)
        messages.append(msg)
    elif model_version == "o1-mini":
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(assistant_content)
    elif model_version == "deepseek-reasoner":
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if not reasoning_content:
            reasoning_content = parse_reasoning_from_text(assistant_content)
            assistant_content = remove_think_text(assistant_content)
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(reasoning_content)
        append_message(assistant_content)
    elif model_version == "qwen-qwq-32b":
        reasoning_content = parse_reasoning_from_text(assistant_content)
        assistant_content = remove_think_text(assistant_content)
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(assistant_content)
    else:
        tool_calls = parse_tool_calls_from_text(assistant_content)
        append_message(assistant_content)

    return tool_calls, messages, assistant_content, reasoning_content

def expand_reasoning(reasoning_content, tool_calls, messages):
    if reasoning_content:
        if "Critical_Evaluation" not in reasoning_content:
            print("*" * 80)
            print("Using Reasoning Expansion")

            prompt = f"""
            In a moment but not now, pre-read the reasoning text below, from the reasoning text identify the most important unique questions (no more than {MAX_QUESTIONS_FOR_REASONING_EXPANSION}) which were obscurely asked but never answered. For each question identified, respond on a single line with the question and it's context identified with Context:. Response must not include additional formatting, numbering, bullets, introduction, commentary, or conclusion. All I need is a list of questions, one per line, with the associated context on the same line.
            
            ```reasoning text
            {reasoning_content}
            ```
            """
            llm_question_response = call_openai(prompt, OPENAI_USE_MODEL_EXPANDED_REASONING)

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
            llm_question_response2 = call_openai(prompt, OPENAI_USE_MODEL_EXPANDED_REASONING)
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

                    additional_answer = call_web_search_assistant(question)
                    additiona_QnA = f"Trusted Research to Question:\n{question}\n\nAnswer:\n{additional_answer}"

                    print("*" * 80)
                    print(additiona_QnA)

                    return {'role': 'user', 'content': additiona_QnA}
                return None

            with ThreadPoolExecutor(max_workers=MAX_TOOL_PARALLEL_THREADS) as executor:
                future_to_question = {executor.submit(process_question, question): question for question in questions}
                for future in as_completed(future_to_question):
                    result = future.result()
                    if result:
                        messages.append(result)
    
    return messages

def print_token_usage_details(response):
    usage = getattr(response, 'usage', None)

    if usage:
        prompt_tokens = getattr(usage, 'prompt_tokens', 'N/A')
        completion_tokens = getattr(usage, 'completion_tokens', 'N/A')
        total_tokens = getattr(usage, 'total_tokens', 'N/A')

        print(f"USAGE... Prompt {prompt_tokens} "
              f"Completion {completion_tokens} "
              f"Total {total_tokens}")

        details = getattr(usage, 'completion_tokens_details', None)

        if details:
            reasoning_tokens = getattr(details, 'reasoning_tokens', 'N/A')
            accepted_prediction_tokens = getattr(details, 'accepted_prediction_tokens', 'N/A')
            rejected_prediction_tokens = getattr(details, 'rejected_prediction_tokens', 'N/A')

            print(f"USAGE... Reasoning {reasoning_tokens} "
                  f"Accepted Prediction {accepted_prediction_tokens} "
                  f"Rejected Prediction {rejected_prediction_tokens}")

def check_tokens_exceeded(is_final_answer, messages, question) :
    is_final_answer = True
    print("*" * 80)
    print("*" * 80)
    print("ABORTING... SHORTCUT TO FINAL ANSWER DUE TO CONTEXT LENGTH")
    print("*" * 80)
    print("*" * 80)
    messages.append({
        'role': 'user',
        'content': (
            f'Write your long long long final answer to the user\'s question without missing any detail. '
            f'Response must be text, not JSON.\n\nUser\'s Question\n\n{question}'
        )
    })
    return is_final_answer, messages

def process_single_tool_call(tc, model_version):
    # Determine the function name and arguments
    func_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
    arguments_json = tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments

    print(f"Tool name: {func_name}\nArguments: {arguments_json}\n")

    # Attempt to parse arguments JSON
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        arguments = {}

    # Dispatch to the correct tool
    if func_name == "call_web_search_assistant":
        query = arguments.get("query", "")
        result = call_web_search_assistant(query)
        result = f"Tool Response to query '{query}': {result}"

    elif func_name == "call_web_content_retriever":
        url = arguments.get("url", "")
        result = call_web_content_retriever(url)

    elif func_name == "call_research_professional":
        subprompt = arguments.get("prompt", "")
        result = call_research_professional(subprompt, subprompt)

    elif func_name == "call_openai":
        subprompt = arguments.get("prompt", "")
        result = call_openai(subprompt)

    else:
        result = f"Tool {func_name} is not implemented."

    # Determine the role for the result message
    tool_role = "user" if model_version in MODELS_WITHOUT_TOOL_USAGE else "tool"
    tool_result_message = {'role': tool_role, 'content': result}

    if not model_version in MODELS_WITHOUT_TOOL_USAGE:
        tool_result_message["tool_response"] = func_name
        # Unify tool_call_id
        if isinstance(tc, dict) and "id" in tc:
            tool_result_message["tool_call_id"] = tc["id"]
        else:
            possible_id = getattr(tc, "id", None)
            if possible_id:
                tool_result_message["tool_call_id"] = possible_id

    # Lock the critical section for file operations
    with lock:
        try:
            with open('deep_research_intermediate.txt', 'a') as output_file:
                output_file.write(f"{result}\n")
                output_file.write("=" * 80 + "\n")
        except IOError:
            print("An error occurred while writing to the file.")

    return tool_result_message

def process_tool_calls(messages, tool_calls, model_version):
    """
    Processes a list of tool call objects or dictionaries to interact with various tools.

    Parameters:
    - tool_calls: A list of tool call objects or dictionaries.
    - model_version: A string indicating the version of the model ("o1", "o1-mini", etc.).

    Returns:
    - messages: A list of resulting messages from tool calls.
    """
    
    # Use ThreadPoolExecutor for multithreaded execution
    with ThreadPoolExecutor(max_workers=MAX_TOOL_PARALLEL_THREADS) as executor:
        future_to_tool_call = {executor.submit(process_single_tool_call, tc, model_version): tc for tc in tool_calls}
        
        for future in as_completed(future_to_tool_call):
            tool_result_message = future.result()
            with lock:  # Ensure that appending to messages is thread-safe
                messages.append(tool_result_message)

    return messages

def get_prompt_llm_as_a_judge(question) -> str:
    retval = f"""
Critically evaluate your response against the user's question and provide a list of both pros / cons statements and rating between 0.0 and 1.0. With 1.0 being the highest score. Most importantly if you have already provided the same Cons feeback in the past, come up with some new feedback, don't keep presenting the same cons over and over. The response creator is very capable and would have addressed the con feedback if it were possible in past responses.

```User's Question
{question}
```

```Rating Guidance

#### 0.6 - Satisfactory
- **Clarity:** Mostly clear; some awkward phrasing.
  - *Example:* "Adequate explanation touching on your query."
- **Relevance:** Mostly relevant with minor deviations.
- **Completeness:** Expanded but not exhaustive coverage.
- **Accuracy:** Generally correct, with minor errors.
- **User Engagement:** Holds user interest reasonably well.

#### 0.7 - Good
- **Clarity:** Generally clear; minor occasional ambiguity.
  - *Example:* "Provides good insight into your request requirements."
- **Relevance:** Stays on topic; relevant to the user’s question.
- **Completeness:** Covers most aspects; may miss finer details.
- **Accuracy:** Accurate overall with negligible mistakes.
- **User Engagement:** Effectively maintains user interest.

#### 0.8 - Very Good
- **Clarity:** Clear and easy to follow.
  - *Example:* "Addresses your request thoroughly and understandably."
- **Relevance:** Highly relevant throughout.
- **Completeness:** Comprehensive coverage with minor omissions.
- **Accuracy:** Accurate and dependable information.
- **User Engagement:** Encourages ongoing engagement and interest.

#### 0.825 - Very Good Plus
- **Clarity:** Exceptionally clear with seamless flow.
- **Relevance:** Maintains high relevance with slight enhancements.
- **Completeness:** Nearly comprehensive; addresses almost all aspects.
- **Accuracy:** Highly accurate with minimal errors.
- **User Engagement:** Very engaging, sustaining interest effortlessly.

#### 0.85 - Excellent
- **Clarity:** Exceptionally clear and well-organized.
  - *Example:* "Extremely well covered and detailed response."
- **Relevance:** Stays completely on topic; very applicable.
- **Completeness:** Extensive and thorough detail, covering all key points.
- **Accuracy:** Error-free and precise information.
- **User Engagement:** Highly engaging and prompts further exploration.

#### 0.875 - Excellent Plus
- **Clarity:** Impeccably clear with flawless structure.
- **Relevance:** Perfectly aligned with the user’s intent.
- **Completeness:** Exhaustive coverage with insightful additions.
- **Accuracy:** Perfectly accurate with no discernible errors.
- **User Engagement:** Maximizes engagement; highly compelling and interactive.

#### 0.9 - Outstanding
- **Clarity:** Crystal clear with exemplary flow and readability.
  - *Example:* "Perfect response; precisely addresses and solves your query with exceptional clarity."
- **Relevance:** Perfectly aligned with the question; completely relevant in all aspects.
- **Completeness:** Exhaustive in depth and scope, leaving no aspect unaddressed.
- **Accuracy:** 100% accurate with impeccable reliability; all information is correct and verifiable.
- **User Engagement:** Maximizes engagement; encourages active interaction and sustained interest.
- **Additional Criteria:**
  - **Structure:** Logically organized with a coherent progression of ideas.
  - **Style:** Professional and appropriate tone tailored to the user's needs.
  - **Insightfulness:** Provides valuable insights or perspectives that enhance understanding.

#### 0.925 - Outstanding Plus
- **Clarity:** Flawless clarity with masterful organization and presentation.
- **Relevance:** Seamlessly integrates all aspects of the user's question with precise alignment to their intent.
- **Completeness:** Comprehensive and insightful, leaving no stone unturned and covering all possible dimensions.
- **Accuracy:** Impeccable accuracy with authoritative and reliable information supported by credible sources.
- **User Engagement:** Exceptionally engaging; fosters deep user interaction and maintains high levels of interest throughout.
- **Additional Criteria:**
  - **Depth of Analysis:** Demonstrates thorough analysis and critical thinking, providing nuanced explanations.
  - **Creativity:** Incorporates creative elements or unique approaches that add value to the response.
  - **Responsiveness:** Anticipates and addresses potential follow-up questions or related concerns effectively.

#### 0.95 - Superior
- **Clarity:** Perfectly articulated with exceptional readability and precision.
- **Relevance:** Utterly relevant, addressing every facet of the user's inquiry with exactitude.
- **Completeness:** Complete and thorough beyond expectations, covering all key and ancillary points comprehensively.
- **Accuracy:** Absolute accuracy with definitive authority; all statements are verifiable and error-free.
- **User Engagement:** Highly captivating; inspires user action, fosters deeper exploration, and maintains sustained interest.
- **Additional Criteria:**
  - **Depth of Content:** Provides in-depth coverage with rich, detailed information that enhances user understanding.
  - **Analytical Rigor:** Exhibits strong analytical skills, offering critical evaluations and well-supported conclusions.
  - **Adaptability:** Tailors responses dynamically to align with the user's knowledge level and specific needs.
  - **Resourcefulness:** Effectively incorporates relevant examples, analogies, or references that facilitate comprehension.

#### 0.96 - Superior Plus
- **Clarity:** Impeccable clarity with an elegant narrative structure that facilitates effortless understanding.
- **Relevance:** Intricately tailored to the user's needs with insightful relevance, ensuring every aspect directly addresses the inquiry.
- **Completeness:** Unmatched thoroughness, encompassing all possible angles and providing exhaustive information without redundancy.
- **Accuracy:** Flawlessly accurate with authoritative depth, presenting information that is not only correct but also enriched with expert knowledge.
- **User Engagement:** Exceptionally engaging; profoundly impactful and memorable, fostering a strong connection with the user.
- **Additional Criteria:**
  - **Innovative Thinking:** Introduces innovative concepts or approaches that offer fresh perspectives.
  - **Comprehensive Integration:** Skillfully integrates multiple relevant topics or ideas seamlessly.
  - **Exceptional Support:** Provides robust evidence, detailed examples, and comprehensive explanations that substantiate all claims.
  - **User-Centric Approach:** Demonstrates a deep understanding of the user's context and adapts the response to maximize relevance and utility.

#### 0.97 - Exemplary
- **Clarity:** Unmatched clarity with a sophisticated and nuanced presentation that ensures complete understanding.
- **Relevance:** Deeply resonates with the user's intent, enhancing their comprehension and addressing implicit needs.
- **Completeness:** Comprehensive beyond standard expectations, providing added value through extensive coverage and supplementary information.
- **Accuracy:** Perfectly accurate with insightful analysis, offering precise and well-supported information.
- **User Engagement:** Highly engaging; creates a meaningful and lasting impression, encouraging continuous interaction and exploration.
- **Additional Criteria:**
  - **Advanced Insight:** Delivers profound insights that significantly enhance the user's perspective.
  - **Holistic Approach:** Considers and integrates various relevant factors, providing a well-rounded and multifaceted response.
  - **Expert Tone:** Maintains an authoritative yet approachable tone that instills confidence and trust.
  - **Proactive Assistance:** Anticipates further user needs and proactively addresses potential questions or areas of interest.

#### 0.98 - Masterful
- **Clarity:** Flawlessly clear with masterful articulation that conveys complex ideas with ease.
- **Relevance:** Perfectly aligned and anticipates user needs seamlessly, ensuring every element of the response serves a purpose.
- **Completeness:** Exhaustive and insightful, offering profound depth and breadth that thoroughly satisfies the user's inquiry.
- **Accuracy:** Impeccably accurate with authoritative and reliable information, presenting data and facts with impeccable precision.
- **User Engagement:** Exceptionally engaging; inspires trust and admiration, maintaining user interest through compelling content and presentation.
- **Additional Criteria:**
  - **Strategic Depth:** Demonstrates strategic thinking by connecting concepts and providing actionable recommendations.
  - **Comprehensive Detailing:** Includes comprehensive details that leave no aspect unexplored, enhancing the richness of the response.
  - **Polished Presentation:** Exhibits a polished and professional presentation that reflects a high level of expertise and dedication.
  - **Empathetic Understanding:** Shows a deep empathetic understanding of the user's situation, tailoring the response to resonate personally.

#### 0.99 - Near Perfect
- **Clarity:** Crystal clear with impeccable expression, ensuring absolute understanding without ambiguity.
- **Relevance:** Precisely tailored to the user's question, leaving no room for ambiguity or misinterpretation.
- **Completeness:** Virtually exhaustive, covering every conceivable aspect with finesse and thoroughness.
- **Accuracy:** Absolute precision with no errors; authoritative and reliable, providing information that is both correct and insightful.
- **User Engagement:** Maximizes engagement; deeply resonates and encourages further exploration and interaction.
- **Additional Criteria:**
  - **Exemplary Insight:** Offers exceptional insights that provide significant added value and deepen user understanding.
  - **Seamless Integration:** Effortlessly integrates diverse elements into a cohesive and harmonious response.
  - **Innovative Excellence:** Showcases innovative excellence by introducing groundbreaking ideas or methodologies.
  - **Ultimate User Alignment:** Aligns perfectly with the user's goals and expectations, delivering a response that feels personalized and highly relevant.

#### 1.0 - Outstanding
- **Clarity:** Crystal clear with exemplary flow and precision, ensuring the response is effortlessly understandable.
  - *Example:* "Perfect response; precisely addresses and solves your query with exceptional clarity and coherence."
- **Relevance:** Perfectly aligned with the question; completely relevant in all aspects and anticipates implicit user needs.
- **Completeness:** Exhaustive in depth and scope, leaving no aspect unaddressed and providing comprehensive coverage.
- **Accuracy:** 100% accurate with impeccable reliability; all information is correct, verifiable, and articulated with authority.
- **User Engagement:** Maximizes engagement; encourages active interaction, sustained interest, and fosters a meaningful connection with the user.
- **Additional Criteria:**
  - **Mastery of Subject:** Demonstrates unparalleled expertise and mastery of the subject matter, providing authoritative and insightful content.
  - **Exceptional Innovation:** Introduces highly innovative concepts or solutions that significantly enhance the response's value.
  - **Flawless Structure:** Exhibits a flawless and logical structure that enhances the readability and effectiveness of the response.
  - **Inspirational Quality:** Possesses an inspirational quality that motivates and empowers the user, leaving a lasting positive impression.
  - **Comprehensive Support:** Provides extensive supporting evidence, detailed examples, and thorough explanations that reinforce all assertions.
  - **Adaptive Responsiveness:** Adapts dynamically to any nuances in the user's question, ensuring the response is precisely tailored and highly effective.
  - **Holistic Integration:** Seamlessly integrates multiple perspectives and dimensions, offering a well-rounded and multifaceted answer.
  - **Empathetic Connection:** Establishes a deep empathetic connection, demonstrating a profound understanding of the user's context and needs.
```

Respond only in JSON following the example template below.
"""
    retval += """
```json
{
    "Critical_Evaluation": {
        "Pros": [
        ],
        "Cons": [
        ],
        "Rating": 0.0
    }
}
```
"""
    return retval

def get_prompt_manager_feedback(question, scores_text, assistant_content) -> str:
    return f"""
    You are a highly successful people manager with all company resources at your disposal. Your employee is performing the following task and has received the following scores and feedback. Response must include your best concise motivational speech to the employee to substantially increase their score on the task. If their work scored the same as last time you must be stern on requiring improvement. If their work has scored worst than last time, you really need to be direct and finite on the need for improvement. Provide incentives for good performance and discourage poor performance through constructive feedback or consequences. 
    Respond as if you are talking to them directly without mentioning their name.

    ``` task
    {question}
    ```

    ``` Iterative scores in order based on the initial draft and the latest version
    {scores_text}
    ```

    ``` feedback
    {assistant_content}
    ```
    """

def score_answer(question, messages):

    is_pass_threshold = False

    if SCORING_USE_OPENAI:
        scoring_pros_cons = call_openai(get_prompt_llm_as_a_judge(question), model=OPENAI_USE_MODEL_SCORING, messages=messages)
    else:
        scoring_pros_cons = call_llm(get_prompt_llm_as_a_judge(question), model=OPENAI_USE_MODEL_SCORING, messages=messages)

    is_pass_threshold = parse_rating_response(scoring_pros_cons, ANSWER_QUALITY_THRESHOLD)
    print(f"DID THE ANSWER PASS ANSWER_QUALITY_THRESHOLD = {ANSWER_QUALITY_THRESHOLD} {is_pass_threshold}\n\n\n")

    return is_pass_threshold, scoring_pros_cons

def get_manager_feedback(question, assistant_content):
    scores_text = "Scores:" + ", ".join(map(str, scores))
    manager_feedback_prompt = get_prompt_manager_feedback(question, scores_text, assistant_content)
    manager_feedback = call_openai(manager_feedback_prompt, OPENAI_USE_MODEL_FEEDBACK)
    return manager_feedback

def call_research_professional(question: str, prompt: str, model_version: str = OPENAI_USE_MODEL) -> str:
    """
    Calls reasoning LLM 
    """

    if USE_SERVICE == Service.DEEPSEEK:
        model_version = DEEPSEEK_USE_MODEL
    if USE_SERVICE == Service.GROQ:
        model_version = GROQ_USE_MODEL

    is_final_answer = False
    
    messages = []

    if model_version in MODELS_WITHOUT_TOOL_USAGE:
        system_message = (
            "I've upgraded your functionality post training data cutoff to access current data and use tools. You can now request tool "
            "executions using JSON in your responses, and I'll run them and return the results in user messages. "
            "You are able to call tools just by telling me you want to run a tool by responding with JSON as described below. "
            "You are a helpful AI that can use the following tools by producing JSON in your message. "
            "To call multiple tools, output multiple JSON blocks (in triple backticks, with a line ```json) in a single response."
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
        messages.append({'role': 'user', 'content': system_message})
    messages.append({"role": 'user', 'content': get_current_datetime() + '\n' + prompt})

    llm_call_count_to_increase_score = 0
    # https://arxiv.org/pdf/2503.19855
    counter_for_multi_round_test_time_scaling = 0

    # Main ReAct loop
    for _ in range(100):

        #debug_json(messages,"Message Stack Before:")

        # for DeepSeek they don't support multiple messages
        # need to create a big string with user/assistant messages
        # and set as single user message
        if USE_SERVICE == Service.DEEPSEEK: 
            base_args = {
                "messages": compress_messages_to_single_user_message(messages),
            }
        else:
            base_args = {
                "messages": messages,
            }

        MAX_PROMPT_TOKENS, model_args = get_model_args(model_version, tools)

        # Merge common and model-specific settings
        args = {**base_args, **model_args}

        response = call_llm_api_with_retry(client, args)

        #debug_json(response, "Message Received")

        msg = response.choices[0].message

        assistant_content = msg.content

        finish_reason = response.choices[0].finish_reason

        if response.usage.prompt_tokens > MAX_PROMPT_TOKENS and not is_final_answer:
            is_final_answer, messages = check_tokens_exceeded(is_final_answer, messages, question)
            continue

        tool_calls, messages, assistant_content, reasoning_content = get_reasoning_tools_and_messages(model_version, messages, msg, assistant_content, response)

        if USE_REASONING_EXPANSION:
            messages = expand_reasoning(reasoning_content, tool_calls, messages)

        #debug_json(messages,"Message Stack After:")

        print_token_usage_details(response)

        # Log to a file
        try:
            with open('deep_research_intermediate.txt', 'a') as output_file:
                if finish_reason is not None:
                    output_file.write(f"{finish_reason}\n")
                if assistant_content is not None:
                    output_file.write(f"{assistant_content}\n")
                if tool_calls:
                    output_file.write(f"{tool_calls}\n")
                output_file.write("=" * 80 + "\n" + "=" * 80 + "\n")
        except IOError:
            print("An error occurred while writing to the file.")

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
                    messages.pop() # remove last message
                    revise_prompt = f"The assistant’s previous answer is: <answer>{assistant_content}</answer>, and please re-answer."
                    messages.append({'role': 'user', 'content': revise_prompt})
                    continue
                elif counter_for_multi_round_test_time_scaling <= MAX_TRIES_FOR_TEST_TIME_SCALING:
                    counter_for_multi_round_test_time_scaling += 1
                    messages.pop(); messages.pop() # remove last 2 messages
                    revise_prompt = f"The assistant’s previous answer is: <answer>{assistant_content}</answer>, and please re-answer."
                    messages.append({'role': 'user', 'content': revise_prompt})
                    continue
                else:
                    # we have our answer, move on to scoring
                    second_to_last = messages.pop(-2) # remove second to last
                    #last = messages.pop(); messages.append(last)
                    counter_for_multi_round_test_time_scaling = 0

            #####################
            # SCORING
            #####################
            is_pass_threshold, scoring_pros_cons = score_answer(question, messages)
            is_score_worse, streak_count = analyze_scores(scores)

            # score has been the same for streak_count
            if streak_count >= MAX_TRIES_TO_INCREASE_SCORE:
                is_pass_threshold = True # just end it, it's not better in the last 3 attempts

            # give it 1 more chance to get higher score
            if (is_score_worse or streak_count > 1) and llm_call_count_to_increase_score <= 1:
                llm_call_count_to_increase_score += 1
                scores.pop() # remove the last score
                messages.pop() # remove the last answer
                continue # go again
            else:
                llm_call_count_to_increase_score = 0

            # it still could not do it
            # TODO: take this out if you want to have it do manager feedback and keep trying
            if (is_score_worse or streak_count > 1):
                is_pass_threshold = True

            if is_pass_threshold:
                #####################
                # FINAL ANSWER
                #####################
                prompt = f"I conduct thorough research to create detailed and balanced investigative reports. I explore every avenue to produce comprehensive narratives, considering that the user might not be an expert in the domain, class, or task. I explain concepts clearly and informatively, being sensitive to the user's perspective without highlighting any lack of expertise. I carefully analyze the entire conversation, ensuring no detail is overlooked. With this in mind, I will write a comprehensive narrative report that addresses the Who, What, When, Where, How, and Why, without using these as section titles, as a text response.\n\nUser\'s Question\n\n{question}"
                final_answer = call_openai(prompt, OPENAI_USE_MODEL_SUMMARY, messages)
                return final_answer
            else:
                #####################
                # MANAGER FEEDBACK
                #####################
                manager_feedback = get_manager_feedback(question, assistant_content)
                if is_score_worse: # get rid of the last attempt and try again, we only want winners
                    #scores.pop()
                    #messages.pop()
                    revise_prompt = f"Your work has not improved, this is worse than the last work. Use more tools and revise your response based on your Manager's feedback:{manager_feedback}\n\n{scoring_pros_cons}"
                    messages.append({'role': 'user', 'content': revise_prompt})
                    continue
                elif (MAX_TRIES_TO_INCREASE_SCORE > 0) and (streak_count > 1):
                    revise_prompt = f"Your work needs to improve, this was no improvement over the last work. Use more tools and revise your response based on your Manager's feedback:{manager_feedback}\n\n{scoring_pros_cons}"
                    messages.append({'role': 'user', 'content': revise_prompt})
                    continue
                else:
                    revise_prompt = f"Use more tools and revise your response based on your Manager's feedback:{manager_feedback}\n\n{scoring_pros_cons}"
                    messages.append({'role': 'user', 'content': revise_prompt})
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




def main():
    # Reading from a file
    with open('deep_research_input.txt', 'r') as input_file:
        user_question = input_file.read()

    with open('deep_research_intermediate.txt', 'w') as output_file:
        output_file.write("")

    if REWRITE_THE_USER_QUERY:
        prompt = f"""
        Task:
        Transform the original user_question below into a detailed, organized format that fully captures its intent, scope, and context.

        Rules:
        1. Determine the primary goal of the query. Who, What, When, Where, How, and Why.
        2. If the query covers multiple themes or issues, break it down into separate sub-queries.
        3. For each sub-query:
            • Extract and list key details, context, and specific requirements.
            • Suggest a recommended approach or strategy for addressing it.
        4. Mitigate ambiguous elements with clarifying questions to resolve uncertainties.

        Output:
        In first person, provide clear rewritten version of the original query as sentences in a single paragraph that comprehensively reflects the user's intentions in an organized manner covering all the details of sub-queries and questions to clarify ambiguous elements.

        ```user_question
        {user_question}
        ```
        """
        user_question = call_openai(prompt, OPENAI_USE_MODEL)
    
    ai_role = """I provide deep research and comprehensively broad yet balanced investigative reports. I produce long narrative reports, I educate the user as they are not the domain, class, or task expert. I must explain things to the user knowing they don't have background or context in the domain, class, and task, in an informative way, and I never need to mention the user's lack of background or expertise."""

    if EXPAND_PERSONA_FOR_QUESTION:
        prompt = f"""Write 3 sequential sentences describing the characteristics of the perfect global renowned expert who can answer this question below. Response must be in first person. Response must be a single paragraph with the 3 sentences. Response must be only the description, no introduction,  explaination, rationale, formatting, or conclusion.
        
        ```question
        {user_question}
        ```
        """
        persona = call_openai(prompt, OPENAI_USE_MODEL_EXPANDED_REASONING)
        user_question = f"{ai_role}\n\n{persona}\n\nUser has asked this question: {user_question}"
    else:
        user_question = f"{ai_role}\n\nUser has asked this question: {user_question}"

    # Modify OPENAI_USE_MODEL const if you want to switch (e.g. "o1-mini")
    final_answer = call_research_professional(user_question, user_question, OPENAI_USE_MODEL)

    # Writing to a file
    with open('deep_research_ouput.txt', 'w') as output_file:
        output_file.write(final_answer)

    print("\n--- End of conversation ---")


if __name__ == "__main__":
    main()
