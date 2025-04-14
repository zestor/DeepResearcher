"""main.py"""

import time

from core.research_professional import call_research_professional
from core.prompt_getter import PromptGetter
from core.llm_helpers import call_llm
from core.config import Service, GROQ_LLAMA_4_MAVERICK


def main():
    """main function"""
    # Read the user's question from a file.
    try:
        with open("deep_research_input.txt", "r", encoding="utf-8") as input_file:
            user_question = input_file.read()
    except IOError:
        print("Error reading input file.")
        return

    # Clear intermediate log file.
    try:
        with open(
            "deep_research_intermediate.txt", "w", encoding="utf-8"
        ) as output_file:
            output_file.write("")
    except IOError:
        print("Error clearing the intermediate file.")
    # Construct the prompt with a persona.
    prompt = PromptGetter.get_prompt(
        "perfect_persona_to_answer.md", user_question=user_question
    )
    persona = call_llm(prompt, GROQ_LLAMA_4_MAVERICK, Service.GROQ)
    full_question = PromptGetter.get_prompt(
        "create_full_question.md", persona=persona, user_question=user_question
    )
    final_answer = call_research_professional(full_question, full_question)
    try:
        content_with_newlines = final_answer.replace("\\n", "\n")
        with open("deep_research_ouput.txt", "w", encoding="utf-8") as file:
            file.write(content_with_newlines)
    except IOError:
        print("Error writing the final output.")
    print("\n--- End of conversation ---")


if __name__ == "__main__":
    start_ns = time.perf_counter_ns()
    try:
        main()
    finally:
        end_ns = time.perf_counter_ns()
        elapsed_sec = (end_ns - start_ns) / 1_000_000_000
        elapsed_min = elapsed_sec / 60
        print(f"Executed in {elapsed_sec:.2f} seconds ({elapsed_min:.2f} minutes)")
