"""config.py"""

import os
from enum import Enum, auto  # pylint: disable=no-name-in-module
from threading import Lock


class Service(Enum):
    """class"""

    DEEPSEEK = auto()
    OPENAI = auto()
    GROQ = auto()
    TOGETHER = auto()
    PERPLEXITY = auto()
    FIREWORKS = auto()


# Model Identifiers
DEEPSEEK_R1 = "deepseek-reasoner"
GROQ_DEEPSEEK_R1_LLAMA = "deepseek-r1-distill-llama-70b"
GROQ_LLAMA_4_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_QWEN_QWQ = "qwen-qwq-32b"
GROQ_LLAMA_3_VERSATILE = "llama-3.3-70b-versatile"
TOGETHER_LLAMA_4_MAVERICK = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
FIREWORKS_LLAMA_4_MAVERICK = "accounts/fireworks/models/llama4-maverick-instruct-basic"
OPENAI_O3_MINI = "o3-mini"
OPENAI_O1_MINI = "o1-mini"
OPENAI_O1 = "o1"
OPENAI_GPT_4O = "gpt-4o"
PERPLEXITY_SONAR = "sonar"
PERPLEXITY_SONAR_PRO = "sonar-pro"
PERPLEXITY_SONAR_REASONING = "sonar-reasoning"
PERPLEXITY_SONAR_REASONING_PRO = "sonar-reasoning-pro"

#####################################################
# SET THIS STUFF FOR YOUR USE CASE
#####################################################
# REASONING
USE_SERVICE_REASONING = Service.GROQ
DEEPSEEK_USE_MODEL = DEEPSEEK_R1
GROQ_USE_MODEL = GROQ_QWEN_QWQ  # or GROQ_DEEPSEEK_R1_LLAMA
OPENAI_USE_MODEL = OPENAI_O3_MINI  # Change as desired
TOGETHER_USE_MODEL = TOGETHER_LLAMA_4_MAVERICK
FIREWORKS_USE_MODEL = FIREWORKS_LLAMA_4_MAVERICK

# SCORING
USE_SERVICE_SCORING = Service.GROQ
OPENAI_USE_MODEL_SCORING = OPENAI_GPT_4O
LLM_USE_MODEL_SCORING = GROQ_LLAMA_4_MAVERICK

# MANAGER FEEDBACK
USE_SERVICE_FEEDBACK = Service.GROQ
OPENAI_USE_MODEL_FEEDBACK = OPENAI_GPT_4O
LLM_USE_MODEL_FEEDBACK = GROQ_LLAMA_4_MAVERICK

# SUMMARY
USE_SERVICE_SUMMARY = Service.OPENAI
OPENAI_USE_MODEL_SUMMARY = OPENAI_O1_MINI
LLM_USE_MODEL_SUMMARY = GROQ_LLAMA_3_VERSATILE

ANSWER_QUALITY_THRESHOLD = 1.0  # 0.00 - 1.00
MODELS_WITH_TOOL_USAGE = {OPENAI_O1, OPENAI_O3_MINI}
MAX_TOOL_PARALLEL_THREADS = 20
MAX_TRIES_TO_INCREASE_SCORE = 3  # number of times with same score

# PERPLEXITY SETTINGS
DEFAULT_PERPLEXITY_MODEL = PERPLEXITY_SONAR  # https://docs.perplexity.ai/guides/pricing
PERPLEXITY_MODELS_WITH_SEARCH_CONTENT_SIZE = {
    PERPLEXITY_SONAR_REASONING_PRO,
    PERPLEXITY_SONAR_REASONING,
    PERPLEXITY_SONAR_PRO,
    PERPLEXITY_SONAR,
}
PERPLEXITY_MODELS_WITH_REASONING_TOKENS = {
    PERPLEXITY_SONAR_REASONING_PRO,
    PERPLEXITY_SONAR_REASONING,
}
PERPLEXITY_SEARCH_CONTENT_SIZE = "high"  # high, medium, low

# RETRY SETTINGS
LLM_RETRY_WAIT_TIME = 20  # in seconds
LLM_RETRY_COUNT = 5

# USE THINK TWICE SETTINGS
USE_MULTI_ROUND_TEST_TIME_SCALING = False
MAX_TRIES_FOR_TEST_TIME_SCALING = 2

# EXPANDED REASONING
USE_REASONING_EXPANSION = False
MAX_QUESTIONS_FOR_REASONING_EXPANSION = 3  # identified from <think> text
USE_SERVICE_EXPANDED_REASONING = Service.OPENAI
OPENAI_USE_MODEL_EXPANDED_REASONING = OPENAI_GPT_4O
LLM_USE_MODEL_EXPANDED_REASONING = TOGETHER_USE_MODEL
if USE_SERVICE_REASONING == Service.OPENAI:
    USE_REASONING_EXPANSION = False  # OpenAI doesn't give reasoning tokens

# Markers for think tokens
THINK_START = "<think>"
THINK_END = "</think>"

# API Keys (ensure these are set in your environment or replace the default placeholder)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "...")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "...")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "...")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "...")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "...")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "...")

# Global Variables
research_questions = []
scores = []
GRAND_TOTAL_COST = 0

# A global threading lock for critical sections
lock = Lock()
