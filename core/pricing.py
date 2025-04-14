"""pricing.py"""

from core.config import (
    Service,
    GROQ_DEEPSEEK_R1_LLAMA,
    GROQ_LLAMA_4_SCOUT,
    GROQ_LLAMA_4_MAVERICK,
    GROQ_QWEN_QWQ,
    GROQ_LLAMA_3_VERSATILE,
    TOGETHER_LLAMA_4_MAVERICK,
    FIREWORKS_LLAMA_4_MAVERICK,
    OPENAI_O3_MINI,
    OPENAI_O1_MINI,
    OPENAI_GPT_4O,
    PERPLEXITY_SONAR,
    PERPLEXITY_SONAR_PRO,
)


def get_model_pricing(
    service: Service, model_id: str, perplexity_content_size: str = None
):
    """
    Retrieves pricing details for a given model and service.

    Args:
        service (Service): The AI service.
        model_id (str): The model identifier.
        perplexity_content_size (str, optional): The content size for Perplexity models.

    Returns:
        Tuple(float, float, float): input_price, output_price, additional_per_request.
    """
    perplexity_content_pricing = {
        PERPLEXITY_SONAR: {"low": 5, "medium": 8, "high": 12},
        PERPLEXITY_SONAR_PRO: {"low": 6, "medium": 10, "high": 14},
    }
    models_pricing = {
        Service.FIREWORKS: {
            FIREWORKS_LLAMA_4_MAVERICK: {"input_price": 0.22, "output_price": 0.88},
            "accounts/fireworks/models/llama4-scout-instruct-basic": {
                "input_price": 0.15,
                "output_price": 0.60,
            },
        },
        Service.PERPLEXITY: {
            PERPLEXITY_SONAR: {"input_price": 1.0, "output_price": 1.0},
            PERPLEXITY_SONAR_PRO: {"input_price": 3.0, "output_price": 15.0},
        },
        Service.GROQ: {
            GROQ_LLAMA_4_SCOUT: {"input_price": 0.11, "output_price": 0.34},
            GROQ_LLAMA_4_MAVERICK: {"input_price": 0.50, "output_price": 0.77},
            GROQ_DEEPSEEK_R1_LLAMA: {"input_price": 0.75, "output_price": 0.99},
            "deepseek-r1-distill-qwen-32b": {"input_price": 0.69, "output_price": 0.69},
            "qwen-2.5-32b": {"input_price": 0.79, "output_price": 0.79},
            "qwen-2.5-coder-32b": {"input_price": 0.79, "output_price": 0.79},
            GROQ_QWEN_QWQ: {"input_price": 0.29, "output_price": 0.39},
            "mistral-saba-24b": {"input_price": 0.79, "output_price": 0.79},
            "llama3-70b-8192": {"input_price": 0.59, "output_price": 0.79},
            "llama3-8b-8192": {"input_price": 0.05, "output_price": 0.08},
            "llama-3.3-70b-specdec": {"input_price": 0.59, "output_price": 0.99},
            "llama-3.2-1b-preview": {"input_price": 0.04, "output_price": 0.04},
            "llama-3.2-3b-preview": {"input_price": 0.06, "output_price": 0.06},
            "gemma2-9b-it": {"input_price": 0.20, "output_price": 0.20},
            GROQ_LLAMA_3_VERSATILE: {"input_price": 0.59, "output_price": 0.79},
        },
        Service.OPENAI: {
            OPENAI_O1_MINI: {"input_price": 1.10, "output_price": 4.40},
            OPENAI_O3_MINI: {"input_price": 1.10, "output_price": 4.40},
            OPENAI_GPT_4O: {"input_price": 2.50, "output_price": 10.00},
        },
        Service.DEEPSEEK: {
            # Add DeepSeek-specific models and their pricing if necessary
        },
        Service.TOGETHER: {
            TOGETHER_LLAMA_4_MAVERICK: {"input_price": 0.27, "output_price": 0.85},
            GROQ_LLAMA_4_SCOUT: {"input_price": 0.18, "output_price": 0.59},
            "deepseek-ai/DeepSeek-V3": {"input_price": 1.25, "output_price": 1.25},
        },
    }
    perplexity_additional_per_request = 0
    if service == Service.PERPLEXITY:
        perplexity_content_pricing_dict = perplexity_content_pricing.get(model_id)
        if (
            perplexity_content_pricing_dict
            and perplexity_content_size in perplexity_content_pricing_dict
        ):
            perplexity_additional = perplexity_content_pricing_dict[
                perplexity_content_size
            ]
            perplexity_additional_per_request = perplexity_additional / 1000
    service_pricing = models_pricing.get(service)
    if not service_pricing:
        return 0, 0, 0
    model_pricing = service_pricing.get(model_id)
    if model_pricing:
        input_price = model_pricing.get("input_price")
        output_price = model_pricing.get("output_price")
        if input_price != "Not Available" and output_price != "Not Available":
            return input_price, output_price, perplexity_additional_per_request
        else:
            return 0, 0, 0
    else:
        return 0, 0, 0
