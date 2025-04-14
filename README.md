# Deep Research Assistant

Deep Research Assistant is an advanced Python-based research pipeline that leverages multiple large language model (LLM) service providers (OpenAI, Groq, DeepSeek, Together.ai, Fireworks.ai) and supplementary tools such as Perplexity AI and Firecrawl to perform comprehensive, multi-faceted research tasks. The repository is designed to transform user queries into detailed, organized investigations – rewriting queries for clarity, executing web searches and content retrieval, handling tool calls asynchronously, and iteratively refining responses based on evaluation scores. The final output is a long narrative research report intended to educate and inform users on complex subjects with precision and depth.

---

## Installation

1. **Clone the Repository:**

   git clone https://github.com/zestor/DeepResearcher.git  
   cd DeepResearcher

2. **Create a Virtual Environment (Optional but Recommended):**

   python3 -m venv venv  
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Install Dependencies:**
   pip install --upgrade pip
   pip install -r requirements.txt

*Note: Ensure that the `requirements.txt` file contains dependencies for requests, openai, firecrawl, groq (or any specific packages you use), and any other libraries (e.g., regex, datetime, threading).*

---

## Configuration

Before running the application, set the following environment variables with your API keys:

- **OPENAI_API_KEY**  
- **GROQ_API_KEY** (if using the GROQ service)
- **DEEPSEEK_API_KEY** (if using DEEPSEEK service)
- **FIREWORKS_API_KEY** (if using FIREWORKS.AI service)
- **TOGETHER_API_KEY** (if using TOGETHER.AI service)
- **PERPLEXITY_API_KEY**  
- **FIRECRAWL_API_KEY**

For example, on Linux or macOS:

   export OPENAI_API_KEY="your_openai_key"  
   export GROQ_API_KEY="your_groq_key"  
   export DEEPSEEK_API_KEY="your_deepseek_key"  
   export FIREWORKS_API_KEY="your_deepseek_key"  
   export TOGETHER_API_KEY="your_deepseek_key"  
   export PERPLEXITY_API_KEY="your_perplexity_key"  
   export FIRECRAWL_API_KEY="your_firecrawl_key"

Alternatively, create a `.env` file and use a package like python-dotenv to load these variables.

In addition, configure parameters in core/config.py

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
