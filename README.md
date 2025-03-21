# Deep Research Assistant

Deep Research Assistant is an advanced Python-based research pipeline that leverages multiple large language model (LLM) service providers (DeepSeek, Groq, OpenAI) and supplementary tools such as Perplexity AI and Firecrawl to perform comprehensive, multi-faceted research tasks. The repository is designed to transform user queries into detailed, organized investigations – rewriting queries for clarity, executing web searches and content retrieval, handling tool calls asynchronously, and iteratively refining responses based on evaluation scores. The final output is a long narrative research report intended to educate and inform users on complex subjects with precision and depth.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Multiple LLM Service Support:** Easily switch between DeepSeek, Groq, and OpenAI by changing a configuration variable.
- **Adaptive Query Rewriting:** Automatically rewrites and expands the user’s original query to capture intent, scope, and context.
- **Tool Integration:** Integrates with web search (via Perplexity AI), content retrieval (via Firecrawl), and additional custom tools.
- **Iterative Research and Scoring:** Uses iterative loops with scoring and manager feedback mechanisms to refine research responses.
- **Concurrent Tool Calls:** Uses multithreading (ThreadPoolExecutor) to process multiple tool calls concurrently.
- **Dynamic Reasoning Expansion:** Optionally expands the reasoning process to extract and answer implicit and sub-questions.
- **Robust JSON Handling:** Contains helper functions to parse, validate, and clean JSON data, ensuring smooth integration of model outputs.
- **Detailed Logging:** Logs intermediate steps and scores to an output file for debugging and transparency.

---

## Architecture Overview

The core functionality is implemented in a single Python script, which features:

- **Configuration Section:**  
  Set service provider, model names, thresholds, and other parameters at the beginning of the code.
  
- **LLM Calls and Tool Functions:**  
  Functions like `call_llm`, `call_openai`, `call_web_search_assistant`, and `call_web_content_retriever` communicate with external APIs. They handle JSON formatting, error retries, and adapt based on the chosen service.
  
- **Reasoning and Scoring Loop:**  
  The central function `call_research_professional` uses an iterative ReAct loop. It collects messages, issues tool calls, processes responses, and refines answers by comparing evaluation scores.
  
- **Helper and Utility Functions:**  
  Several helper functions (e.g., `parse_reasoning_from_text`, `remove_think_text`, `fix_json`, etc.) handle specific tasks such as JSON conversion, string manipulation, and logging.
  
- **Main Functionality:**  
  The `main()` function reads the original user query from `deep_research_input.txt`, optionally rewrites it for clarification, prepares the AI persona context, and initiates the research process. Final outputs are written to `deep_research_ouput.txt`.

---

## Installation

1. **Clone the Repository:**

   git clone https://github.com/zestor/DeepResearcher.git  
   cd DeepResearcher

2. **Create a Virtual Environment (Optional but Recommended):**

   python3 -m venv venv  
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Install Dependencies:**

   pip install -r requirements.txt

*Note: Ensure that the `requirements.txt` file contains dependencies for requests, openai, firecrawl, groq (or any specific packages you use), and any other libraries (e.g., regex, datetime, threading).*

---

## Configuration

Before running the application, set the following environment variables with your API keys:

- **OPENAI_API_KEY**  
- **GROQ_API_KEY** (if using the GROQ service)
- **PERPLEXITY_API_KEY**  
- **FIRECRAWL_API_KEY**

For example, on Linux or macOS:

   export OPENAI_API_KEY="your_openai_key"  
   export GROQ_API_KEY="your_groq_key"  
   export PERPLEXITY_API_KEY="your_perplexity_key"  
   export FIRECRAWL_API_KEY="your_firecrawl_key"

Alternatively, create a `.env` file and use a package like python-dotenv to load these variables.

In addition, configure parameters at the top of the main script:

- **USE_SERVICE**: Change between `Service.DEEPSEEK`, `Service.GROQ`, and `Service.OPENAI`.
- **ANSWER_QUALITY_THRESHOLD**: Set the minimum rating threshold required.
- **Model Configuration:** Set the model names for each service (e.g., `OPENAI_USE_MODEL`, `GROQ_USE_MODEL`).
- **Tool Parallelism:** Limit the max parallel threads for tool calls using `MAX_TOOL_PARALLEL_THREADS`.

---

## Usage

1. **Prepare the Input:**  
   Write your research query in `deep_research_input.txt`.  
   *Tip:* The query can be a rough question; the system will rewrite and expand it.

2. **Run the Script:**

   python main.py

3. **Output:**  
   - The final narrative answer is written to `deep_research_ouput.txt`.  
   - Intermediate details and scores are logged to `deep_research_intermediate.txt` for review.

4. **Customize Your Experience:**  
   Adjust configuration variables or extend the tool functions for specialized research needs.

---

## File Structure

   deep-research-assistant/  
   ├── deep_research_input.txt        // Input query file  
   ├── deep_research_ouput.txt        // Final output report  
   ├── deep_research_intermediate.txt // Log file for intermediate steps  
   ├── main.py                        // Main script containing all code  
   ├── requirements.txt               // List of required packages  
   └── README.md                      // This documentation file

---

## Contributing

Contributions are welcome! If you want to report bugs, suggest improvements, or add new features, please do the following:

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Commit your changes and open a pull request with a detailed description of your modifications.
4. Ensure that your code adheres to existing style guidelines and that new functionality is well documented.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgements

- Thanks to the providers of DeepSeek, Groq, and OpenAI for their powerful APIs.
- Inspired by the need for accurate, in-depth research tools that integrate modern LLM capabilities.
- Special thanks to contributors and testing users who provided feedback.

---

Now you’re ready to explore the capabilities of the Deep Research Assistant and leverage advanced LLM-powered research for your projects. Happy researching!
