"""prompt_getter.py"""

import os
import logging

# Set up the logger for this module with lazy evaluation
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


class PromptGetter:
    """
    A class to retrieve and format markdown prompt files.
    """

    @staticmethod
    def get_prompt(filename: str, **kwargs) -> str:
        """
        Read a markdown file from the prompt directory, format it with provided keyword arguments,
        and return the formatted prompt string.

        Args:
            filename (str): The name of the markdown file located in the prompt_directory.
            **kwargs: Keyword arguments to substitute into the markdown file content.

        Returns:
            str: The formatted prompt string.

        Raises:
            FileNotFoundError: If the specified prompt file does not exist.
            IOError: If there is an error reading the file.
            KeyError: If a placeholder required for formatting is missing in kwargs.
        """
        prompt_directory = "./core/prompts"

        filepath = os.path.join(prompt_directory, filename)
        logger.debug("Attempting to read prompt file: %s", filepath)

        if not os.path.isfile(filepath):
            logger.error("Prompt file not found: %s", filepath)
            raise FileNotFoundError(f"Prompt file '{filepath}' not found.")

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
            logger.debug("Successfully read prompt file: %s", filepath)
        except IOError as io_err:
            logger.error("Error reading file %s: %s", filepath, io_err)
            raise IOError(f"Error reading file '{filepath}': {io_err}") from io_err

        try:
            formatted_prompt = content.format(**kwargs)
            logger.debug("Successfully formatted the prompt with provided parameters.")
        except KeyError as key_err:
            logger.error("Placeholder missing for formatting: %s", key_err)
            raise KeyError(
                f"Missing placeholder for formatting: {key_err}"
            ) from key_err
        except Exception as fmt_err:
            logger.error("Unexpected error during prompt formatting: %s", fmt_err)
            raise fmt_err

        return formatted_prompt
