import os
import logging
from dotenv import load_dotenv
from datasets import load_dataset
from rich.logging import RichHandler

from distilabel.models.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


load_dotenv(override=True)

with Pipeline() as pipeline:
    TextGeneration(
        llm=OpenAILLM(
            base_url=os.getenv("OPENAI_API_BASE"),
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("OPENAI_API_KEY"),
            generation_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        ),
    )

if __name__ == "__main__":
    dataset = load_dataset("distilabel-internal-testing/instructions", split="test")
    distiset = pipeline.run(dataset=dataset, use_cache=False)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    distiset.push_to_hub(repo_id="minpeter/distilabel-example")
