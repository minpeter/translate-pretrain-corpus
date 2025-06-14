import os

from distilabel.models.llms import OpenAILLM
from dotenv import load_dotenv

load_dotenv(override=True)

llm = OpenAILLM(
    base_url=os.getenv("OPENAI_API_BASE"),
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 512},
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
print(output)
