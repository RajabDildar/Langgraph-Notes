import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = (
    "sequential llm app langchain"  # ideally we should make a seperate folder and make seperate project on langsmith. but since this is a demo and we are in a single folder, so we are overriding the environment variable project name here so that we can make a seperate project for this file on langsmith.
)

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"],
)

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

parser = StrOutputParser()

config = {
    "run_name": "sequential chain",
    "tags": ["llm app", "report generation", "summarization"],
    "metadata": {
        "model": "llama-3.3-70b-versatile",
        "model_temp": 0.5,
        "parser": "stroutputparser",
    },
}

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "AI in healthcare"}, config=config)

print(result)
