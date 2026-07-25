import os

from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from langsmith import Client

client = Client()
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Langchain ReAct agent"

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}"

    response = requests.get(url)
    data = response.json()

    return f"Current temperature in {city}: {data}"


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True,
)  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm, tools=[search_tool, get_weather_data], prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True, max_iterations=5
)

# What is the release date of 3 idiots?
# What is the current temp of lahore
# Identify the birthplace city of salman khan (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of lahore"})
print(response)

print(response["output"])
