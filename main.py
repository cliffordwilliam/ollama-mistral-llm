# This is a solid example of making an agent
# Agent is a software that uses a model + tool of any kind to achieve a goal and it uses multi action
# Model: Mistal LLM (understandable human language guesser)
# Tool: pokeapi
# Goal: "Tell me about Pikachu including its stats."
# Multi actions:
# 1. Deciding to use a tool, the pokeapi
# 2. Processing http response tool result
# 3. Stopping once the goal is met (it may retry also)

from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import requests  # type: ignore


# Actual Tool: This is what the agent can call
def get_pokemon_data(pokemon_name: str) -> str:
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        stats = {stat["stat"]["name"]: stat["base_stat"] for stat in data["stats"]}
        return (
            f"{data['species']['name']} has a height of {data['height']} decimetres.\n"
            f"Stats: {stats}"
        )
    else:
        return f"Sorry, I couldn't find data for {pokemon_name}."


# Wrap that function into a LangChain Tool
pokemon_tool = Tool(
    name="PokeAPIFetcher",
    func=get_pokemon_data,
    description="Useful when you need to get detailed info about a specific Pokémon by name.",
)


# Create the LLM instance
llm = OllamaLLM(model="mistral")


# Create the actual agent (tools + llm)
agent = initialize_agent(
    tools=[pokemon_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# Now run the agent with a goal!
response = agent.invoke({"input": "Tell me about Pikachu including its stats."})

print(response)
