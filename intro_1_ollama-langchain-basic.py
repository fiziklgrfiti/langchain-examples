"""
Basic example of using LangChain with Ollama to interact with gemma3:12b model
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import warnings
warnings.filterwarnings("ignore")


# Initialize the Ollama LLM with gemma3:12b model
llm = Ollama(model="gemma3:12b")

# Create a simple prompt template
prompt = PromptTemplate.from_template(
    "You are a cybersecurity expert. Briefly explain the concept of {concept} in 2-3 sentences."
)

# Create a simple chain: prompt -> llm -> output parser
chain = prompt | llm | StrOutputParser()

# Execute the chain with a specific concept
response = chain.invoke({"concept": "zero trust architecture"})

# Print the response
print(response)
