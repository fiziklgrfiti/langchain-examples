"""
Example demonstrating streaming output with Ollama and LangChain
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import warnings
warnings.filterwarnings("ignore")

def setup_streaming_chain():
    """
    Sets up a LangChain chain with streaming capability
    
    Returns:
        chain: A LangChain chain that supports streaming
    """
    # Initialize the Ollama LLM
    # Note: In newer LangChain versions, streaming is handled during invocation,
    # not during initialization
    llm = Ollama(
        model="gemma3:12b",
        temperature=0.7
        # 'stream' parameter is no longer passed here
    )
    
    # Create a prompt template
    prompt = PromptTemplate.from_template(
        "You are a cybersecurity expert writing a detailed report. "
        "Write a detailed analysis about {topic} with at least 3 sections. "
        "Include specific examples and best practices."
    )
    
    # Create a chain
    chain = prompt | llm | StrOutputParser()
    
    return chain

def stream_response(topic):
    """
    Streams the response token by token
    
    Args:
        topic (str): The topic to analyze
    """
    chain = setup_streaming_chain()
    
    print(f"\nStreaming analysis for: {topic}")
    print("-" * 50)
    
    # Use the stream method to get tokens one by one
    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
        time.sleep(0.01)  # Small delay to make streaming visible
        
    print("\n" + "-" * 50)
    print("Streaming complete!")

if __name__ == "__main__":
    topic = "security capabilities, focusing on strategic capability deliveries"
    stream_response(topic)