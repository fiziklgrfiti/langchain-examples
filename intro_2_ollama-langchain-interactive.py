"""
Interactive example of using LangChain with Ollama to interact with gemma3:12b model
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import warnings
warnings.filterwarnings("ignore")

def setup_ollama_chain(model_name="gemma3:12b", temperature=0.7):
    """
    Sets up a LangChain chain using Ollama
    
    Args:
        model_name (str): Name of the Ollama model to use
        temperature (float): Temperature parameter for the model (0.0-1.0)
        
    Returns:
        chain: A LangChain chain ready to be invoked
    """
    # Initialize the Ollama LLM with specified model and parameters
    llm = Ollama(
        model=model_name,
        temperature=temperature,
        # Optional: Adjust these parameters based on your needs
        # num_predict=256,  # Maximum number of tokens to predict
        timeout=60,       # Request timeout in seconds
    )
    
    # Create a cybersecurity expert prompt template
    prompt = PromptTemplate.from_template(
        "You are a cybersecurity expert. Provide a clear explanation about {query} in 2-3 sentences."
    )
    
    # Create a chain: prompt -> llm -> output parser
    chain = prompt | llm | StrOutputParser()
    
    return chain

def run_interactive_session():
    """Runs an interactive session with the Ollama model via LangChain"""
    print("Setting up Ollama with gemma3:12b model...")
    chain = setup_ollama_chain()
    print("Setup complete! You can now interact with the model.")
    print("Type 'exit' to quit the session.")
    
    while True:
        query = input("\nEnter your cybersecurity question: ")
        if query.lower() == 'exit':
            print("Exiting session.")
            break
        
        print("\nGenerating response...")
        try:
            response = chain.invoke({"query": query})
            print("\nResponse:")
            print(response)
        except Exception as e:
            print(f"Error occurred: {e}")
            
if __name__ == "__main__":
    run_interactive_session()
