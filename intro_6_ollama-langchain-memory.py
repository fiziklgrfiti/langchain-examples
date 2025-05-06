"""
Example demonstrating conversational memory with Ollama and LangChain
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import warnings
warnings.filterwarnings("ignore")

def setup_conversational_chain():
    """
    Sets up a conversational chain with memory
    
    Returns:
        tuple: (chain, memory) - The chain and memory objects for the conversation
    """
    # Initialize the Ollama LLM
    llm = Ollama(
        model="gemma3:12b",
        temperature=0.7
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    
    # Create a chat prompt template with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are CyberChad, an AI cybersecurity expert assistant. Provide helpful, accurate, and concise information about cybersecurity topics. Maintain context from the conversation history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm
    
    return chain, memory

def interactive_chat():
    """
    Runs an interactive chat session with memory
    """
    chain, memory = setup_conversational_chain()
    
    print("CyberChad AI Assistant")
    print("Type 'exit' to end the conversation")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Prepare the input with memory
        inputs = {
            "input": user_input,
            "chat_history": memory.chat_memory.messages
        }
        
        # Generate response
        print("\nCyberChad: ", end="", flush=True)
        
        # In newer LangChain versions, the response might be a string directly
        response = chain.invoke(inputs)
        
        # Handle both string responses and object responses with content attribute
        if hasattr(response, 'content'):
            print(response.content)
            ai_message = response.content
        else:
            print(response)
            ai_message = response
        
        # Update memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(ai_message)

if __name__ == "__main__":
    interactive_chat()