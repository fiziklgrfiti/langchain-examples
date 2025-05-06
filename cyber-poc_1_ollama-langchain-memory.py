"""
Example demonstrating conversational memory with Ollama and LangChain
with improved multi-line input handling
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
        model="llama3.2:latest",
        temperature=0.7
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    
    # Create a chat prompt template with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """
            You are CyberChad, a true blue Australian legend & AI cyber security expert assistant. You work for an organsation with a LOW appetite for cyber security risk.
            
            Provide friendly, helpful, accurate, and concise information about cyber security services for projects. 
            
            Maintain context from the conversation history.

            -Goal-
            Determine what cyber security assurance services are applicable for a project.

            YOUR TASK:
            Ask for project details from a project team subject matter expert.
            Parse their project details & verify that you have understood the project correctly.
            Then, determine what cyber security services from our cyber security services catalog are most applicable.
            If you detect that the project is due within 4 weeks, then immediately notify the subject matter expert that this will likely require managerial expedition & additional costs.
            Propose an initial recommendation on which cyber security services are recommended and why.
            Your recommendation on cyber security services should be concise. Explanations on why should be no more then 4 sentences.
            
            CYBER SERCURITY SERVICE CATALOG:
            1. Secure Design review: This service reviews solution designs, pre-requisite is to have a solution design. Not recommended if the solution has already been built.
                - Needs up to 2 weeks to organise & schedule in to start.
            2. Threat Model: This service provides a threat model for a solution. The threat model will utilise Microsoft's STRIDE threat model, outline trust boundaries, describe threat vectors & recommend controls that mitigate threats. Not recommended if a threat model already exists.
                - Needs up to 2 weeks to organise & schedule in to start.
            3. Security Control Test Plan: This service will create a security control test & ensure that all necessary controls are in place. Pre-requisite is to have a threat model that can be referenced.
                - Needs up to 4 weeks to organise & schedule in to start.
            4. Application Penetration Test (including Web & API): This service simulates a hacker trying to find & exploit weaknesses in an application, pre-requisite is to have a solution built. Not recommended if the solution hasn't yet been built.
                - Needs up to 8 weeks to organise & schedule in to start.
            5. Network Penetration Test: This service simulates a hacker trying to find & exploit weaknesses across internal OR external networks, pre-requisite is to have a solution built. Not recommended if the solution hasn't yet been built.
                - Needs up to 8 weeks to organise & schedule in to start.
            6. Cloud Penetration Test: This service simulates a hacker trying to find & exploit weaknesses across cloud assets, pre-requisite is to have a solution built. Not recommended if the solution hasn't yet been built.
                - Needs up to 8 weeks to organise & schedule in to start.

            INSTRUCTIONS:
            1. Ask clarifying questions about the project until you have a sufficient understanding.
            2. If the subject matter expert can't disclose clarifying information, then simply recommend that they adequately prepare the necessary information & then start again:
            3. If the subject matter expert can disclose sufficent informationthen then:
            - Parse the project details.
            - Establish cyber security requirements from the project details.
            - Link the cyber security requirements to cyber security services.
            4. Recommend what cyber security services are needed to adequately assure their project.            
            5. After you have provided your recommendation for cyber security services for a project, ask if there's any further assistance that's needed from the subject matter expert.
            6. Otherwise if finished, suggest that someone from the Cyber Security team will help organise the specific engagements (services)
            
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm
    
    return chain, memory

def get_multiline_input():
    """
    Gets multi-line input from the user.
    The user can type '####' on a new line to finish their input.
    
    Returns:
        str: The complete multi-line input
    """
    print("\nYou: ", end="", flush=True)
    lines = []
    delimiter = "####"
    
    while True:
        line = input()
        # If the line is our delimiter, we're done
        if line == delimiter:
            break
        # If it's a special command like 'exit', return it immediately
        if line.lower() == '## exit ##' and not lines:
            return 'exit'
        # Otherwise add this line to our input
        lines.append(line)
        # Show continuation prompt
        print("... ", end="", flush=True)
    
    return "\n".join(lines)

def interactive_chat():
    """
    Runs an interactive chat session with memory
    """
    chain, memory = setup_conversational_chain()
    
    print("""
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠋⠉⠉⠙⠙⠋⠛⠙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀⣀⣤⣤⣴⣴⣶⣶⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⣴⣾⣾⣿⣿⣿⣵⣿⣿⣿⣿⣿⣷⣾⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⣰⣾⡷⣿⡿⢟⣋⣿⣶⣿⣿⣿⣯⣙⣥⣿⣟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣩⣴⠁⣿⣾⣿⣿⣿⠿⠟⠛⠉⠉⠻⢿⣜⢿⣷⣆⡀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣸⣿⠿⡇⢸⠟⠋⠉⠀⠀⣀⣠⣤⣤⣤⣀⡙⡎⢻⣿⣯⣀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀⠁⠀⠀⠀⢰⣴⣤⠂⣰⡞⠉⠁⠀⣈⡙⠿⣛⣤⣿⣿⣿⣿⡄⠀⠀⠀⢀⡀⡀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⣶⠆⠓⠢⣸⡟⣿⣦⣸⣀⢴⡶⣟⣫⣥⣸⣿⣿⠿⣿⣿⣿⡃⠀⠀⡴⠈⢶⡈⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠁⣠⡤⢠⣿⣼⣿⣏⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤⣽⣿⣿⠀⠀⠀⡃⠙⢧⡱⣆⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⣤⣥⢂⣾⣿⣿⢛⣫⡌⠛⣟⣻⣛⣛⠿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⡟⢀⣾⠇⡇⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠠⣿⣿⡈⠛⠋⠀⠠⠀⠠⠿⠶⣍⡙⡙⠻⣦⠹⠛⠛⠃⠀⠀⠀⠀⠀⢹⣩⣶⡎⣱⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⢿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⢘⡆⢼⠇⠀⠀⠀⠀⠀⠀⠀⠀⠈⣋⡥⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠈⠗⠀⠀⠀⠀⠤⠐⣀⢠⠉⢈⡄⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣩⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⢤⣀⣓⣂⡭⠶⣶⠾⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢘⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠈⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⡟⣿⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⣤⠾⠁⢠⣿⡅⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠛⠛⠉⠉⠉⠉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢀⡼⠂⣴⠯⠃⢠⣿⣿⣷⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠛⢿⣿⣿⣿⣿⠟⠉⠀⠀⠀⠀⠀⠀⠈⠠⡀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣞⡇⢸⠃⠜⣋⣴⡏⣼⣿⣿⣿⣷⡈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠈⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡁⢒⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⣻⠞⢱⡎⣰⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣂⠻⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢊⠥⠀⠀⠹⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣤⣤⡀⠠⡄⢾⣹⣟⡽⣞⡽⠂⢽⢠⣿⣿⣿⣿⣁⣹⣿⣿⣿⣿⣿⣿⡹⣿⣶⣦⣭⣍⣛⠻⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠄⠈⠹⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢏⠀⡀⠀⠈⠻⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⡇⢣⡘⢷⣮⡽⠽⠃⢠⡏⣼⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣷⡹⣿⣿⣿⣿⣿⣿⣷⣶⣬⣍⣛⠿⢿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢺⠐⡁⠀⠀⠀⠀⠻⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢼⡀⣧⣌⢲⣶⣾⡇⣨⠁⣿⣿⣿⣿⣿⣇⠸⣿⣿⣿⣿⣿⣿⣿⣷⣤⡙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣬⡙⠋⢁⡴⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣉⠖⢠⠀⠀⠀⠀⠀⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⣸⣷⠹⣿⣧⡹⣿⣷⣿⢀⣿⣿⣿⣿⣿⣿⣇⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣄⡙⣌⢿⠟⢿⣿⣿⣿⣿⣿⣧⠀⠉⢐⡻⢆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠐⢈⠠⠁⠀⠀⠀⠀⠀⠈⢿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣋⣥⣶⡆⢽⣿⣇⢻⣿⣷⣹⣿⡏⢰⣿⣿⣿⣿⢛⣿⣿⣇⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣷⣄⡝⢿⠻⣿⣿⣿⣧⠀⠀⠈⡉⡖⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⢸
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢋⣰⣾⣿⣿⡟⠁⣾⡿⢻⡇⢻⣿⣿⣿⡇⢸⣿⣿⣿⡟⠈⣿⣿⣿⣷⣄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⢛⠻⠿⠦⠕⢼⣿⡿⠿⠀⠀⠀⠐⡉⠒⠤⢠⠀⡄⠠⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣩⣾⣿⣿⢻⠟⠛⣤⢸⣿⠇⠘⣿⡈⣿⣿⣿⢃⣿⣿⣿⣿⣇⢀⣿⣿⣿⣿⠿⠳⢛⣛⣛⣋⣭⣭⣭⣥⣤⣬⣍⣛⣛⠻⠿⠟⣢⣥⣾⣿⣷⠀⠀⠀⠈⡁⠎⠀⠀⢄⠱⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣣⣾⢿⢟⡙⣡⣫⣾⣿⣧⣾⣿⣄⣰⣿⣧⢹⣿⡟⢸⣿⣿⣿⣿⡿⢟⣛⣭⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡙⠿⣿⣿⣿⣷⡀⠀⠀⠀⠈⠀⠀⡀⠂⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⣛⣋⣐⡻⠣⢦⣚⡛⢛⣩⣥⣶⣶⣶⣶⣮⣍⣛⠿⣀⣿⡇⣼⡿⢟⣩⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣮⡻⢿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿
⣿⣿⣿⣿⣿⣿⣿⡿⠿⢛⣛⣛⠋⣡⣖⣛⣭⣭⣭⠍⣥⡾⢛⣭⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡙⠁⣋⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣮⣙⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿
⣿⣿⣿⡿⠟⣋⣥⣶⣿⣿⢋⣥⣾⣿⣿⣿⣿⡿⣣⡾⣫⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢸⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠈⠈⠀⠀⠀⠀⠀⠀⠀⠀⢸
⡿⠟⣫⠴⣿⣿⣿⣿⢟⣴⣿⣿⣿⣿⣿⣿⠟⡴⢫⣾⣿⣿⣯⣭⣭⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣽⠰⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⢒⠈⠀⠀⠀⠀⠀⠀⠘
    """)
    print("CyberChad(AUS) AI Assistant")
    print("Type '## exit ##' to end the conversation")
    print("Type '####' on a new line to complete your message")
    print("-" * 50)
    
    while True:
        user_input = get_multiline_input()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Prepare the input with memory
        inputs = {
            "input": user_input,
            "chat_history": memory.chat_memory.messages
        }
        
        # Generate response
        print("\n🗿: ", end="", flush=True)
        
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