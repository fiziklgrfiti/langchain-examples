"""
Example of using LangChain with Ollama for simpler structured output
using a more reliable approach
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
import warnings
warnings.filterwarnings("ignore")

# Initialize the Ollama LLM
llm = Ollama(model="gemma3:12b", temperature=0.1)

# Create a simple output parser
parser = CommaSeparatedListOutputParser()

# Get the format instructions
format_instructions = parser.get_format_instructions()

# Create a prompt template that includes the format instructions
template = """
You are a cybersecurity expert. For the given security threat, provide a list of specific mitigation strategies.

{format_instructions}

Security Threat: {threat}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["threat"],
    partial_variables={"format_instructions": format_instructions}
)

# Create the chain
chain = prompt | llm | parser

def get_mitigation_strategies(threat):
    """
    Get a list of mitigation strategies for a security threat
    
    Args:
        threat (str): The security threat to address
        
    Returns:
        list: List of mitigation strategies
    """
    try:
        result = chain.invoke({"threat": threat})
        return result
    except Exception as e:
        print(f"Error: {e}")
        
        # Fallback to direct parsing
        try:
            raw_output = llm.invoke(prompt.format(threat=threat))
            # Try to extract items separated by commas or newlines
            items = [item.strip() for item in raw_output.replace('\n', ',').split(',')]
            # Filter out empty items
            return [item for item in items if item]
        except Exception as inner_e:
            print(f"Fallback error: {inner_e}")
            return []

if __name__ == "__main__":
    threat = "Accidentally adding a journalist to a signal chat discussing national confidential information (war plans)"
    print(f"Getting mitigation strategies for: {threat}")
    
    strategies = get_mitigation_strategies(threat)
    
    if strategies:
        print("\nRecommended Mitigation Strategies:")
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
    else:
        print("Failed to generate mitigation strategies.")
