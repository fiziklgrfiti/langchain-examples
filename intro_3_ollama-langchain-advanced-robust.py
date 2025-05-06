"""
Ultra-robust example for structured output with Ollama models
Using a step-by-step approach instead of direct JSON generation
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List, Any
import re
import warnings
warnings.filterwarnings("ignore")

def setup_ollama_model():
    """Initialize the Ollama model with appropriate parameters"""
    return Ollama(
        model="gemma3:12b",
        temperature=0.1,  # Low temperature for consistency
        num_predict=2048  # Generous token limit
    )

def extract_field(text: str, field_name: str) -> str:
    """Extract a field value from text using regex patterns"""
    # Try different patterns that might match field responses
    patterns = [
        rf"{field_name}:\s*(.*?)(?:\n|$)",  # Field: value
        rf"{field_name}[:\-–—]\s*(.*?)(?:\n|$)",  # Field - value
        rf"{field_name}\s*[:\-–—]\s*(.*?)(?:\n|$)",  # Field : value
        rf"\*\*{field_name}\*\*:\s*(.*?)(?:\n|$)",  # **Field**: value
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches.group(1).strip()
    
    return ""

def extract_list_items(text: str) -> List[str]:
    """Extract numbered or bulleted list items from text"""
    # Look for various list formats
    list_patterns = [
        r"^\s*(\d+\.|\*|\-|\•)\s*(.*?)(?:\n|$)",  # 1. item or * item or - item or • item
    ]
    
    items = []
    for pattern in list_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            item = match.group(2).strip()
            if item:
                items.append(item)
    
    # If no structured list found, try splitting by newlines or sentences
    if not items:
        # Try splitting by lines that look like list items without markers
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('#', 'Steps', 'Mitigation', 'Here')):
                items.append(line)
    
    return items

def assess_vulnerability_step_by_step(technology: str) -> Dict[str, Any]:
    """
    Uses a step-by-step approach to extract structured information about a vulnerability
    
    Args:
        technology: The technology to assess
        
    Returns:
        Dict containing vulnerability assessment data
    """
    llm = setup_ollama_model()
    results = {}
    
    # Step 1: Get vulnerability name
    name_prompt = PromptTemplate.from_template(
        "As a cybersecurity expert, identify a common vulnerability name associated with {technology}. "
        "Respond with ONLY the vulnerability name, nothing else."
    )
    
    try:
        vulnerability_name = llm.invoke(name_prompt.format(technology=technology)).strip()
        results["vulnerability_name"] = vulnerability_name
    except Exception as e:
        print(f"Error getting vulnerability name: {e}")
        results["vulnerability_name"] = f"Unknown vulnerability in {technology}"
    
    # Step 2: Get severity
    severity_prompt = PromptTemplate.from_template(
        "For the vulnerability '{vulnerability_name}' in {technology}, "
        "what is its severity level? Choose only one: Low, Medium, High, or Critical. "
        "Respond with ONLY the severity level, nothing else."
    )
    
    try:
        severity = llm.invoke(severity_prompt.format(
            vulnerability_name=results["vulnerability_name"], 
            technology=technology
        )).strip()
        
        # Normalize severity
        severity = severity.lower()
        if "critical" in severity:
            results["severity"] = "Critical"
        elif "high" in severity:
            results["severity"] = "High"
        elif "medium" in severity or "moderate" in severity:
            results["severity"] = "Medium"
        else:
            results["severity"] = "Low"
    except Exception as e:
        print(f"Error getting severity: {e}")
        results["severity"] = "Unknown"
    
    # Step 3: Get description
    description_prompt = PromptTemplate.from_template(
        "Provide a brief description (2-3 sentences) of the '{vulnerability_name}' vulnerability "
        "that affects {technology}. Be concise and specific."
    )
    
    try:
        description = llm.invoke(description_prompt.format(
            vulnerability_name=results["vulnerability_name"], 
            technology=technology
        )).strip()
        results["description"] = description
    except Exception as e:
        print(f"Error getting description: {e}")
        results["description"] = "No description available."
    
    # Step 4: Get mitigation steps
    mitigation_prompt = PromptTemplate.from_template(
        "List 3-5 specific steps to mitigate the '{vulnerability_name}' vulnerability "
        "in {technology}. Format as a numbered list with each step on a new line:\n"
        "1. First step\n"
        "2. Second step\n"
        "And so on."
    )
    
    try:
        mitigation_text = llm.invoke(mitigation_prompt.format(
            vulnerability_name=results["vulnerability_name"], 
            technology=technology
        )).strip()
        
        # Extract list items
        mitigation_steps = extract_list_items(mitigation_text)
        
        # If extraction failed, try splitting by newlines
        if not mitigation_steps:
            mitigation_steps = [step.strip() for step in mitigation_text.split('\n') 
                              if step.strip() and not step.lower().startswith(('here', 'steps', 'mitigation'))]
        
        # If still no steps, create a default step
        if not mitigation_steps:
            mitigation_steps = ["Consult a cybersecurity professional for specific mitigation steps."]
            
        results["mitigation_steps"] = mitigation_steps
    except Exception as e:
        print(f"Error getting mitigation steps: {e}")
        results["mitigation_steps"] = ["Consult a cybersecurity professional for mitigation steps."]
    
    return results

if __name__ == "__main__":
    technology = "Sharing confidential national security information via group chat in the secure messaging application called signal."
    print(f"Performing vulnerability assessment for: {technology}")
    print("This may take a moment...")
    
    assessment = assess_vulnerability_step_by_step(technology)
    
    if assessment:
        print("\nVulnerability Assessment:")
        print(f"Name: {assessment.get('vulnerability_name', 'N/A')}")
        print(f"Severity: {assessment.get('severity', 'N/A')}")
        print(f"Description: {assessment.get('description', 'N/A')}")
        print("\nMitigation Steps:")
        for i, step in enumerate(assessment.get('mitigation_steps', []), 1):
            print(f"{i}. {step}")
    else:
        print("Failed to generate structured assessment.")
