"""
Example demonstrating how to switch between different models in Ollama
"""
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import warnings
warnings.filterwarnings("ignore")

class OllamaModelManager:
    """Helper class to manage different Ollama models"""
    
    def __init__(self):
        # Dictionary to store model configurations
        self.model_configs = {
            "gemma3:12b": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512
            },
            "llama3-chatqa:8b": {  # Add any other models you have installed
                "temperature": 0.8,
                "top_p": 0.9,
                "num_predict": 512
            },
            "deepseek-r1:8b": {  # Add any other models you have installed
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        
        # Dictionary to store initialized models
        self.models = {}
        
    def list_available_models(self):
        """List all configured models"""
        return list(self.model_configs.keys())
    
    def get_model(self, model_name):
        """
        Get an initialized model by name
        
        Args:
            model_name (str): Name of the model to initialize
            
        Returns:
            Ollama: Initialized Ollama model
        """
        # If model is not initialized yet, initialize it
        if model_name not in self.models:
            if model_name not in self.model_configs:
                raise ValueError(f"Model '{model_name}' not configured")
                
            print(f"Initializing model: {model_name}")
            config = self.model_configs[model_name]
            self.models[model_name] = Ollama(
                model=model_name,
                **config
            )
            
        return self.models[model_name]
    
    def compare_models(self, prompt, model_names=None):
        """
        Compare responses from multiple models
        
        Args:
            prompt (str): Prompt to send to all models
            model_names (list, optional): List of models to compare. Defaults to all models.
            
        Returns:
            dict: Dictionary mapping model names to their responses
        """
        if model_names is None:
            model_names = self.list_available_models()
            
        responses = {}
        for model_name in model_names:
            try:
                print(f"Getting response from {model_name}...")
                start_time = time.time()
                model = self.get_model(model_name)
                response = model.invoke(prompt)
                end_time = time.time()
                elapsed = end_time - start_time
                
                responses[model_name] = {
                    "response": response,
                    "elapsed_time": elapsed
                }
                
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                responses[model_name] = {
                    "response": f"ERROR: {str(e)}",
                    "elapsed_time": 0
                }
                
        return responses

def run_comparison():
    """Run a comparison between different Ollama models"""
    manager = OllamaModelManager()
    available_models = manager.list_available_models()
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    # For each available model, check if it exists
    existing_models = []
    for model_name in available_models:
        try:
            # Create a simple test prompt to check if model exists
            test_prompt = f"Hello, testing {model_name}."
            model = Ollama(model=model_name)
            # Try to invoke with a timeout
            _ = model.invoke(test_prompt, stop_sequences=[""], timeout=5)
            existing_models.append(model_name)
            print(f"✓ {model_name} is available")
        except Exception as e:
            print(f"✗ {model_name} is not available: {e}")
    
    if not existing_models:
        print("No configured models are available. Please check your Ollama installation.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Compare models with a custom prompt")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ")
        
        if choice == "2":
            print("Exiting...")
            break
            
        elif choice == "1":
            prompt = input("\nEnter a prompt to send to all models: ")
            
            responses = manager.compare_models(prompt, existing_models)
            
            print("\n=== Results ===\n")
            for model, result in responses.items():
                print(f"=== {model} (Time: {result['elapsed_time']:.2f}s) ===")
                print(result["response"])
                print("=" * 50)

if __name__ == "__main__":
    run_comparison()
