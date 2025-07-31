#!/usr/bin/env python3
"""
Inference script for fine-tuned Llama 3 7B model on EV Charging QA
"""

import torch
import yaml
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVChargingChatbot:
    """Chatbot for EV Charging questions using fine-tuned Llama 3."""
    
    def __init__(self, model_path, base_model="meta-llama/Meta-Llama-3-7B-Instruct"):
        """Initialize the chatbot with the fine-tuned model."""
        self.model_path = model_path
        self.base_model = base_model
        self.system_prompt = "You are an expert on electric vehicle charging technology and infrastructure. Provide accurate, detailed, and helpful information about EV charging."
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            model_path,
            torch_dtype=torch.bfloat16
        )
        
        logger.info("Model loaded successfully!")
    
    def format_chat_prompt(self, question: str) -> str:
        """Format the question using Llama 3 chat format."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        chat_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return chat_prompt
    
    def generate_response(self, question: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a response to the question."""
        # Format the prompt
        prompt = self.format_chat_prompt(question)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
        if assistant_start in response:
            response = response.split(assistant_start)[-1].strip()
        
        return response
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("🚗⚡ EV Charging Expert Chatbot")
        print("Ask me anything about electric vehicle charging!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! 👋")
                    break
                
                if not question:
                    continue
                
                print("Assistant: ", end="", flush=True)
                response = self.generate_response(question)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")

def test_model(model_path: str):
    """Test the model with sample questions."""
    # Sample EV charging questions
    test_questions = [
        "What are the different types of EV charging connectors?",
        "How long does it take to charge an electric vehicle?",
        "What factors should be considered when planning EV charging infrastructure?",
        "What is smart charging and how does it work?",
        "How do EV charging costs compare to gasoline?",
        "What is the difference between AC and DC charging?",
        "How does weather affect EV charging?",
        "What are the benefits of workplace charging?",
        "How does Vehicle-to-Grid (V2G) technology work?",
        "What are the main challenges in EV charging infrastructure deployment?"
    ]
    
    # Initialize chatbot
    chatbot = EVChargingChatbot(model_path)
    
    print("🧪 Testing EV Charging Expert Model")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        print("-" * 40)
        
        try:
            response = chatbot.generate_response(question, max_new_tokens=300)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EV Charging Chatbot Inference")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the fine-tuned LoRA model")
    parser.add_argument("--base_model", type=str, 
                      default="meta-llama/Meta-Llama-3-7B-Instruct",
                      help="Base model path")
    parser.add_argument("--test", action="store_true",
                      help="Run automated tests")
    parser.add_argument("--interactive", action="store_true",
                      help="Start interactive chat")
    
    args = parser.parse_args()
    
    if args.test:
        test_model(args.model_path)
    elif args.interactive:
        chatbot = EVChargingChatbot(args.model_path, args.base_model)
        chatbot.interactive_chat()
    else:
        print("Please specify --test or --interactive mode")

if __name__ == "__main__":
    main() 