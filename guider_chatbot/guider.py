# !/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def initialize_model():
    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, tokenizer, device

def generate_response(prompt, model, tokenizer, device, max_length=500):
    # Create a system prompt that defines the bot's purpose
    system_prompt = """You are a helpful assistant that answers questions about the 42 coding schools network, 
    particularly 42 Abu Dhabi. You provide accurate information about the curriculum, admission process, 
    and campus life."""
    
    # Combine system prompt with user's question
    full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nAnswer:"
    
    # Tokenize input with padding and attention mask
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    return response.split("Answer:")[-1].strip()

def main():
    print("Initializing chatbot... (this may take a few moments)")
    model, tokenizer, device = initialize_model()
    print(f"Chatbot is ready! Running on {device}. Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nYour question: ")
            if user_input.lower() == 'quit':
                break
                
            response = generate_response(user_input, model, tokenizer, device)
            print("\nChatbot:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try asking your question again.")

if __name__ == "__main__":
    main()