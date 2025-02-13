# !/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from cache_utils import generate_cache_key, get_cached_response
import psutil
import gc


def initialize_model():
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Different loading strategy based on device
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            # CPU-specific loading without quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

def check_memory():
    """Check if system has enough memory available."""
    memory = psutil.virtual_memory()
    if memory.available < 500 * 1024 * 1024:  # Less than 500MB available
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:
            raise MemoryError("Insufficient memory available")

def is_42ad_related(prompt: str) -> bool:
    """Check if the question is related to 42 Abu Dhabi."""
    keywords = {
        'general': ['42', 'abu dhabi', '42 ad', '42ad', 'coding school', 'programming school'],
        'admission': ['piscine', 'pool', 'admission', 'apply', 'application', 'register', 'registration', 'check-in'],
        'curriculum': ['core', 'projects', 'peer learning', 'peer-to-peer', 'blackhole', 'evaluation', 'correction'],
        'campus': ['mina zayed', 'campus', 'facility', 'facilities', 'lab', 'labs', 'cluster', 'clusters'],
        'program': ['common core', 'specialization', 'project', 'exam', 'holy graph', 'levels', 'peer evaluation'],
        'logistics': ['housing', 'accommodation', 'transportation', 'visa', 'stipend', 'scholarship'],
        'events': ['hackathon', 'workshop', 'event', 'meetup', 'coalition'],
        'partners': ['hub71', 'mubadala', '42 network', 'mina zayed']
    }
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for category in keywords.values() for keyword in category)

def generate_response(prompt, model, tokenizer, device, max_length=512):
    # Validate if question is related to 42 Abu Dhabi
    if not is_42ad_related(prompt):
        return "I can only answer questions related to 42 Abu Dhabi coding school. Please ask me about admissions, curriculum, campus life, or other 42 Abu Dhabi related topics."

    # Check cache first
    cache_key = generate_cache_key(prompt)
    cached_response = get_cached_response(cache_key)
    if cached_response:
        return cached_response

    # Comprehensive system prompt focused on 42 Abu Dhabi
    system_prompt = """I am a specialized guide for 42 Abu Dhabi coding school, located in Mina Zayed, Abu Dhabi, UAE. 
    I only provide information about 42 Abu Dhabi and will not answer questions unrelated to the school.

    Key Information:
    - 42 Abu Dhabi is a coding school that follows a peer-to-peer learning methodology
    - No traditional teachers, no lectures, and no tuition fees
    - The curriculum includes a 4-week Piscine (intensive coding bootcamp) for selection
    - Main program consists of a Common Core and Specializations
    - Located in Mina Zayed with state-of-the-art facilities
    - Partners include Hub71, Mubadala, and the 42 Network
    - Offers project-based learning with peer evaluations
    - Students must be 18+ years old to apply
    - Provides opportunities for UAE residents and international students
    - Features include 24/7 campus access, gaming area, and collaboration spaces
    
    I will only respond to questions about:
    1. Admission process and Piscine
    2. Curriculum and projects
    3. Campus facilities and location
    4. Student life and community
    5. Prerequisites and requirements
    6. Application deadlines and processes
    7. School policies and procedures
    8. Available resources and support
    
    I will not answer questions unrelated to 42 Abu Dhabi."""
    
    # Combine system prompt with user's question
    full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nAnswer:"
    
    try:
        check_memory()
        
        # Optimize tokenizer settings
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length // 2,
            add_special_tokens=True
        ).to(device)
        
        # Optimized generation parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length // 2,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        
        # Cache the response
        get_cached_response.cache_info = {cache_key: answer}
        
        return answer
        
    except MemoryError:
        return "I apologize, but I'm currently experiencing memory constraints. Please try asking a shorter question."
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

def main():
    print("Initializing chatbot... (this may take a few moments)")
    try:
        model, tokenizer, device = initialize_model()
        print(f"Chatbot is ready! Running on {device}. Type 'quit' to exit.")
        
        while True:
            try:
                user_input = input("\nYour question: ").strip()
                if user_input.lower() == 'quit':
                    break
                if not user_input:
                    continue
                    
                response = generate_response(user_input, model, tokenizer, device)
                print("\nChatbot:", response)
            except KeyboardInterrupt:
                print("\nExiting chatbot...")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Please try asking your question again.")
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")

if __name__ == "__main__":
    main()