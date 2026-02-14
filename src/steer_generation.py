import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def steer_hook(module, input, output, direction, coefficient):
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    device = hidden_states.device
    dir_tensor = torch.from_numpy(direction).to(device).to(hidden_states.dtype)
    steer_vec = coefficient * dir_tensor
    
    new_hidden_states = hidden_states + steer_vec
    
    if isinstance(output, tuple):
        return (new_hidden_states,) + output[1:]
    else:
        return new_hidden_states

def generate_steered(model, tokenizer, prompt, layer_idx, direction, coefficient, max_new_tokens=100):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Register hook
    # Qwen layers are in model.model.layers
    target_layer = model.model.layers[layer_idx]
    
    hook_handle = target_layer.register_forward_hook(
        lambda m, i, o: steer_hook(m, i, o, direction, coefficient)
    )
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
    finally:
        # Always remove hook
        hook_handle.remove()
        
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    direction_path = "results/ai_direction.npy"
    layer_idx = 14
    
    if not os.path.exists(direction_path):
        print("Direction not found. Run analysis first.")
        return
        
    direction = np.load(direction_path)
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    prompts = [
        "Write a short abstract about the future of AI in medical imaging.",
        "The recent advancements in artificial intelligence combined with the extensive amount of data"
    ]
    
    coefficients = [-5.0, 0.0, 5.0] # Subtract, Neutral, Add
    
    results = []
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for coeff in coefficients:
            print(f"Steering coeff: {coeff}")
            text = generate_steered(model, tokenizer, prompt, layer_idx, direction, coeff)
            print(f"Output: {text[:200]}...")
            results.append({
                'prompt': prompt,
                'coeff': coeff,
                'text': text
            })
            
    # Save results
    with open("results/steering_outputs.json", 'w') as f:
        import json
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
