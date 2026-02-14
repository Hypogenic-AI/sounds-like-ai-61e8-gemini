import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_activations(model, tokenizer, text, device, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states is a tuple of (num_layers + 1)
    # each element is (batch, seq_len, hidden_size)
    # We want the residual stream (hidden states)
    # Let's take the last token of each layer
    hidden_states = outputs.hidden_states
    
    # We'll return a dictionary of layer_idx -> activation
    # Layer 0 is the embedding layer, Layer L is the final layer
    last_token_acts = {l: hidden_states[l][0, -1, :].cpu().numpy() for l in range(len(hidden_states))}
    mean_acts = {l: hidden_states[l][0, :, :].mean(dim=0).cpu().numpy() for l in range(len(hidden_states))}
    
    return last_token_acts, mean_acts

def main():
    set_seed(42)
    model_name = "Qwen/Qwen2.5-1.5B"
    data_path = "datasets/raid_sample.json"
    output_dir = "results/activations"
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    data = load_data(data_path)
    
    results = []
    
    print("Extracting activations...")
    for i, entry in enumerate(tqdm(data)):
        text = entry['generation']
        label = 1 if entry['model'] == 'llama-chat' else 0 # 1 for AI, 0 for Human
        
        last_acts, mean_acts = get_activations(model, tokenizer, text, device)
        
        # We'll save these in a list and then convert to a big array
        results.append({
            'id': entry['id'],
            'label': label,
            'last_acts': last_acts,
            'mean_acts': mean_acts,
            'domain': entry['domain']
        })
        
        # Periodic saving or just at the end
    
    # Process results into arrays per layer
    num_layers = len(results[0]['last_acts'])
    for l in range(num_layers):
        layer_last_acts = np.stack([r['last_acts'][l] for r in results])
        layer_mean_acts = np.stack([r['mean_acts'][l] for r in results])
        labels = np.array([r['label'] for r in results])
        domains = np.array([r['domain'] for r in results])
        
        np.save(os.path.join(output_dir, f"layer_{l}_last.npy"), layer_last_acts)
        np.save(os.path.join(output_dir, f"layer_{l}_mean.npy"), layer_mean_acts)
    
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump([{'id': r['id'], 'label': r['label'], 'domain': r['domain']} for r in results], f)

    print(f"Activations saved to {output_dir}")

if __name__ == "__main__":
    main()
