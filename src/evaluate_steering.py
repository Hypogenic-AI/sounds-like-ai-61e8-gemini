import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_probe(layer_idx, activation_dir, labels):
    acts = np.load(os.path.join(activation_dir, f"layer_{layer_idx}_mean.npy"))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(acts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, labels)
    return clf, scaler

def get_mean_activation(model, tokenizer, text, device, layer_idx):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    mean_act = outputs.hidden_states[layer_idx][0, :, :].mean(dim=0).cpu().numpy()
    return mean_act

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    activation_dir = "results/activations"
    layer_idx = 14
    steering_results_path = "results/steering_outputs.json"
    
    with open(steering_results_path, 'r') as f:
        steering_data = json.load(f)
        
    labels = np.load(os.path.join(activation_dir, "labels.npy"))
    
    print("Loading model and training probe for evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    clf, scaler = load_probe(layer_idx, activation_dir, labels)
    
    evaluation_results = []
    for entry in steering_data:
        text = entry['text']
        coeff = entry['coeff']
        prompt = entry['prompt']
        
        act = get_mean_activation(model, tokenizer, text, device, layer_idx)
        act_scaled = scaler.transform(act.reshape(1, -1))
        
        prob_ai = clf.predict_proba(act_scaled)[0][1] # Probability of being AI
        
        print(f"Prompt: {prompt[:30]}... Coeff: {coeff} -> AI Prob: {prob_ai:.4f}")
        
        evaluation_results.append({
            'prompt': prompt,
            'coeff': coeff,
            'prob_ai': prob_ai,
            'text': text
        })
        
    with open("results/steering_evaluation.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2)

if __name__ == "__main__":
    main()
