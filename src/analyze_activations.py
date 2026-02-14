import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def load_data(activation_dir):
    labels = np.load(os.path.join(activation_dir, "labels.npy"))
    # Get number of layers from files
    files = [f for f in os.listdir(activation_dir) if f.startswith("layer_") and f.endswith("_last.npy")]
    num_layers = len(files)
    return labels, num_layers

def analyze_layer(layer_idx, activation_dir, labels, act_type="last"):
    acts = np.load(os.path.join(activation_dir, f"layer_{layer_idx}_{act_type}.npy"))
    
    # Difference in means
    ai_mean = acts[labels == 1].mean(axis=0)
    human_mean = acts[labels == 0].mean(axis=0)
    diff_direction = ai_mean - human_mean
    
    # Normalize direction
    diff_direction = diff_direction / np.linalg.norm(diff_direction)
    
    # Train probe
    X_train, X_test, y_train, y_test = train_test_split(acts, labels, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return acc, f1, diff_direction

def main():
    activation_dir = "results/activations"
    labels, num_layers = load_data(activation_dir)
    
    results_last = []
    results_mean = []
    directions_last = []
    directions_mean = []
    
    print(f"Analyzing {num_layers} layers...")
    for l in range(num_layers):
        acc_l, f1_l, dir_l = analyze_layer(l, activation_dir, labels, "last")
        acc_m, f1_m, dir_m = analyze_layer(l, activation_dir, labels, "mean")
        
        results_last.append({'layer': l, 'accuracy': acc_l, 'f1': f1_l})
        results_mean.append({'layer': l, 'accuracy': acc_m, 'f1': f1_m})
        directions_last.append(dir_l)
        directions_mean.append(dir_m)
        
    df_last = pd.DataFrame(results_last)
    df_mean = pd.DataFrame(results_mean)
    
    # Find best layer
    best_layer_last = df_last.loc[df_last['accuracy'].idxmax()]
    best_layer_mean = df_mean.loc[df_mean['accuracy'].idxmax()]
    
    print(f"Best layer (last token): {best_layer_last['layer']} with accuracy {best_layer_last['accuracy']:.4f}")
    print(f"Best layer (mean pool): {best_layer_mean['layer']} with accuracy {best_layer_mean['accuracy']:.4f}")
    
    # Save results
    df_last.to_csv("results/probe_results_last.csv", index=False)
    df_mean.to_csv("results/probe_results_mean.csv", index=False)
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(df_last['layer'], df_last['accuracy'], label='Last Token')
    plt.plot(df_mean['layer'], df_mean['accuracy'], label='Mean Pool')
    plt.xlabel('Layer')
    plt.ylabel('Probe Accuracy')
    plt.title('AI vs Human Probe Accuracy by Layer')
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/probe_accuracy.png")
    
    # Cosine similarity between layers for the direction
    cos_sims = []
    for l in range(num_layers - 1):
        sim = np.dot(directions_mean[l], directions_mean[l+1])
        cos_sims.append(sim)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_layers - 1), cos_sims)
    plt.xlabel('Layer pair (l, l+1)')
    plt.ylabel('Cosine Similarity')
    plt.title('Direction Cosine Similarity between Consecutive Layers')
    plt.grid(True)
    plt.savefig("figures/cosine_similarity.png")
    
    # Save the AI direction from a middle layer (e.g., layer 14) for steering
    target_layer = 14
    np.save("results/ai_direction.npy", directions_mean[target_layer])
    print(f"Saved AI direction from layer {target_layer}")

if __name__ == "__main__":
    main()
