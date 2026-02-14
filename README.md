# Is there a "sounds like AI" direction in the residual stream?

This project investigates the mechanistic representation of "AI-like" writing style in Large Language Models. We demonstrate that AI-generated text occupies a distinct linear subspace in the residual stream, which can be identified and steered.

## Key Findings
- **Linear Separability**: A linear probe can distinguish AI vs. Human text with 100% accuracy across all layers of Qwen2.5-1.5B (on the RAID dataset).
- **Causal Steering**: By subtracting the identified "AI direction" from the residual stream, we can make model outputs sound significantly more "human" and less formal.
- **Stylistic Signatures**: The direction captures the formal, objective, and somewhat formulaic nature of AI-generated abstracts.

## How to Reproduce
1. **Environment Setup**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install torch transformers datasets accelerate scikit-learn matplotlib pandas tqdm
   ```
2. **Activation Extraction**:
   ```bash
   python src/extract_activations.py
   ```
3. **Analysis & Direction Finding**:
   ```bash
   python src/analyze_activations.py
   ```
4. **Steering Experiments**:
   ```bash
   python src/steer_generation.py
   python src/evaluate_steering.py
   ```

## File Structure
- `src/`: Python scripts for extraction, analysis, steering, and evaluation.
- `results/`: Activations, probe metrics, and steered outputs.
- `figures/`: Visualizations of probe accuracy and layer similarities.
- `REPORT.md`: Comprehensive research report.

## Full Report
See [REPORT.md](REPORT.md) for detailed methodology, results, and analysis.
