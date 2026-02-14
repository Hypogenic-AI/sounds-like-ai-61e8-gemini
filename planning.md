# Planning: Is there a "sounds like AI" direction in the residual stream?

## Motivation & Novelty Assessment

### Why This Research Matters
As LLMs become ubiquitous, distinguishing between human-written and AI-generated text is increasingly critical for academic integrity, misinformation mitigation, and maintaining the "human-ness" of digital discourse. Understanding the internal representation of "AI-ness" allows us to:
1. Develop more robust, interpretability-based AI detectors.
2. Causal intervention: Adjust the "AI-ness" of model outputs to make them more human-like or explicitly AI-labeled.
3. Understand if "AI-style" is a fundamental semantic feature or a surface-level artifact.

### Gap in Existing Work
Prior work has identified linear directions for "Truth" (Marks & Tegmark, 2023), "Refusal" (Arditi et al., 2024), and "Honesty" (Zou et al., 2023). However, "sounding like an AI" is a more nebulous, stylistic property. While AI detection (RAID, 2024) focuses on external classification, we lack a mechanistic understanding of whether LLMs themselves represent their own output style as a distinct linear feature in the residual stream.

### Our Novel Contribution
We propose to identify a "sounds like AI" direction using Representation Engineering (RepE) techniques on the RAID dataset. We will test if this direction is universal across domains and if it can be used to steer models to change their stylistic "AI-ness" without losing semantic coherence.

### Experiment Justification
- **Experiment 1: Activation Collection & Probe Training**: To confirm that "AI-ness" is indeed linearly decodable from the residual stream.
- **Experiment 2: Direction Identification (Diff-in-Means & PCA)**: To find the actual vector representing this feature.
- **Experiment 3: Cross-Domain Generalization**: To test if the "AI direction" found in news text is the same as in creative writing, indicating a universal stylistic representation.
- **Experiment 4: Steering/Intervention**: To prove the causal relevance of the identified direction. Subtracting it should make AI text sound more human; adding it should "AI-ify" human text.

---

## Research Question
Does there exist a universal linear direction in the residual stream of LLMs that represents the "AI-generated" stylistic quality of a text?

## Hypothesis Decomposition
1. **Linearity**: The difference between AI-generated and human-written text can be captured by a single linear direction in the residual stream.
2. **Universality**: This direction is consistent across different domains (e.g., news vs. code vs. recipes).
3. **Causality**: Moving activations along this direction will change the perceived "AI-ness" of the model's output.

## Proposed Methodology

### Approach
We will use a 7B or 8B parameter model (e.g., Mistral-7B or Llama-3-8B). We will use the RAID dataset (Human vs. AI pairs) to collect activations. We will then use the "Difference-in-Means" method to identify the candidate direction and validate it via linear probing and steering.

### Experimental Steps
1. **Setup**: Initialize environment, download model, and load RAID dataset samples.
2. **Activation Extraction**: For a set of Human/AI text pairs, extract residual stream activations at various layers.
3. **Direction Identification**:
    - Compute the mean activation for "Human" and "AI" classes.
    - Calculate the difference vector: `V_ai = Mean(Act_ai) - Mean(Act_human)`.
    - Perform PCA on the difference of pairs to find the primary axis of variation.
4. **Validation (Probing)**: Train a linear probe on 80% of the data and test on 20% to measure separability.
5. **Cross-Domain Test**: Find directions in Domain A and test probes in Domain B.
6. **Steering Experiments**:
    - Implement a steering hook to add/subtract `V_ai` during generation.
    - Generate text with various steering strengths.
    - Evaluate steered text using:
        - External AI detectors (e.g., RoBERTa-based).
        - Perplexity (to ensure quality).
        - Qualitative inspection.

### Baselines
- **Random Direction**: To ensure the effect is not due to any random vector intervention.
- **Top PCA Component (unsupervised)**: To see if "AI-ness" is the naturally most dominant feature.

### Evaluation Metrics
- **Probe Accuracy/F1**: Success of the linear detection.
- **Cosine Similarity**: Between directions found in different domains.
- **Detector Score Delta**: Change in external AI detector probability after steering.
- **Perplexity Delta**: Measurement of degradation in language quality.

## Expected Outcomes
- We expect to find a strong linear direction, particularly in the middle-to-late layers of the model.
- Steering should successfully shift the scores of external AI detectors.
- We hypothesize that "AI-ness" is characterized by specific lexical patterns (overuse of certain transitions, neutral tone) that have a linear signature.

## Timeline and Milestones
- **Hour 1**: Environment setup, model download, data prep.
- **Hour 2-3**: Activation collection and direction identification.
- **Hour 4**: Probing and cross-domain analysis.
- **Hour 5-6**: Steering experiments and evaluation.
- **Hour 7**: Final analysis and REPORT.md.

## Success Criteria
- Probe accuracy > 85% on held-out data.
- Steering significantly changes AI detector scores (p < 0.05).
- Identified direction shows high cosine similarity (> 0.5) across at least 3 domains.
