# Resources Catalog: "Sounds Like AI" Direction Research

## Summary
This catalog lists the papers, datasets, and code repositories gathered to investigate the existence of a linear "sounds like AI" direction in the residual stream of LLMs.

## Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| A Single Direction of Truth | O'Neill et al. | 2025 | [link](papers/oneill_2025_single_direction_truth.pdf) | Linear direction for hallucinations. |
| Refusal in Language Models | Arditi et al. | 2024 | [link](papers/arditi_2024_refusal_single_direction.pdf) | Single direction for refusal behavior. |
| The Geometry of Truth | Marks & Tegmark | 2023 | [link](papers/marks_2023_geometry_truth.pdf) | Linear structure of truth. |
| The Linear Representation Hypothesis | Park et al. | 2023 | [link](papers/park_2023_linear_representation_hypothesis.pdf) | Theoretical framework for LRH. |
| Mechanistic Understanding of Alignment | Lee et al. | 2024 | [link](papers/lee_2024_mechanistic_understanding_alignment.pdf) | Study on DPO and toxicity mechanisms. |
| Inference-Time Intervention | Li et al. | 2023 | [link](papers/li_2023_inference_time_intervention.pdf) | Steering for truthfulness. |

## Datasets
Total datasets downloaded: 1 (Sampled)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| RAID Sample | HuggingFace (liamdugan/raid) | 200 samples | AI Detection | datasets/raid_sample.json | 100 human, 100 AI samples. |

### Dataset Download Instructions
The RAID dataset is available on HuggingFace. To load the full dataset:
```python
from datasets import load_dataset
dataset = load_dataset("liamdugan/raid")
```

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| refusal_direction | github.com/andyrdt/refusal_direction | Finding/steering directions. | code/refusal_direction | Very relevant implementation. |
| representation_engineering | github.com/andyzoujm/representation-engineering | General RepE framework. | code/representation_engineering | SOTA methodology. |
| raid | github.com/liamdugan/raid | AI detection benchmark. | code/raid | Evaluation and data handling. |

## Recommendations for Experiment Design

1.  **Primary Dataset**: **RAID**. It provides a diverse set of human vs. AI pairs across multiple domains (news, wiki, abstracts) and models (GPT-4, Claude, etc.).
2.  **Baseline Methods**: 
    - **Difference-in-means** between AI and Human activations in the middle layers (layers 10-20 for 7B models).
    - **Logistic Regression Probe** on the same activations.
3.  **Evaluation Metrics**:
    - **F1 Score** for the probe's ability to distinguish AI from Human text.
    - **Cosine Similarity** between directions identified across different domains/models to test universality.
    - **Qualitative Steering**: Subtracting the "AI direction" from an AI-generated response to see if it becomes less formulaic/repetitive.
4.  **Code to adapt**:
    - `code/refusal_direction/refusal_direction.py` for finding the direction.
    - `code/representation_engineering/` for advanced RepE techniques.
