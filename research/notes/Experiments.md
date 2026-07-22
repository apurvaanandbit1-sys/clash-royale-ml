# Hyperparameter Search & Ablations

## 1. Hyperparameter Search Results

We swept hyperparameter coordinates to evaluate sensitivity:

*   **Embedding Dimension (D)**:
    *   16-D: Accuracy = **56.75%**, ROC-AUC = **0.6011**
    *   32-D: Accuracy = **56.89%**, ROC-AUC = **0.6050**
    *   64-D: Accuracy = **57.36%**, ROC-AUC = **0.6089**
*   **Pooling Type**:
    *   Mean: Accuracy = **56.75%**, ROC-AUC = **0.6011**
    *   Sum: Accuracy = **57.01%**, ROC-AUC = **0.6051**
*   **Dropout**:
    *   0.0: Accuracy = **57.44%**, ROC-AUC = **0.6079**
    *   0.1: Accuracy = **56.75%**, ROC-AUC = **0.6011**
    *   0.2: Accuracy = **56.76%**, ROC-AUC = **0.6004**

---

## 2. Ablation Study Results

To justify the architecture, we evaluated ablated configurations:

| Configuration | Test Accuracy | Test ROC-AUC | Test Loss | ECE |
| :--- | :---: | :---: | :---: | :---: |
| **Full Model** | **56.75%** | **0.6011** | **0.6762** | **4.04%** |
| **Without SkillDifference** | 56.57% | 0.5973 | 0.6774 | 4.25% |
| **Without Bradley-Terry** | 56.69% | 0.5953 | 0.6796 | 4.41% |
| **Without Deep Sets** | 56.56% | 0.6012 | 0.6762 | 4.05% |
| **Without Trophy Difference** | 56.57% | 0.5973 | 0.6774 | 4.25% |
