# 🧠 Vision-Language Model for Autoregressive Generation of Radiological Reports

**Bachelor's Thesis in Computer Engineering (ETSIIT - University of Granada)**  
**Author:** Rafael Carrillo Arroyo
---

## 📋 Overview
This thesis details the design, development, and empirical evaluation of an advanced multimodal pipeline that integrates computer vision and natural language processing (NLP) for the automated drafting of radiological reports. The system addresses and mitigates the inherent challenges of **normality bias** (linguistic prior) and **domain misalignment** through a modular architecture divided into three phases of strict engineering and continuous auditing[cite: 1].

---

## 🏗️ System Architecture: 3-Phase Pipeline

### PHASE 1: Clinical Visual Encoder (Spatial Feature Extraction)
**Objective:** Develop a robust visual base architecture capable of encoding radiological anatomy while preserving spatial topology, generating a foundational vocabulary for the generative model.

**Architectural and Optimization Strategy:**
* **Medical Backbone:** Adaptation of a **DenseNet-121** convolutional network through *Weight Surgery* (transfer of pre-trained weights and adaptation to monochromatic input).
* **Topological Preservation:** Replacement of the conventional *Global Average Pooling* layer to extract a **10x10 spatial grid** (100 continuous visual tokens of 1024 channels), isolating specific anatomical regions.
* **Imbalance Mitigation:** Concurrent multi-label learning (14 pathologies) to combat extreme imbalance (*Long-Tail*) by applying a weighted asymmetric loss function (BCE + Pos-Weight) and clinical uncertainty mapping (*Soft Targets*).

**Global Performance and Robustness (Test Cohort):**
The model demonstrates high global statistical separation capacity, maintaining structural integrity against the massive prevalence of healthy classes, as certified by the precision-recall area under the curve (PR AUC) analysis.

<div align="center">
  <img src="assets/Resultados_Evaluacion/Modulo_1/test/curvas_roc_test_finales.png" width="48%" alt="Joint Multi-label ROC Curve">
  <img src="assets/Resultados_Evaluacion/Modulo_1/test/curvas_pr_test_finales.png" width="48%" alt="Joint Multi-label PR Curve">
</div>

**Net Clinical Performance (F1-Score):**
The analysis on the test set confirms clinical-grade performance, exhibiting its highest diagnostic reliability in solid parenchymal pathologies and effusions. 

> ⚠️ **Methodological Note on Extreme Imbalance (Long-Tail):** The *Fracture* and *Lung Lesion* classes record an F1-Score affected by the natural distribution of the dataset. In *Fracture*, there is a net absence of positive cases in this test partition (null statistical support), disabling the calculation of the integral precision metric (PR AUC = N/A). In *Lung Lesion*, although there is residual clinical support, its prevalence is minimal, resulting in reduced sensitivity compared to the healthy majority class.

<div align="center">
  <img src="assets/Resultados_Evaluacion/Modulo_1/test/f1_scores_test.png" width="75%" alt="F1-Score Bar Chart">
</div>

📄 **[View full clinical metrics and thresholds table (CSV)](assets/Resultados_Evaluacion/Modulo_1/test/metricas_clinicas_test.csv)**

> **Clinical Explainability and Visual Auditing (Grad-CAM):**
> 
> To empirically certify that the convolutional network grounds its diagnosis on the correct anatomical coordinates and evades *Shortcut Learning*, a paired counterfactual comparison is used:
> * **Left Sample (True Positive, TP):** Shows the precise focus of thermal gradients on the real lesion or anatomical alteration (e.g., enhancement on the cardiac contour for cardiomegaly or at the lung bases for effusions).
> * **Right Sample (True Negative, TN):** Illustrates the behavior in a healthy patient. The network explores risk regions but deactivates thermal maps upon verifying topological normality, validating prediction robustness.
> 
> ![Showcase Grad-CAM](assets/Resultados_Evaluacion/Modulo_1/test/GradCAM/gradcam_showcase_readme.png)
>
> 🔍 **Exhaustive Auditing (14 Pathologies):** To guarantee the model's interpretability and safety, a complete visual validation was generated, evaluating the network's topological attention focus against all clinical labels in the dataset.
> * 📄 **[View Complete Grad-CAM Matrix (14 Pathologies)](assets/Resultados_Evaluacion/Modulo_1/test/GradCAM/gradcam_matriz_completa.png)**

---

### PHASE 2: Multimodal Bridge and Geometric Alignment
**Objective:** Establish the projection nexus between the visual extractor and the language decoder (BioGPT). This phase operates under a "Pure Alignment" protocol: the LLM is kept strictly frozen, shifting the responsibility of projecting visual semantics into the syntactic space onto a linear projector assisted by 2D Positional *Embeddings*.

**Architectural Validation and Stability (Test Cohort N=380):**
The multimodal bridge design successfully stabilizes the visual injection without causing saturation in the receiving network.
* **LLM Immutability:** Null maximum absolute difference (`Max-Diff = 0.00`), ensuring the preservation of BioGPT's original weights.
* **Energetic Stability:** Average L2 norm anchored at **32.40**, preventing mathematical saturation of the Softmax layer.
* **"Cone Effect" Mitigation:** Reduction of centered cosine similarity to **-0.0004**, ensuring optimal isotropic dispersion and mitigating the inherent anisotropy of language models.

> **Geometric Demonstration of Spatial Coherence:**
> <div align="center">
>   <img src="assets/Resultados_Evaluacion/Modulo_2/post_auditoria_topologica_2d.png" width="60%" alt="Topological Affinity Matrix pos_embeddings">
> </div>
> 
> *The affinity matrix empirically shows how the model manages to reconstruct the two-dimensional (2D) topology from a flat (1D) sequence. Since the radiograph is processed as a 10x10 grid, the appearance of stripes parallel to the main diagonal with an exact "jump" of 10 positions mathematically demonstrates that the network has discovered the vertical axis. Thus, the projector internalizes the anatomical structure without the need for explicit spatial supervision.*

**Analysis of Linguistic Inertia and "Clinical Blindness":**
Despite correct geometric alignment, clinical evaluation yields a critical **Macro Pathological Capture Rate of 9.41%**. The immutable conservation of BioGPT causes its strong linguistic bias (normality *prior*) to eclipse the visual signal, resulting in the generation of generic normality reports even in the presence of severe lesions evidence.

> **⚠️ Example of Normality Bias Hallucination (Patient with active pathologies):**
> * **[ORIGINAL REPORT]:** *"Findings: Right jugular catheter present... Scar / subsegmental atelectasis in the lingula..."*
> * **[GENERATED TEXT]:** *"Normal, the heart is unremarkable. Impression is clear, there is no pneumothorax..."*

**Methodological Verdict:** Unimodal pure projection alignment is insufficient to break the language model's inertia. This empirical finding validates the starting hypothesis and justifies the transition to **PHASE 3**, introducing domain adaptation into the LLM.

---

### PHASE 3: Domain Adaptation and Conditioned Generation (LoRA)
**Objective:** Eradicate the normality bias detected in the previous phase by injecting low-rank adapters (**LoRA**) into the self-attention layers of BioGPT-Large. A Supervised Fine-Tuning (SFT) process is applied over 15 optimization epochs, consolidating a focal loss minimum at epoch 15 ($\text{Val Focal Loss} = 1.2436$).

**Visual Plasticity and Semantic Coherence Auditing (Test Cohort N=380):**
The visual ablation protocol (*Blind Masking*) corroborates the success of the algorithmic intervention, evidencing the transition from an inertially linguistic model to one dependent on visual stimuli.

* **Visual Dependency Index (Macro VDI): 0.8145.** This threshold confirms that the system bases over 81% of its semantic inference on the input radiograph.
* **Contextual Semantic Quality:** A BERTScore F1 of **0.8780** and an SBERT similarity of **0.6389** (Absolute Scale) are obtained, ensuring fluid syntax consistent with human clinical terminology.

📄 **[View Natural Language Metrics Table (BLEU, ROUGE, BERTScore)](assets/Resultados_Evaluacion/Modulo_3/metricas_generacion_nlg.csv)**

**Net Diagnostic Performance (Zero-Shot NLP Auditing with BART):**
To validate the clinical fidelity of the generated reports, an automated extraction of pathologies was performed using an external model (BART-Large-MNLI) in *Zero-Shot* configuration, comparing the entities drafted by the system against the *Ground Truth*.

Global evaluation consolidates the multi-label diagnosis of the generated text:
* **Micro F1-Score (57.62% - Stable Convergence):** Net performance adjusted to population volume, demonstrating the model's overall reliability in a clinical setting.
* **Macro F1-Score (48.73% - Academic VLM Standard):** Average performance per class, reflecting the challenge of extreme imbalance in the cohort.

At the level of specific entities, the model exhibits a high capacity to document complex morphological alterations directly in natural language:

<div align="center">
  <img src="assets/Resultados_Evaluacion/Modulo_3/f3_distribucion_casos_nucleus.png" width="75%" alt="Clinical F1-Score of Generated Texts">
</div>

*Outstanding results are observed in highly relevant clinical pathologies such as **Lung Opacity (F1: 0.7593)**, **Consolidation (F1: 0.7161)**, and **Atelectasis (F1: 0.7043)**.*

**Epidemiological Auditing (Observed vs. Generated Distribution):**
This audit visualizes the total volume of diagnoses for each pathological category. The balanced contrast between both bars demonstrates the definitive overcoming of the inference collapse suffered in Phase 2. The model now closely approximates the real-world multi-label distribution, confirming that it drafts proportionally to clinical evidence and not due to statistical bias.

📄 **[View detailed cross-clinical validation report (CSV)](assets/Resultados_Evaluacion/Modulo_3/metricas_clinicas_bart.csv)**

---

## 🛠️ Tech Stack and Reproducibility

The *pipeline* was developed and iterated in GPU-accelerated environments (NVIDIA CUDA), optimizing VRAM usage through Automatic Mixed Precision (AMP) and gradient accumulation.

* **Deep Learning Framework:** PyTorch & PyTorch Lightning.
* **Clinical Preprocessing and Vision:** MONAI (Medical Open Network for AI) for medical-grade transformations, and TorchVision (DenseNet-121).
* **Language Processing (NLP):** HuggingFace Transformers (BioGPT, BART, SBERT).
* **Metrics and Evaluation:** Scikit-Learn, Evaluate, BERTScore.
* **Data Manipulation:** Pandas, NumPy, OpenCV.

> **💡 Methodological Documentation and Source Code:**
> The complete source code and experimentation notebooks (`notebooks/`) are temporarily kept private in this repository, pending the formal publication process of the project's results. They will be released publicly at the earliest opportunity.

---
* 📉 **Training Auditing (Convergence):** Stability graphs of the loss functions are available for **[Phase 1 (CNN)](assets/Resultados_Evaluacion/Modulo_1/curvas_entrenamiento_final.png)**, **[Phase 2 (Multimodal Bridge)](assets/Resultados_Evaluacion/Modulo_2/grafica_convergencia_fase2.png)**, and **[Phase 3 (LoRA)](assets/Resultados_Evaluacion/Modulo_3/convergencia_fase3_lora.jpg)**.

---
## 📂 Repository Resources and Structure

```text
TFG-Vision-Language-Clinical-Reports/
├── assets/          # Comprehensive repository of graphic resources, metrics, and audits
│   ├── Resultados_EDA/              # Exhaustive Exploratory Data Analysis (EDA)
│   │   ├── Modulo_1/                # Demographics, photometry, QC, and pathology correlation
│   │   └── Modulo_2_3_Indiana/      # Clinical bimodal inspection, corpus completeness, and Bigrams
│   └── Resultados_Evaluacion/       # Empirical tests and algorithmic validation of the 3 phases
│       ├── Modulo_1/                # [Phase 1] ROC/PR curves, confusion matrices, and Grad-CAM
│       ├── Modulo_2/                # [Phase 2] Bridge convergence and 2D topology demonstration
│       └── Modulo_3/                # [Phase 3] NLP Metrics (BERTScore) and evaluation CSVs
│
├── checkpoints/     # (Git Ignored) Local model weights
├── data/            # (Git Ignored) Dynamic dataset mounts
├── outputs/         # (Git Ignored) Local execution logs
└── README.md        # Architectural documentation and project index