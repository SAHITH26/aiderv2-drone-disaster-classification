# Aerial Disaster Scene Classification on AIDERv2

Multi-class classification of aerial disaster scenes (**Earthquake / Fire / Flood / Normal**) on the AIDERv2 dataset, comparing a **custom CNN trained from scratch** against **MobileNetV2 transfer learning** under an identical training protocol.

MobileNetV2 reaches **88.3% test accuracy** versus **82.0%** for the custom CNN, a ~6-point advantage. Both models share the same dominant failure mode: the `Normal` class.

---

## Motivation

After a disaster, aerial and UAV imagery arrives faster than responders can review it. Automatically triaging frames into *fire*, *flood*, *earthquake damage*, or *nothing happening* narrows the search space for human operators. This project asks a practical question for that setting: **is a small purpose-built CNN competitive with a pre-trained backbone when compute and data are limited?**

---

## Dataset

**AIDERv2** — Aerial Image Dataset for Emergency Response, accessed via a Kaggle mirror.

| | |
|---|---|
| Classes | Earthquake, Fire, Flood, Normal |
| Test set | 1,654 images |
| Input resolution | 224 × 224 × 3 |
| Class balance | Imbalanced — Earthquake is the minority class |

Class imbalance is handled with **balanced (inverse-frequency) class weights**: approximately 0.82 (Flood), 0.86 (Normal), 0.95 (Fire), 1.74 (Earthquake).

> **A note on framing.** The imagery is aerial and drone-like, but AIDERv2 is not exclusively UAV footage — it aggregates aerial and elevated-viewpoint disaster imagery from multiple sources. The repository name reflects the project's UAV motivation, not a claim that every image is drone-captured.

---

## Repository structure

```
.
├── notebook.ipynb      # Full pipeline: data → training → evaluation → Grad-CAM
├── report.pdf          # Final report (elsarticle format)
├── proposal.docx       # Project proposal
├── figures/            # All figures as PDF (curves, confusion matrices, Grad-CAM)
└── README.md
```

---

## Method

### Models

**Custom CNN** — stacked convolution/pooling blocks with batch normalisation, global pooling, and a dropout-regularised dense head. Trained from random initialisation.

**MobileNetV2 (frozen)** — ImageNet-pretrained backbone with all convolutional layers frozen; only a lightweight classification head is trained on the extracted features.

### Training configuration

Both models were trained under an identical protocol so that the comparison isolates the effect of the backbone.

| Setting | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Max epochs | 12 |
| Optimizer | Adam |
| Learning rate | 1e-3 (custom CNN) · 1e-4 (MobileNetV2 head) |
| LR schedule | ReduceLROnPlateau — factor 0.2, patience 2, min 1e-6 |
| Early stopping | Val loss, patience 4, restore best weights |
| Checkpointing | Best validation loss |
| Loss | Categorical cross-entropy |
| Class weighting | Balanced (inverse frequency) |
| Dropout | 0.5 (custom CNN) · 0.4 (MobileNetV2 head) |
| Random seed | 42 (Python, NumPy, TensorFlow) |

Hyperparameters were chosen from common practice and adjusted lightly against validation behaviour. No automated search was performed.

---

## Results

### Overall (held-out test set, 1,654 images)

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Custom CNN | 0.820 | 0.813 | 0.835 | 0.812 | 0.820 |
| **MobileNetV2 (frozen)** | **0.883** | **0.887** | **0.887** | **0.881** | **0.879** |

### Per-class recall

| Class | Custom CNN | MobileNetV2 |
|---|---|---|
| Earthquake | 0.93 | 0.91 |
| Fire | 0.90 | **0.97** |
| Flood | 0.86 | **0.96** |
| Normal | 0.64 | **0.70** |

---

## Key findings

**1. `Normal` is the hardest class for both models.** Recall drops to 0.64 (custom CNN) and 0.70 (MobileNetV2), while `Normal` *precision* stays high (0.87 and 0.94). The models leak non-disaster scenes into disaster categories rather than the reverse — they over-trigger. In an emergency-response context this is the safer error direction, but it drives the false-alarm rate.

**2. The custom CNN's real weakness is Earthquake precision, not recall.** At 0.62 precision, it pulls 80 `Normal` and 44 `Flood` images into the Earthquake class. MobileNetV2 lifts this to 0.86. Aggregate accuracy alone hides this; the confusion matrix does not.

**3. Class weighting rescues the minority class.** Earthquake, the smallest class, achieves 0.93 and 0.91 recall — both models handle it well despite the imbalance.

**4. Training order matters.** An early run pinned validation accuracy at chance level because alphabetically-ordered directory listings produced single-class batches. Shuffling the full training dataframe before batching fixed it. Worth stating explicitly: a silent data-ordering bug is indistinguishable from a broken model.

---

## Explainability

Grad-CAM heatmaps are generated for representative correct and incorrect predictions from both models. They confirm the models attend to semantically relevant regions — flame and smoke regions for Fire, inundated surfaces for Flood, rubble texture for Earthquake — rather than exploiting background artefacts.

All Grad-CAM figures are in `figures/`.

---

## Reproducing

The full pipeline lives in `notebook.ipynb` and runs end-to-end on a **Kaggle Notebook with GPU** (TensorFlow 2.19).

1. Open the notebook in Kaggle.
2. Attach the AIDERv2 dataset.
3. Enable GPU, then run all cells.

Approximate runtime: both models train within the 12-epoch cap under early stopping.

---

## Limitations

- **Single run per model.** No repeated runs with variance estimates; reported differences are not accompanied by confidence intervals.
- **No hyperparameter search.** Values were selected manually within a GPU budget.
- **Frozen backbone only.** Fine-tuning the upper MobileNetV2 blocks was not attempted and would likely improve results further.
- **Single dataset.** No cross-dataset evaluation, so generalisation beyond AIDERv2's collection conditions is untested.

---

## Author

**Sahith Ganta**
MSc student, University of Europe for Applied Sciences
Machine Learning course project — supervised by **Prof. Raja Hashim Ali**

See `report.pdf` for the full methodology, figures, and bibliography.
