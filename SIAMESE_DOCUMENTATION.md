# Siamese Neural Networks — Complete Documentation

> Personal context: You built a Siamese Network for offline signature verification as your thesis.
> This document covers everything from basics to research-level depth.

---

## Table of Contents

1. [What is a Siamese Neural Network?](#1-what-is-a-siamese-neural-network)
2. [History — Where Did It Come From?](#2-history--where-did-it-come-from)
3. [Architecture — How It Is Built](#3-architecture--how-it-is-built)
4. [How It Learns — Contrastive Loss](#4-how-it-learns--contrastive-loss)
5. [Signature Verification System](#5-signature-verification-system)
6. [Evaluation Metrics for Biometrics](#6-evaluation-metrics-for-biometrics)
7. [Real-World Applications](#7-real-world-applications)
8. [Has It Been Replaced?](#8-has-it-been-replaced)
9. [Drawbacks and Limitations](#9-drawbacks-and-limitations)
10. [Comparison Table](#10-comparison-table)

---

## 1. What is a Siamese Neural Network?

A **Siamese Neural Network** is a special architecture that contains **two identical sub-networks** (called "twins") that **share the same weights**.

Instead of asking "what class does this input belong to?", it asks a fundamentally different question:

> **"Are these two inputs similar or different?"**

### The Core Idea

```
Input A ──► [Neural Network] ──► Embedding A ──┐
             (shared weights)                   ├──► Distance ──► Same? / Different?
Input B ──► [Neural Network] ──► Embedding B ──┘
```

The network maps each input into an **embedding space** (a compressed vector), and then measures the **distance** between the two embeddings:
- **Small distance** → inputs are from the same class
- **Large distance** → inputs are from different classes

### Simple Analogy

Imagine you have two photos of signatures. A human expert looks at both and says:
- "These loops match, the pressure pattern is similar, the slant is the same → same person"

A Siamese network learns to do exactly this comparison automatically.

---

## 2. History — Where Did It Come From?

### 1993 — The Original Paper

Siamese Networks were **invented specifically for signature verification**.

**Paper:** *"Signature Verification using a Siamese Time Delay Neural Network"*
**Authors:** Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, Roopak Shah
**Published:** NeurIPS 1993
**Institution:** Bell Laboratories (AT&T)

This is the **exact same problem** you solved in your thesis — offline signature verification.

The name "Siamese" comes from Siamese twins (conjoined twins who share biology) — the two networks share weights the way twins share DNA.

### Timeline of Evolution

```
1993  ─── Bromley et al.
           Original Siamese Network for signature verification
           Used 1D convolutions on pen movement sequences

2005  ─── Chopra, Hadsell & LeCun
           "Learning a Similarity Metric Discriminatively, with
            Application to Face Verification" (CVPR 2005)
           Introduced Contrastive Loss formally
           Applied to face verification

2015  ─── Koch, Zemel & Salakhutdinov
           "Siamese Neural Networks for One-Shot Image Recognition" (ICML 2015)
           Applied to Omniglot dataset
           Popularised one-shot learning

2015  ─── Schroff, Kalenichenko & Philbin (Google)
           "FaceNet: A Unified Embedding for Face Recognition" (CVPR 2015)
           Introduced Triplet Loss as an improvement over contrastive loss
           Achieved state-of-the-art face recognition

2017+ ─── Transformer-based approaches begin emerging

2020+ ─── Self-supervised contrastive learning (SimCLR, MoCo, CLIP)
           Siamese principles adapted at massive scale
```

### Key Insight from History

Siamese Networks were not just an ML technique — they redefined how we think about **similarity learning** (also called metric learning). The idea that you can learn a distance function from data, rather than engineering one manually, was revolutionary.

---

## 3. Architecture — How It Is Built

### Building Blocks

```
┌─────────────────────────────────────────────────────────┐
│                  SIAMESE NETWORK                         │
│                                                          │
│  Signature A ──► ┌─────────────┐ ──► Embedding A (64D) │
│                  │ BASE NETWORK│                         │
│  Signature B ──► │ (SHARED     │ ──► Embedding B (64D) │
│                  │  WEIGHTS)   │                         │
│                  └─────────────┘                         │
│                                          │               │
│                              Euclidean Distance          │
│                                          │               │
│                              ┌───────────┴────────┐      │
│                              │ Same person?       │      │
│                              │ D < threshold → YES│      │
│                              │ D > threshold → NO │      │
│                              └────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### The Base Network (our implementation)

```
Input (128 features)
     ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
     ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.2)
     ↓
Dense(64)  → ReLU
     ↓
L2 Normalize  ← critical: projects onto unit sphere
     ↓
Embedding (64D unit vector)
```

### Why L2 Normalization?

Without normalization, embeddings can have different magnitudes. Euclidean distance would then measure both direction AND magnitude differences, making it harder to learn.

After L2 normalization, all embeddings lie on a **unit hypersphere**. Distance only measures angular separation — much cleaner for similarity learning.

```
Before L2 norm:   ||v|| = any value (could be 0.1 or 1000)
After  L2 norm:   ||v|| = 1.0 always

Distance formula on sphere ≈ angular separation
```

### Weight Sharing — Why It Matters

The two branches MUST share weights. Here is why:

**Without shared weights:**
```
Branch A: learns Feature A → random projection
Branch B: learns Feature B → different random projection
Result: comparing apples and oranges — meaningless distance
```

**With shared weights:**
```
Branch A: maps Signature A → point in embedding space E
Branch B: maps Signature B → point in SAME embedding space E
Result: distance in E is meaningful — same transformation was applied
```

**Code:**
```python
base_network = build_base_network()   # one model instance

# Apply THE SAME model to both inputs
embedding_a = base_network(input_a)   # same weights
embedding_b = base_network(input_b)   # same weights ← this is weight sharing
```

---

## 4. How It Learns — Contrastive Loss

### What is Contrastive Loss?

Introduced by **Hadsell, Chopra & LeCun (2006)**. It directly optimises the distance between pairs.

```
L = y × D²  +  (1−y) × max(margin − D, 0)²

Where:
  D      = Euclidean distance between two embeddings
  y      = 1 (same person / genuine) or 0 (different / forged)
  margin = minimum distance forced between different-class embeddings
```

### How Each Case Works

**Case 1: Genuine pair (y=1)**
```
L = 1 × D² + 0
L = D²

Effect: minimise D → pull the two embeddings CLOSER
If D=0, Loss=0  (perfect)
```

**Case 2: Forged pair (y=0)**
```
L = 0 + (max(margin - D, 0))²

Effect:
  If D < margin: Loss > 0 → push embeddings APART
  If D ≥ margin: Loss = 0 (already far enough apart)
```

### Visual Intuition

```
BEFORE TRAINING (random embeddings):
● Person 0 sig 1
              ● Person 0 sig 2     ← too far apart
  ● Person 1 sig 1    ← too close to Person 0

AFTER TRAINING:
  ●● Person 0 sigs    ← clustered together
                 ●● Person 1 sigs  ← clustered separately
```

### Contrastive Loss vs Triplet Loss

| | Contrastive Loss | Triplet Loss |
|--|-----------------|-------------|
| **Inputs** | Pairs (A, B) | Triplets (Anchor, Positive, Negative) |
| **Formula** | y×D(A,B)² + (1-y)×max(m-D,0)² | max(D(A,P) - D(A,N) + margin, 0) |
| **Training data** | Pairs with label | Triplets (harder to sample) |
| **Convergence** | Easier | Faster if triplets are hard |
| **Used in** | Our project, original Siamese | FaceNet, modern face recognition |

**Triplet Loss (used in FaceNet):**
```
For each Anchor signature:
  Positive = another genuine signature (same person)
  Negative = any forged signature (different person)

L = max(D(Anchor,Positive) - D(Anchor,Negative) + margin, 0)
```

---

## 5. Signature Verification System

### How Your Thesis System Works

```
ENROLLMENT (one-time setup per user):
  User provides 5-10 genuine signatures
  System computes average embedding (template)
  Template stored in database

VERIFICATION (real-time):
  User claims to be Person X
  Provides one signature
  System: compare new signature embedding vs stored template
  If distance < threshold → ACCEPT (genuine)
  If distance > threshold → REJECT (forged)
```

### Signature Feature Extraction (real system)

In a real offline signature verification system, before the Siamese network, you extract features from the image:

```python
# Features extracted from signature image (128 dimensions):
features = [
    # Global features (12)
    aspect_ratio, ink_coverage, normalized_area,
    baseline_slope, pen_tilt_angle, ...

    # Structural features (32)
    num_strokes, stroke_lengths, curvature_histogram,
    loop_count, endpoints_count, ...

    # Statistical features (32)
    pixel_density_grid (4×8 grid = 32 values), ...

    # Texture features (52)
    LBP_histogram, Gabor_filter_responses, HOG_features, ...
]
# Total: 128 feature dimensions
```

### Offline vs Online Signature Verification

| | Offline | Online |
|--|---------|--------|
| **Input** | Scanned image | Time-series (x,y, pressure, velocity) |
| **When captured** | After signing | During signing |
| **Features** | Shape, texture | Dynamics (speed, pressure, direction) |
| **Difficulty** | Harder | Easier (more info available) |
| **Your thesis** | Yes | No |

---

## 6. Evaluation Metrics for Biometrics

Biometric systems use different metrics than standard classification. This section is critical for your thesis understanding.

### FAR and FRR

**FAR — False Acceptance Rate** (Type II error in biometrics)
```
FAR = FP / (FP + TN)
    = (forgeries wrongly accepted) / (total forgeries tested)
```
High FAR → security risk (attackers get in).

**FRR — False Rejection Rate** (Type I error in biometrics)
```
FRR = FN / (FN + TP)
    = (genuine signatures wrongly rejected) / (total genuine tested)
```
High FRR → usability problem (real users get locked out).

### The FAR/FRR Tradeoff

FAR and FRR move in opposite directions as you change the threshold:

```
Threshold ↑ (accept more):   FAR ↑,  FRR ↓
Threshold ↓ (accept fewer):  FAR ↓,  FRR ↑
```

For a bank: keep FAR very low (security priority)
For a hospital login: balance FAR and FRR (usability priority)

### Equal Error Rate (EER)

EER is where FAR = FRR. Used to compare systems with a single number.
**Lower EER = Better system.**

```
State-of-the-art offline signature verification:
  Traditional methods (SVM, DTW): EER ~5-15%
  Early deep learning:            EER ~3-8%
  Modern Siamese nets:            EER ~1-5%
  (CEDAR dataset, SigNet paper)
```

### ROC Curve for Biometrics

The DET (Detection Error Tradeoff) curve is often used instead of ROC in biometrics:
- X-axis: FAR
- Y-axis: FRR (instead of TAR)
- Ideal point: origin (0% FAR, 0% FRR)

---

## 7. Real-World Applications

### Signature Verification (your thesis)
- Bank cheque processing (automated signature verification)
- Legal document authentication
- Point-of-sale terminal verification

### Face Verification
- FaceID on iPhone (not face recognition — face *verification*)
- Access control systems
- Border control (passport vs face)

### One-Shot Learning
- Learning from very few examples (1-5 shots)
- Critical when labelled data is scarce
- Medical imaging: rare disease classification from few examples

### Document Similarity
- Plagiarism detection (Are these two documents similar?)
- Legal document matching
- Patent similarity search

### Drug Discovery
- Molecular similarity (Are these two compounds structurally similar?)
- Protein structure comparison

### Visual Search / Recommendation
- "Find similar products" on e-commerce
- Pinterest visual search
- Reverse image search

### Question Duplicate Detection
- StackOverflow duplicate question detection
- Quora: "Are these two questions asking the same thing?"

---

## 8. Has It Been Replaced?

### Short Answer: Not replaced — but evolved.

The core Siamese principle (shared-weight twin networks + metric learning) is **alive and central** to modern AI. But specific implementations have advanced significantly.

### Evolution Timeline

```
Siamese Networks (1993)
         ↓
    Contrastive Loss (2005) — formalised training
         ↓
    Triplet Loss / FaceNet (2015) — better sampling strategy
         ↓
    ArcFace / CosFace (2018) — angular margin losses for faces
         ↓
    SimCLR / MoCo (2020) — self-supervised contrastive learning
         ↓
    CLIP (2021, OpenAI) — contrastive learning at billion scale
         ↓
    SimSiam / BYOL (2021) — no negatives needed
```

### Modern Alternatives for Specific Tasks

#### For Signature Verification:
| Method | EER | Notes |
|--------|-----|-------|
| Siamese + Contrastive Loss | ~3-8% | Your thesis |
| Siamese + Triplet Loss | ~2-5% | Better negative mining |
| SigNet (SigNet paper, 2016) | ~1.6% | Improved CNN architecture |
| Transformer-based | ~1-3% | Recent, expensive |

#### For Face Verification:
| Method | Accuracy (LFW) | Notes |
|--------|----------------|-------|
| DeepFace (Facebook, 2014) | 97.35% | CNN + Siamese |
| FaceNet (Google, 2015) | 99.63% | Triplet loss |
| ArcFace (2018) | 99.82% | Angular margin, current SOTA |
| Transformer-based | ~99.8% | Similar to ArcFace |

### The Bigger Picture: Contrastive Learning is Everywhere

The core Siamese principle — **learn by comparing pairs** — is one of the most important ideas in modern ML:

**CLIP (OpenAI, 2021):**
```
Image encoder ──► Image embedding ──┐
                  (shared learning)  ├──► Contrastive loss
Text  encoder ──► Text  embedding ──┘

Trained on 400 million (image, caption) pairs from the internet.
Powers DALL-E, GPT-4V visual understanding.
```

**SimCLR (Google, 2020):**
```
Same image ──► Augmentation A ──► [Encoder] ──► Embedding A ──┐
              Augmentation B ──► [Encoder] ──► Embedding B ──┴──► Contrastive
Same principle as Siamese, but self-supervised (no labels needed).
```

**Conclusion:** Siamese Networks were not replaced. They became the foundation of self-supervised and contrastive learning — which powers modern large language models and vision models.

---

## 9. Drawbacks and Limitations

### 1. Pair/Triplet Sampling Problem
Creating meaningful training pairs is difficult and critical.

```
Problem: If most pairs are "easy negatives" (very different people),
         the network learns trivially and stops improving.

Solution: Hard negative mining — deliberately find similar-but-different pairs
          (e.g., two people with similar handwriting styles)

Code example:
  # Hard negative: find the forged signature most similar to a genuine one
  distances = model.predict([genuine_embeddings, all_other_embeddings])
  hard_negatives = all_others[distances < threshold + small_margin]
```

### 2. Quadratic Growth of Pairs
With N signatures, possible pairs = N²/2. For large datasets, this becomes unmanageable.

```
100 signatures   →    4,950 pairs   (OK)
10,000 signatures →  49,995,000 pairs  (problematic)
1,000,000 signatures → 5×10¹¹ pairs  (impossible to enumerate)
```

Solution: Smart sampling, online triplet mining during training.

### 3. Not Directly Scalable to Many Classes
Siamese networks verify (1:1 comparison), not identify (1:N search).

```
Verification:  Is this Person 0? → compare vs stored template  O(1)
Identification: Who is this person? → compare vs ALL stored   O(N)

For N=1,000,000 users: 1 million comparisons per query.
Modern approach: ANN (Approximate Nearest Neighbour) search with FAISS.
```

### 4. Threshold Sensitivity
The decision threshold must be tuned carefully and may not generalise:

```
Trained threshold on Dataset A: works well
Same threshold on Dataset B (different paper, pen, lighting): may fail

Real-world deployment requires:
  - Per-user threshold calibration
  - Domain adaptation
  - Periodic retraining
```

### 5. Limited to Feature-Level Comparison
The base network compresses inputs to a fixed-size vector. Fine-grained spatial relationships may be lost, especially for high-resolution signatures.

Modern fix: Attention mechanisms let the network focus on discriminative regions.

### 6. Imbalanced Real-World Data
In production:
- Genuine attempts >> Forged attempts (most users are legitimate)
- Creates class imbalance during training
- Requires careful oversampling or loss weighting

### 7. Adversarial Vulnerability
Skilled forgers can craft signatures specifically designed to fool the model. Standard Siamese networks are not robust to adversarial attacks without specific defences.

---

## 10. Comparison Table

| Feature | Siamese Network | Standard CNN | SVM + Features | Transformer |
|---------|----------------|-------------|----------------|-------------|
| **Task type** | Similarity / verification | Classification | Classification | Any |
| **Training data** | Pairs of examples | Individual examples | Individual examples | Large corpus |
| **New user** | Just enrol (no retrain) | Requires retraining | Requires retraining | Fine-tune |
| **One-shot learning** | Yes | No | No | Few-shot |
| **Interpretability** | Low (distance only) | Low | Medium (SVM weights) | Very low |
| **Training time** | Moderate | Moderate | Fast | Slow |
| **Inference time** | Fast (2× base network) | Fast | Very fast | Slow |
| **Works with few samples** | Yes | No | No | No |

---

## Project Files Summary

```
ML-AI/
├── siamese_signature_verification.py  ← main training script
├── siamese_model.h5                   ← full Siamese model (generated)
├── base_network.h5                    ← base encoder network (generated)
├── siamese_results.png                ← 6-panel visualization (generated)
├── SIAMESE_DOCUMENTATION.md           ← this file
└── SIAMESE_INTERVIEW_QA.md            ← interview preparation
```

---

## Key Papers to Read

1. Bromley et al. (1993) — *Signature Verification using a Siamese Time Delay Neural Network*
2. Chopra, Hadsell & LeCun (2005) — *Learning a Similarity Metric Discriminatively*
3. Koch et al. (2015) — *Siamese Neural Networks for One-Shot Image Recognition*
4. Schroff et al. (2015) — *FaceNet: A Unified Embedding for Face Recognition and Clustering*
5. Dey et al. (2017) — *SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification*
