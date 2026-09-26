# Stage 1 — Base Multilingual BERT 🌍

> **Goal:** Build and pretrain a multilingual BERT model from scratch — from raw multilingual text and tokenization to Transformer architecture and Masked Language Modeling.

This stage forms the linguistic foundation of the **Multilingual Medical BERT** project.

The objective is not simply to train a model using an existing BERT implementation. We want to understand and implement the core components ourselves: how text becomes tokens, how tokens become contextual representations, how self-attention allows information to move across a sequence, and how Masked Language Modeling teaches a Transformer to represent language.

The model developed here will later undergo **medical-domain** and **radiology-domain** adaptive pretraining.

---

## 🧭 Where Stage 1 Fits

```text
                     MULTILINGUAL MEDICAL BERT
                              │
                              ▼
              ┌──────────────────────────────┐
              │           STAGE 1            │
              │   Base Multilingual BERT     │
              │                              │
              │  General multilingual text  │
              │             ↓                │
              │     Shared Tokenizer         │
              │             ↓                │
              │      BERT from Scratch       │
              │             ↓                │
              │   Masked Language Modeling   │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │           STAGE 2            │
              │  Medical Domain Adaptation   │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │           STAGE 3            │
              │ Radiology Domain Adaptation  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │           STAGE 4            │
              │  Downstream Medical Tasks    │
              └──────────────────────────────┘
```

Stage 1 deliberately contains **no medical specialization**.

The objective is first to teach the model the statistical and semantic structure of human language across multiple languages.

---

# 🌐 1. Multilingual Corpus

Our starting point is raw text.

```text
English ──────┐
Hindi ────────┤
Spanish ──────┤
French ───────┤
German ───────┤
Arabic ───────┤
Chinese ──────┤
Japanese ─────┤
...           │
              ▼
      ┌─────────────────┐
      │ Multilingual    │
      │ Text Corpus     │
      └─────────────────┘
```

Potential sources include:

- Wikipedia
- books and open text collections
- multilingual web corpora
- other appropriately licensed public datasets

The corpus should contain sufficiently diverse linguistic structures so that the model does not learn language from a single linguistic family or writing system.

### Data pipeline

```text
Raw Sources
     │
     ▼
Download / Stream
     │
     ▼
Language Filtering
     │
     ▼
Text Extraction
     │
     ▼
Cleaning
     │
     ▼
Deduplication
     │
     ▼
Language Balancing / Sampling
     │
     ▼
Training Corpus
```

Large raw datasets themselves should **not be committed to GitHub**. The repository should instead contain the scripts and configurations required to reproduce the data pipeline.

---

# ✂️ 2. Tokenization

Neural networks cannot directly operate on text.

We therefore need a mapping:

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
```

For example:

```text
"The patient is walking"

              ↓ TOKENIZER

["The", "patient", "is", "walk", "##ing"]

              ↓ VOCABULARY

[128, 4921, 214, 1832, 617]
```

A shared tokenizer is particularly important for multilingual learning.

Instead of:

```text
English tokenizer
Hindi tokenizer
French tokenizer
Arabic tokenizer
...
```

we want:

```text
             ┌───────── English
             │
             ├───────── Hindi
             │
Raw Text ────┼───────── French
             │
             ├───────── Arabic
             │
             └───────── ...
                     │
                     ▼
             SHARED TOKENIZER
                     │
                     ▼
             SHARED VOCABULARY
```

This allows all languages to enter the **same representation space**.

## Tokenizer learning

As part of this project, we study and implement the mechanics behind subword tokenization rather than treating tokenization as a black box.

The learning path includes:

```text
Characters
    │
    ▼
Token Frequencies
    │
    ▼
Pair Frequencies
    │
    ▼
Pair Scores
    │
    ▼
Best Merge
    │
    ▼
Updated Vocabulary
    │
    └───────────────┐
                    │
                    ▼
                 Repeat
```

Concepts explored include:

- character-level vocabularies
- word frequencies
- token frequencies
- adjacent token-pair frequencies
- merge operations
- subword construction
- vocabulary size
- unknown tokens
- multilingual vocabulary allocation

---

# 🧠 3. BERT Architecture — From Scratch

Once text has been converted into token IDs, we build the Transformer encoder architecture.

The high-level flow is:

```text
Input Text
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
┌───────────────────────────┐
│     Input Embeddings      │
│                           │
│ Token Embedding           │
│        +                  │
│ Position Embedding        │
│        +                  │
│ Segment Embedding         │
└─────────────┬─────────────┘
              │
              ▼
      Transformer Encoder
              │
              ▼
      Transformer Encoder
              │
              ▼
             ...
              │
              ▼
      Transformer Encoder
              │
              ▼
   Contextual Representations
```

Each Transformer encoder block will be implemented and studied independently.

```text
                    INPUT
                      │
                      ▼
              ┌───────────────┐
              │ Multi-Head    │
              │ Self-Attention│
              └───────┬───────┘
                      │
                + Residual
                      │
                      ▼
                 LayerNorm
                      │
                      ▼
              ┌───────────────┐
              │ Feed-Forward  │
              │    Network    │
              └───────┬───────┘
                      │
                + Residual
                      │
                      ▼
                 LayerNorm
                      │
                      ▼
                    OUTPUT
```

---

# 🔎 4. Self-Attention

Self-attention is the central mechanism through which BERT constructs contextual representations.

Given hidden representations \(X\), we learn three projections:

\[
Q = XW_Q
\]

\[
K = XW_K
\]

\[
V = XW_V
\]

Attention is then computed as:

\[
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Conceptually:

```text
                 TOKEN REPRESENTATIONS
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
             WQ         WK         WV
              │          │          │
              ▼          ▼          ▼
              Q          K          V
              │          │          │
              └─────┬────┘          │
                    ▼               │
                   QKᵀ              │
                    │               │
                    ▼               │
                 / √dₖ              │
                    │               │
                    ▼               │
                 Softmax            │
                    │               │
                    └───────┬───────┘
                            ▼
                    Attention Output
```

We will build toward multi-head attention:

\[
\text{MultiHead}(Q,K,V)
=
\text{Concat}(head_1,\ldots,head_h)W_O
\]

allowing different attention heads to learn different relationships between tokens.

---

# 🧩 5. Masked Language Modeling

BERT is pretrained using **Masked Language Modeling (MLM)**.

Suppose the original sentence is:

```text
The patient has severe pain in the knee.
```

During training, we may transform it into:

```text
The patient has severe [MASK] in the knee.
```

The model receives the corrupted sentence:

```text
                    [MASK]
                       │
                       ▼
The ─ patient ─ has ─ severe ─ [MASK] ─ in ─ the ─ knee
                       │
                       ▼
              Transformer Encoder
                       │
                       ▼
                Contextual Vector
                       │
                       ▼
                 Vocabulary Head
                       │
                       ▼
             Probability Distribution

             pain        0.72
             swelling    0.08
             injury      0.05
             ...
```

The model learns by minimizing the cross-entropy loss between the predicted token distribution and the original masked token.

\[
\mathcal{L}_{MLM}
=
-\sum_{i \in M}
\log P(x_i \mid x_{\setminus M})
\]

where \(M\) represents the set of masked positions.

Over a sufficiently large multilingual corpus, this objective forces the model to learn increasingly useful contextual representations.

---

# 🌍 6. How Multilingual Representations Emerge

We do not create a separate BERT for every language.

Instead:

```text
English sentence ─────┐
Hindi sentence ───────┤
Spanish sentence ─────┤
Arabic sentence ──────┤
French sentence ──────┤
                      ▼
              Shared Tokenizer
                      │
                      ▼
             Shared Embeddings
                      │
                      ▼
              Shared Transformer
                      │
                      ▼
          Shared Representation Space
```

Every language updates the **same model parameters**.

As training progresses, representations can capture structural and semantic regularities that are useful across languages.

This shared multilingual foundation is what Stage 2 will later adapt to medicine.

---

# 🏗️ 7. Complete Stage 1 Pipeline

```text
                    RAW MULTILINGUAL TEXT
                            │
                            ▼
                ┌─────────────────────┐
                │ Data Collection     │
                │ Cleaning            │
                │ Deduplication       │
                │ Sampling            │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Shared Subword      │
                │ Tokenizer           │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Token IDs           │
                │ Attention Masks     │
                │ MLM Masks           │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Embedding Layer     │
                └──────────┬──────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │ Transformer Encoder × N   │
             │                           │
             │ Multi-Head Attention      │
             │ Residual Connections      │
             │ Layer Normalization       │
             │ Feed-Forward Networks     │
             └─────────────┬─────────────┘
                           │
                           ▼
                Contextual Representations
                           │
                           ▼
                ┌─────────────────────┐
                │ MLM Prediction Head │
                └──────────┬──────────┘
                           │
                           ▼
                    Cross-Entropy Loss
                           │
                           ▼
                    Backpropagation
                           │
                           ▼
                   Update Parameters
                           │
                           └──────────────┐
                                          │
                                          ▼
                                    Repeat Training
```

---

# 📚 8. Learning Philosophy

A central objective of this repository is **understanding**, not merely reproducing an existing implementation.

Whenever practical, important components will first be derived conceptually and mathematically before relying on high-level abstractions.

For example:

```text
Use existing BERT model
        ❌

Understand:
Tokenization
    ↓
Embeddings
    ↓
Q, K, V
    ↓
Scaled Dot-Product Attention
    ↓
Multi-Head Attention
    ↓
Feed-Forward Networks
    ↓
Residual Connections
    ↓
Layer Normalization
    ↓
Transformer Encoder
    ↓
Masked Language Modeling
        ✅
```

Libraries such as PyTorch can provide tensor operations, automatic differentiation, GPU execution, optimizers, and data utilities while the core model architecture remains implemented explicitly.

---

# 📁 Stage 1 Structure

The directory will evolve as the project progresses.

```text
stage_1_base_multilingual_model/
│
├── README.md
│
├── data/
│
├── tokenizer/
│
├── model/
│
├── pretraining/
│
├── evaluation/
│
└── notebooks/
```

Large datasets, checkpoints, and other generated artifacts should remain outside Git version control where appropriate.

---

# 🗺️ Stage 1 Roadmap

| Phase | Objective | Status |
|---|---|---|
| 1 | Understand tokenization fundamentals | 🚧 In Progress |
| 2 | Gather multilingual corpus | 🚧 In Progress |
| 3 | Clean and normalize multilingual text | ⬜ |
| 4 | Train shared multilingual tokenizer | ⬜ |
| 5 | Implement embedding layer | ⬜ |
| 6 | Implement self-attention | ⬜ |
| 7 | Implement multi-head attention | ⬜ |
| 8 | Implement feed-forward network | ⬜ |
| 9 | Implement residual connections & LayerNorm | ⬜ |
| 10 | Assemble Transformer encoder | ⬜ |
| 11 | Stack Transformer encoders into BERT | ⬜ |
| 12 | Implement MLM data pipeline | ⬜ |
| 13 | Implement MLM prediction head & loss | ⬜ |
| 14 | Pretrain multilingual BERT | ⬜ |
| 15 | Evaluate learned representations | ⬜ |
| 16 | Save Stage 1 pretrained checkpoint | ⬜ |

---

# 🎯 Stage 1 Output

At the end of Stage 1 we should have:

```text
        General Multilingual Corpus
                    │
                    ▼
          Multilingual Pretraining
                    │
                    ▼
       ┌────────────────────────┐
       │                        │
       │  BASE MULTILINGUAL     │
       │         BERT           │
       │                        │
       └────────────┬───────────┘
                    │
                    ▼
       understands representations
       across multiple languages
                    │
                    ▼
              Stage 2
                    │
                    ▼
          Medical Adaptation 🩺
```

The resulting checkpoint will **not yet be a medical model**.

It will instead provide the multilingual linguistic foundation on which the subsequent medical and radiology specialization stages can be built.

---

## 🔬 Long-Term Direction

The complete project ultimately aims to connect:

```text
             LANGUAGE
                │
                ▼
      Multilingual Understanding
                │
                ▼
       Medical Understanding
                │
                ▼
      Radiology Understanding
                │
                ▼
       Clinical Representations
                │
                ├───────────────┐
                ▼               ▼
         Report Analysis    Medical Imaging
                │               │
                └───────┬───────┘
                        ▼
               Vision-Language Grounding
                        │
                        ▼
              Abnormality Detection
                        │
                        ▼
          Clinically Useful Representations
```

Stage 1 is where that journey begins.

---

**Project:** `multilingual-medical-BERT`  
**Stage:** `01 — Base Multilingual Model`  
**Current focus:** Multilingual data collection and tokenizer fundamentals
