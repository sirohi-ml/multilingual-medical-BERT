# Tokenization Fundamentals

This folder contains implementations of fundamental NLP tokenization methods from scratch. The objective is to understand how raw text is transformed into tokens and eventually numerical representations before building the tokenizer for the multilingual BERT model.

## Dataset

Experiments use the **WikiText-2 Raw** dataset (`Salesforce/wikitext`, `wikitext-2-raw-v1`) from Hugging Face.

## Topics Covered

### 1. Whitespace Tokenization
Basic tokenization using Python's `split()` method.

### 2. Regex-Based Word Tokenization
Separating words and punctuation using regular expressions and constructing a simple word-level vocabulary.

### 3. Character-Level Tokenization
Building character frequencies and representing text as individual characters.

### 4. Byte Pair Encoding (BPE)
Implementing the core BPE procedure from scratch:

- Initialize words as sequences of characters
- Count adjacent token-pair frequencies
- Identify the most frequent pair
- Merge the selected pair
- Repeat the process to progressively construct subword tokens

### 5. WordPiece
Exploring the WordPiece token-selection strategy and comparing it with BPE.

Instead of selecting pairs purely by frequency, WordPiece scores candidate pairs using:

`score(a, b) = freq(a, b) / (freq(a) × freq(b))`

The highest-scoring pair is selected for merging.

## Purpose

These implementations are intentionally simple and educational rather than production tokenizers. They are designed to build an understanding of:

`Raw Text → Tokens → Vocabulary → Token IDs → Subword Tokens`

This forms the foundation for the next stage of the project: training a shared multilingual subword tokenizer for the Multilingual Medical BERT model.
