# Evaluation Metrics Guide

## Why layered evaluation

A single metric hides problems. ROUGE-L says "high overlap" while the model outputs Hindi-flavored text; BLEU says "low" while the script is structurally perfect but lexically creative. This project therefore evaluates on four layers:

1. **Lexical overlap** (ROUGE, BLEU, chrF) — regression detection
2. **Structure** (script format score) — did the model learn the format
3. **Fluency** (perplexity, human review) — is the Urdu natural
4. **Safety/contamination** (stress tests) — is the Urdu actually Urdu

## Metric details

### ROUGE-L
Longest common subsequence between prediction and reference. Use the F-measure. For drama scripts, expect low absolute values (creative variance) — track the *delta* between baseline and fine-tuned.

### BLEU (sacrebleu)
Corpus-level BLEU with Urdu tokenization. sacrebleu handles Unicode correctly; never use naive whitespace tokenization for Urdu.

### chrF
Character n-gram F-score. Preferred over BLEU for Urdu because:
- Urdu morphology (inflections, izafat, compound verbs) makes word-level matching brittle
- Character-level matching captures partial word overlap

### Script Structure Score
Per-output binary checks averaged:
- scene heading present
- speaker-labeled dialogue present
- action lines present

Defined in `src/evaluate.py:script_structure_score`. Deterministic, no model required.

### Perplexity (optional)
On held-out reference text. Requires the base tokenizer; interpretable only as a relative comparison between checkpoints. Not included in the default eval report because it requires different batching; add if you need it.

## Human evaluation protocol

For a portfolio-grade repo, run a small human eval:

1. Sample 20 outputs from the test set (fixed seed)
2. Rate each 1-5 on: naturalness, register correctness, character consistency, format correctness
3. Have at least one native Urdu speaker review
4. Record the rubric and results in `docs/BENCHMARKS.md`

## LLM-as-judge (optional layer)

Use a strong external model to judge held-out outputs on naturalness and Hindi-isms. Keep the judge prompt versioned in this repo so results are reproducible. Never use the same model family as the fine-tuned model for judging (correlated errors).

## Common failure modes this evaluation catches

| Symptom | Caught by |
|---|---|
| Model never stops generating | avg prediction length at max |
| Hindi-leaning vocabulary | stress test blocklist + human review |
| Script format ignored | script structure score |
| Base model regurgitating training episodes | split leakage check |
| Same episode in train and eval | split leakage check |
| Degenerate repetition | repetition penalty + avg length + human review |
| BOM/mojibake in data | stress test contamination check |
