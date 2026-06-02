# Beautiful Music

Neural network models for generating four-part Bach-style chorales. Two approaches to symbolic music generation using the JSB Chorales dataset (382 chorales at 16th-note resolution):

1. **Unconditioned Generation** — an LSTM language model that learns the distribution of four-voice chorale sequences and samples new compositions from scratch
2. **Soprano Harmonization** — a bidirectional LSTM that takes a soprano melody as input and generates musically coherent alto, tenor, and bass voices

## Results

| Task | Model | Test Perplexity | Notes |
|------|-------|----------------|-------|
| Unconditioned | Bigram Markov | 13.59 | Baseline — only models adjacent token pairs |
| Unconditioned | LSTM | 1.87 | Captures long-range harmonic and melodic structure |
| Harmonization | Frequency Lookup | 125.02 | Baseline — most common chord per soprano pitch |
| Harmonization | BiLSTM | 5.76 | Bidirectional context produces coherent voice leading |

The harmonizer achieves 41.1% exact pitch accuracy and 43.4% pitch-class accuracy across the three generated voices, with significantly fewer voice-leading violations (parallel fifths/octaves) than the frequency baseline.

## Listen

- [`symbolic_unconditioned.mid`](symbolic_unconditioned.mid) — a chorale generated entirely from the LSTM language model
- [`symbolic_conditioned.mid`](symbolic_conditioned.mid) — a Bach soprano melody harmonized by the BiLSTM

## Architecture

**LSTM Language Model** (unconditioned): Chorales are represented as interleaved SATB token sequences. A 2-layer LSTM (64-dim embeddings, 128 hidden units, ~242K parameters) models the sequence autoregressively with temperature-controlled sampling.

**BiLSTM Harmonizer** (conditioned): A 2-layer bidirectional LSTM reads the full soprano melody, then three parallel linear heads predict alto, tenor, and bass simultaneously at each timestep. Bidirectional context is key — harmonization depends on where the melody is going, not just where it's been.

Both models use data augmentation via transposition, cosine annealing LR scheduling, and gradient clipping.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook workbook.ipynb
```

The notebook runs end-to-end: downloads the dataset, trains both models, evaluates, and exports MIDI files. GPU recommended (CUDA or MPS) but CPU works.

## Tech Stack

PyTorch, pretty_midi, music21, matplotlib, seaborn, NumPy, pandas
