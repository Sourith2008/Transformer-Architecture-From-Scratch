# Transformer from Scratch (English → Bengali)

A from-scratch PyTorch implementation of the **Transformer architecture** ("Attention Is All You Need"), built for a sequence-to-sequence machine translation task from **English to Bengali**. The model implements every core component manually — scaled dot-product attention, multi-head self/cross-attention, positional encoding, layer normalization, and the encoder-decoder stack — without relying on `torch.nn.Transformer` or other high-level abstractions.

This project is primarily intended as an **educational implementation** to understand how Transformers work internally.

---

## ✨ Features

- **Scaled Dot-Product Attention** implemented from first principles
- **Multi-Head Self-Attention** for the encoder
- **Multi-Head Cross-Attention** for the decoder
- **Sinusoidal Positional Encoding**
- **Custom Layer Normalization**
- **Position-wise Feed-Forward Network**
- **Sentence Embedding** with tokenization, start/end tokens, and padding
- Full **Encoder–Decoder Transformer** stack with configurable depth
- Designed for **GPU acceleration** (auto-detects CUDA availability)

---

## 🏗️ Architecture

The implementation follows the standard Transformer design:

```
Input Sentence (English)          Target Sentence (Bengali)
        │                                   │
        ▼                                   ▼
 Sentence Embedding                 Sentence Embedding
        │                                   │
        ▼                                   ▼
   [ Encoder Layers ] ───────────►   [ Decoder Layers ]
   (Self-Attention +                 (Self-Attention +
    Feed Forward)                     Cross-Attention +
                                       Feed Forward)
                                              │
                                              ▼
                                     Linear Projection
                                              │
                                              ▼
                                   Output Vocabulary Logits
```

### Core Modules

| Module | Description |
|---|---|
| `scaled_dot_product` | Computes attention scores and weighted values |
| `MultiHeadAttention` | Self-attention used inside the encoder and decoder |
| `MultiHeadCrossAttention` | Cross-attention connecting encoder outputs to the decoder |
| `Positional_Encoding` | Generates sinusoidal position embeddings |
| `LayerNormalization` | Custom layer norm with learnable gain/bias |
| `FeedForward` | Two-layer feed-forward network with ReLU and dropout |
| `SentenceEmbedding` | Tokenizes text and combines token + positional embeddings |
| `EncoderLayer` / `Encoder` | Stacked self-attention + feed-forward blocks |
| `DecoderLayer` / `Decoder` | Stacked self-attention + cross-attention + feed-forward blocks |
| `Transformer` | Combines encoder and decoder into the full model |

---

## 📦 Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA support recommended for training)
- NumPy

Install dependencies:

```bash
pip install torch numpy
```

---

## 🚀 Usage

Import and instantiate the model with your desired hyperparameters and vocabulary mappings:

```python
from transformer import Transformer

model = Transformer(
    d_model=512,
    max_len=200,
    num_heads=8,
    ffn_hidden=2048,
    num_layers=6,
    bn_vocab_size=len(bengali_vocab),
    bengali_to_index=bengali_to_index,
    english_to_index=english_to_index,
    START_TOKEN=START_TOKEN,
    END_TOKEN=END_TOKEN,
    PADDING_TOKEN=PADDING_TOKEN,
)

output = model(
    english_batch,
    bengali_batch,
    encoder_self_attention_mask=None,
    decoder_self_attention_mask=None,
    decoder_cross_attention_mask=None,
)
```

Where:
- `english_to_index` / `bengali_to_index` are dictionaries mapping tokens (characters/words) to vocabulary indices
- `START_TOKEN`, `END_TOKEN`, `PADDING_TOKEN` are special tokens used during tokenization
- `english_batch` / `bengali_batch` are lists of raw sentence strings

The model runs on GPU automatically if `torch.cuda.is_available()` returns `True`, otherwise it falls back to CPU.

---

## ⚙️ Hyperparameters

| Parameter | Description |
|---|---|
| `d_model` | Embedding / hidden dimension size |
| `num_heads` | Number of attention heads |
| `ffn_hidden` | Hidden layer size in the feed-forward network |
| `num_layers` | Number of stacked encoder/decoder layers |
| `max_len` | Maximum sequence length |
| `bn_vocab_size` | Size of the target (Bengali) vocabulary |

---

## 📌 Notes & Limitations

- This is a **learning-focused, work-in-progress implementation**. Some components (e.g. mask handling in the decoder, a few method calls) are still being refined and may need debugging before training end-to-end.
- Tokenization is minimal (character/word-level) and assumes pre-built `language_to_index` vocabularies for both source and target languages.
- No training script, data loader, or checkpointing is included in this file — this module defines the model architecture only.

Contributions and fixes are welcome via pull requests.

---

## 📖 References

- Vaswani et al., *["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)* (2017)

---

## 📄 License

This project is open-sourced for educational purposes. Add a license of your choice (e.g. MIT) if distributing publicly.
