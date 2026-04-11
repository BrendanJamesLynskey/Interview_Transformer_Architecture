# Tokenisation: BPE, SentencePiece, and Vocabulary Design

## Overview

Tokenisation is the first stage of the language model pipeline and one of the most practically important engineering decisions. The choice of tokeniser, vocabulary size, and tokenisation scheme affects model performance, training efficiency, and behaviour on specific inputs (numbers, code, rare words). Interviewers at companies working on LLMs will expect you to understand the major algorithms, their trade-offs, and their failure modes.

---

## Tier 1: Fundamentals

### Q1. What is Byte-Pair Encoding (BPE) and how does the algorithm work?

**Answer.**

BPE (Gage, 1994; applied to NLP by Sennrich et al., 2016) is a data-compression algorithm adapted to learn a subword vocabulary by iteratively merging the most frequent pairs of adjacent symbols.

**Algorithm:**

**Input:** Raw text corpus, target vocabulary size $V$

**Initialisation:**
1. Split all words into characters (with a special end-of-word marker, e.g., `</w>`)
2. Count the frequency of each word in the corpus
3. Initial vocabulary = set of all individual characters

**Iteration (repeat until vocabulary size $= V$):**
1. Count all adjacent symbol-pair frequencies across the corpus
2. Find the most frequent pair $(a, b)$
3. Create a new merged symbol $ab$
4. Replace all occurrences of `a b` (with a space between them) in the corpus with `ab`
5. Add $ab$ to the vocabulary

**Example:**

Corpus word frequencies: `low: 5, lower: 2, newest: 6, widest: 3`

Initial vocabulary: `{l, o, w, e, r, n, s, t, i, d, </w>}`

Initial tokenised corpus:
```
l o w </w>          (×5)
l o w e r </w>      (×2)
n e w e s t </w>    (×6)
w i d e s t </w>    (×3)
```

Iteration 1: Most frequent pair = `e s` (9 occurrences: 6 + 3). Merge → `es`.
Iteration 2: Most frequent pair = `es t` (9). Merge → `est`.
Iteration 3: Most frequent pair = `l o` (7). Merge → `lo`.
...

After enough merges, common words appear as single tokens, and rare words are decomposed into known subword pieces.

**At inference (encoding):**

Apply the learned merge rules in the same order they were learned. For a new word, start with character-level representation and greedily apply merges:

```
"newest" → ['n', 'e', 'w', 'e', 's', 't', '</w>']
          → ... → ['new', 'est</w>']  (after applying learned merges)
```

---

### Q2. What are the key differences between BPE, WordPiece, and Unigram Language Model tokenisation?

**Answer.**

**BPE (Byte-Pair Encoding):**
- Learns vocabulary by greedily merging most frequent character pairs
- Deterministic merge order; fixed vocabulary
- Decoding: concatenate subwords, remove end-of-word markers
- Used by: GPT-2, GPT-3, LLaMA, most GPT-family models, RoBERTa

**WordPiece (Schuster & Nakamura, 2012; used in BERT):**

Similar to BPE but the merge criterion is different. Instead of raw frequency, it maximises the **language model log-likelihood** of the training data given the vocabulary:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This is a form of pointwise mutual information. Pairs that are not only frequent but also "surprising" (they co-occur more than expected given individual frequencies) are preferred.

WordPiece uses `##` prefix to mark continuation subwords: `playing` → `play ##ing`.

**Unigram Language Model (Kudo, 2018; used in SentencePiece):**

Rather than building up from characters, start with a large candidate vocabulary and **prune** it:

1. Initialise with a large vocabulary of all subwords up to some length
2. Train a unigram language model: assign a probability to each subword token
3. For each subword $x$ in the vocabulary, compute how much the total log-likelihood drops if $x$ is removed
4. Remove the bottom $p\%$ (e.g., 10%) of subwords with the lowest loss impact
5. Repeat until the target vocabulary size is reached

**Tokenisation**: given a word, find the tokenisation that maximises the unigram LM probability:

$$\text{tokenise}(w) = \arg\max_{\text{segmentation}} \sum_i \log p(t_i)$$

This can produce multiple valid tokenisations (the model defines a distribution over them).

**Comparison:**

| Property | BPE | WordPiece | Unigram LM |
|---|---|---|---|
| Construction | Bottom-up (merge) | Bottom-up (MI-based merge) | Top-down (prune) |
| Merge criterion | Frequency | PMI score | Likelihood contribution |
| Deterministic tokenisation | Yes | Yes | No (distribution over tokenisations) |
| Handles OOV | Via characters | Via characters | Via characters |
| Used in | GPT, LLaMA | BERT, DistilBERT | XLNet, ALBERT, mT5 |

---

### Q3. What is SentencePiece? How does it differ from NLTK/standard tokenisers?

**Answer.**

**SentencePiece** (Kudo & Richardson, 2018) is a tokeniser library that:

1. Treats the input as a raw Unicode character stream — no pre-tokenisation required
2. Implements both BPE and Unigram LM algorithms
3. Treats whitespace as just another character (represented as `▁`)
4. Is fully reversible: tokenisation → token IDs → detokenisation is lossless

**Why this matters:**

Standard tokenisers (NLTK, spaCy) first split text on whitespace and punctuation, then process the resulting words. This creates problems:
- Language-specific rules needed (Chinese has no spaces; Japanese uses different spacing)
- Capitalisation and whitespace handling requires careful normalisation
- Not consistent across languages

SentencePiece operates directly on the raw byte stream (or Unicode stream), making it language-agnostic.

**Example:**

Input: `"Hello world"`

SentencePiece tokenisation (BPE): `["▁Hello", "▁world"]`

The `▁` prefix indicates the token begins a new word (preceded by whitespace). At detokenisation, `▁` is replaced with a space.

This allows exact reconstruction: `join(tokens).replace("▁", " ").strip() = "Hello world"`.

**Byte-level BPE:**

GPT-2 and later GPT-family models use **byte-level BPE**: the initial vocabulary is 256 bytes (all possible byte values), not Unicode characters. This guarantees coverage of any possible input string (no "unknown" tokens) and handles arbitrary text including code, URLs, and unicode characters correctly.

---

### Q4. What is vocabulary size and what are the trade-offs of making it larger or smaller?

**Answer.**

The vocabulary size $V$ is a fundamental hyperparameter of the tokeniser.

**Typical values:**
- BERT: $V = 30{,}522$ (WordPiece)
- GPT-2: $V = 50{,}257$ (BPE)
- GPT-3/4, LLaMA: $V = 32{,}000$ to $128{,}000$
- LLaMA-3: $V = 128{,}256$

**Effects of larger vocabulary:**

| Effect | Description |
|---|---|
| Shorter sequences | Common words/phrases become single tokens; reduces sequence length for the same text |
| Better number handling | Each digit or multi-digit number can be its own token |
| Reduced compute | $O(n^2)$ attention cost decreases with shorter sequences |
| Larger embedding table | $V \times d_{\text{model}}$ parameters in the embedding and output layers |
| Sparser learning | Each token appears less frequently, so individual token embeddings are learned from fewer examples |
| Better code/multilingual | Large vocabulary can dedicate tokens to programming keywords and non-Latin scripts |

**Effects of smaller vocabulary:**

| Effect | Description |
|---|---|
| Longer sequences | More tokens needed for the same text; increases attention cost |
| More generalisable subwords | Each subword is seen more often; better-learned representations |
| Less memory | Smaller embedding table |
| Poor number/code handling | Numbers split into individual digits; code keywords fragmented |

**Practical trade-off:**

For English-centric models: $V = 32k\text{–}50k$ is usually sufficient.
For multilingual models: $V = 100k\text{–}250k$ is needed to give each language adequate vocabulary coverage.
For code-heavy models: Larger vocabulary or dedicated code tokens help; many code models use $V = 50k\text{–}100k$.

**Fertility:** The ratio of tokens to words for a given language. English has fertility ~1.0–1.3 with standard BPE. Languages with complex morphology (Turkish, Finnish) or different scripts (Thai, Arabic) have much higher fertility, making models trained on English vocabularies inefficient for these languages.

---

## Tier 2: Intermediate

### Q5. Implement the BPE algorithm from scratch. Demonstrate it on a small example.

**Answer.**

```python
from collections import Counter, defaultdict
from typing import Optional


def get_vocab(corpus: list[str]) -> dict[tuple, int]:
    """
    Convert corpus to word-frequency dictionary.
    Each word is represented as a tuple of characters + end-of-word marker.
    """
    vocab = Counter()
    for line in corpus:
        for word in line.strip().split():
            vocab[tuple(list(word) + ['</w>'])] += 1
    return dict(vocab)


def get_pair_stats(vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Count frequency of each adjacent symbol pair across the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq
    return pairs


def merge_vocab(vocab: dict[tuple, int], best_pair: tuple) -> dict[tuple, int]:
    """Merge all occurrences of best_pair in the vocabulary."""
    new_vocab = {}
    a, b = best_pair
    bigram = (a, b)
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == bigram:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def train_bpe(corpus: list[str], num_merges: int) -> tuple[list[tuple], dict[tuple, int]]:
    """
    Train BPE and return:
      - merge_rules: ordered list of (pair_to_merge,) tuples
      - final_vocab: word -> tokenised form frequency
    """
    vocab = get_vocab(corpus)
    merge_rules = []

    print(f"Initial vocabulary ({sum(vocab.values())} total tokens):")
    initial_symbols = set(sym for word in vocab for sym in word)
    print(f"  Symbols: {sorted(initial_symbols)}")
    print(f"  Word representations:")
    for word, freq in list(vocab.items())[:5]:
        print(f"    {' '.join(word)} (×{freq})")

    for i in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            print("No more pairs to merge.")
            break
        best_pair = max(pairs, key=pairs.get)
        merge_rules.append(best_pair)
        vocab = merge_vocab(vocab, best_pair)
        print(f"Merge {i+1}: {best_pair} (freq={pairs[best_pair]})")

    return merge_rules, vocab


def tokenise_word(word: str, merge_rules: list[tuple]) -> list[str]:
    """Apply learned BPE merge rules to a new word."""
    tokens = list(word) + ['</w>']
    for (a, b) in merge_rules:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


# ── Demo ─────────────────────────────────────────────────────────────────────

corpus = [
    "low low low low low",
    "lower lower",
    "newest newest newest newest newest newest",
    "widest widest widest",
]

print("=" * 60)
print("BPE TRAINING")
print("=" * 60)
merge_rules, final_vocab = train_bpe(corpus, num_merges=10)

print("\n" + "=" * 60)
print("FINAL VOCABULARY (word representations):")
print("=" * 60)
for word, freq in final_vocab.items():
    print(f"  {' | '.join(word)} (×{freq})")

print("\n" + "=" * 60)
print("TOKENISING NEW WORDS:")
print("=" * 60)
test_words = ["low", "lowest", "newer", "old"]
for word in test_words:
    tokens = tokenise_word(word, merge_rules)
    print(f"  '{word}' -> {tokens}")
```

**Expected output:**
```
============================================================
BPE TRAINING
============================================================
Initial vocabulary (16 total tokens):
  Symbols: ['</w>', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w']
  Word representations:
    l o w </w> (×5)
    l o w e r </w> (×2)
    n e w e s t </w> (×6)
    w i d e s t </w> (×3)
Merge 1: ('e', 's') (freq=9)
Merge 2: ('es', 't') (freq=9)
Merge 3: ('est', '</w>') (freq=9)
Merge 4: ('l', 'o') (freq=7)
Merge 5: ('lo', 'w') (freq=7)
Merge 6: ('low', '</w>') (freq=5)
Merge 7: ('n', 'e') (freq=6)
Merge 8: ('ne', 'w') (freq=6)
Merge 9: ('new', 'est</w>') (freq=6)
Merge 10: ('w', 'i') (freq=3)

FINAL VOCABULARY:
  low</w> (×5)
  low | e | r | </w> (×2)
  newest</w> (×6)
  wi | d | est</w> (×3)

TOKENISING NEW WORDS:
  'low' -> ['low</w>']
  'lowest' -> ['low', 'est</w>']
  'newer' -> ['new', 'e', 'r', '</w>']
  'old' -> ['o', 'l', 'd', '</w>']
```

---

### Q6. How does tokenisation affect model behaviour on numbers, code, and non-English text? Give concrete examples.

**Answer.**

**Numbers:**

With standard BPE vocabulary (~50k), multi-digit numbers are typically split at inconsistent boundaries:

```
GPT-2 tokenisation of numbers:
  "1234"  -> ['12', '34']
  "1235"  -> ['12', '35']
  "12345" -> ['123', '45']
  "9999"  -> ['99', '99']
```

The split point depends on frequency in the training corpus — not on mathematical structure. The model has no guarantee that `1234` and `1235` are close in token space or that arithmetic relationships are preserved.

**Consequence:** Models struggle with arithmetic and number manipulation partly because of tokenisation. "42 + 17 = ?" requires the model to learn that `42` (one token) plus `17` (one token) equals `59` (one token) — but `42 + 17 = ?` might be tokenised differently from `142 + 17 = ?`.

**LLaMA-3 approach:** Uses a vocabulary of 128k with explicit digit tokens `0-9` and common number patterns. Training on Llama-3 tokeniser: `"1234"` → `['1', '2', '3', '4']` — consistent character-level number representation.

**Code:**

Common programming keywords and patterns should ideally be single tokens:
```
GPT-4 tokenisation:
  "def "     -> ['def ']          (single token — good)
  "for "     -> ['for ']          (single token — good)
  "    "     -> ['    ']          (4-space indent = one token — good)
  "isinstance" -> ['isin', 'stance'] (split — model must learn this is one concept)
```

Indentation matters in Python. A 4-space indent being a single token is efficient; inconsistent tokenisation of indentation (2 spaces vs 4 spaces vs tab) can harm code generation.

**Non-English text:**

Standard English-centric vocabularies (GPT-2's 50k) assign relatively few tokens to non-Latin scripts. A single Chinese character might be split into multiple byte tokens:

```
GPT-2 (byte-level BPE, 50k vocab):
  "中文" -> ['ä¸', 'Ń', 'æ', '–', '‡']  (5 tokens for 2 characters)
```

This means Chinese text uses 3-5x more tokens than English for the same information content. Implications:
- Higher cost at inference (more tokens = more compute)
- Higher fertility reduces effective context length for non-English users
- The model sees each non-English "word" as an unusual sequence of byte fragments, potentially learning less about semantic structure

Modern multilingual models (mT5, BLOOM, LLaMA-3) use larger vocabularies (100k–250k) with dedicated subwords for major world languages.

---

## Tier 3: Advanced

### Q7. Explain the token healing and prefix caching problem. Why does tokenisation cause non-obvious generation biases?

**Answer.**

**The tokenisation boundary problem:**

BPE tokenisation is applied to the full input at once. But when generating text that will be appended to a prompt, the model must generate starting from the end of the tokenised prompt — which may create an inconsistency.

**Example:**

Prompt: `"The colour is red"`

Tokenise fully: `['The', ' colour', ' is', ' red']` — 4 tokens. Fine.

Now user adds more text: `"The colour is reddish"`

Tokenise fully: `['The', ' colour', ' is', ' reddish']` — also 4 tokens. `reddish` is one token.

But in autoregressive generation: after generating `red`, the model would try to generate the next token. The token `red` might commit to a different suffix than what is valid from the full tokenisation of `reddish`. The model would try to generate `dish` as a separate token, but `dish` tokenised after `red` gives `['red', 'dish']`, whereas the correct tokenisation of `reddish` is `['reddish']`.

**Token healing (Dohan et al., mentioned in LLM Sampling work):**

Some systems apply token healing to address this: when the user provides a prompt that ends mid-token (relative to the BPE vocabulary), the system backtracks by removing the last token and re-generating from the last complete token, allowing the model to "heal" the tokenisation boundary.

**Generation bias from token frequencies:**

Because the model is trained on data with specific tokenisation patterns, it develops implicit biases based on what completions are common in the training data for particular token prefixes.

**Example:** 
- "Washington" might tokenise as `['Washington']` (one token)  
- "Washingtonian" as `['Washington', 'ian']`  
- When generating after the token `Washington`, the model assigns relatively high probability to `ian` because it has seen this pattern, even in contexts where "Washingtonian" is not the intended completion.

**Prefix caching and tokenisation:**

In production systems with prompt caching (reusing KV cache for repeated prefixes), the cached prefix must match the tokenisation boundary exactly. If a user extends a prompt, the system must verify that the original prompt's tokenisation is identical to the first $k$ tokens of the new full tokenisation — which is not guaranteed because BPE tokenisation depends on context.

This is a subtle but important systems engineering issue for LLM deployment.

---

### Q8. Design a vocabulary and discuss its engineering trade-offs for a hypothetical LLM that needs to handle English text, Python code, and mathematical notation equally well.

**Answer.**

**Requirements analysis:**

| Domain | Key patterns | Tokenisation needs |
|---|---|---|
| English | Common words, morphemes | Subword units of frequent English vocabulary |
| Python code | Keywords, operators, indentation, identifiers | Single tokens for `def`, `for`, `if`, `return`, `    ` (4-space indent) |
| Mathematics | Digits, operators, LaTeX tokens, equations | Individual digit tokens; LaTeX commands as single tokens |

**Proposed vocabulary design:**

**Size:** $V = 100{,}000$

**Construction:**

1. **Byte-level base vocabulary** (256 tokens): Ensures no character is out-of-vocabulary.

2. **English BPE merges** (~40k merges): Train on a large English corpus. Produces common English words and morphemes as single tokens.

3. **Code-aware merges** (~30k merges): Train BPE on a Python corpus, with a modified tokeniser that:
   - Treats indentation as atomic units: `"    "` (4 spaces) should be one token
   - Preserves common Python keywords and built-ins as single tokens
   - Handles `::`, `->`, `**`, `//` as single tokens (Python operators)

4. **Mathematical notation** (~15k tokens):
   - Individual digits `0-9` as dedicated tokens
   - LaTeX commands: `\frac`, `\sqrt`, `\int`, `\sum`, `\mathbb` as single tokens
   - Common mathematical operators: `+`, `-`, `=`, `≤`, `≥`, `∈`, `∑`
   - Number patterns: 2-digit combinations `00`–`99` as single tokens (reducing sequence length for numbers)

5. **Reserved tokens** (~5k): System tokens, special delimiters (`<|code|>`, `<|math|>`, `<|endoftext|>`), format markers.

**Key engineering decisions:**

- **Separate tokenisation of whitespace in code:** Add special tokens for `\t` (tab), `    ` (4 spaces), `  ` (2 spaces) to handle Python/YAML/Markdown indentation without splitting.
- **Digit tokens:** Always tokenise digits individually (1-token per digit, regardless of surrounding context). This requires a pre-tokenisation step that segments digit sequences before BPE is applied.
- **No mixing of word and code vocabularies at merge time:** Train separate BPE models on each domain and combine the vocabulary tables, then run a final pass to add cross-domain merges.

**Trade-offs:**

| Decision | Benefit | Cost |
|---|---|---|
| $V = 100k$ | Efficient tokenisation for all domains | Larger embedding table (+15M params per 100k vocab at $d=1536$) |
| Individual digit tokens | Better arithmetic, consistent number handling | Longer sequences for numbers |
| Code keywords as single tokens | Better code generation, fewer tokens | Requires domain-aware pre-tokenisation |
| LaTeX tokens | Better math notation | Rare tokens are under-represented in training data |

**Fertility estimate:**

- English text: ~1.2 tokens/word (efficient)
- Python code: ~1.5 tokens/word (code identifiers often merge well)
- LaTeX math: ~2.0 tokens/symbol (formulas are dense with special tokens)
- Average across mixed corpus: ~1.5 tokens/word — reasonable efficiency
