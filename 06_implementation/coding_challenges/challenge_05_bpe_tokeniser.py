"""
Challenge 05: BPE Tokeniser from Scratch
=========================================
Implement Byte-Pair Encoding (BPE) tokenisation from scratch using only the
Python standard library (no external NLP libraries).

BPE is the tokenisation algorithm used by GPT-2, GPT-3, GPT-4, Llama, and
most modern LLMs. Understanding it is essential for:
  - Debugging tokenisation issues (out-of-vocabulary tokens, split words)
  - Understanding vocabulary size tradeoffs
  - Answering tokeniser-related interview questions

Algorithm overview:
  1. Start with a character-level vocabulary (+ special tokens).
  2. Count all adjacent symbol pairs in the corpus.
  3. Merge the most frequent pair into a new symbol.
  4. Repeat for a fixed number of merges (= vocab_size - initial_vocab_size).
  5. At encode time, apply merges in order (highest priority first).

This implementation is educational. Production BPE (e.g., tiktoken) is ~100x
faster due to C extensions and regex pre-tokenisation.

Learning objectives:
  1. Understand the BPE merge algorithm.
  2. See how subword units emerge from frequency statistics.
  3. Implement encode/decode correctly including the word-boundary marker.
  4. Understand the tradeoffs between vocab size and tokenisation quality.
"""

import re
from collections import Counter, defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Core BPE Implementation
# ---------------------------------------------------------------------------

class BPETokeniser:
    """
    Byte-Pair Encoding tokeniser trained on a text corpus.

    Args:
        vocab_size: Target vocabulary size (including base characters).
        unk_token:  Token used for characters not in the vocabulary.

    Attributes:
        vocab:       dict mapping token string -> integer ID.
        id_to_token: dict mapping integer ID -> token string.
        merges:      Ordered list of (a, b) merge pairs learned during training.
                     Order matters: earlier merges have higher priority.
        word_cache:  Cache of {word: [token, ...]} for fast repeated encoding.
    """

    # End-of-word marker. We append this to every word before tokenising.
    # This allows the tokeniser to distinguish "ing" inside a word
    # from "ing" at the end of a word (which would be "ing</w>").
    EOW = "</w>"

    def __init__(self, vocab_size: int = 500, unk_token: str = "<unk>") -> None:
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.word_cache: dict[str, list[str]] = {}
        self._is_trained = False

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, corpus: str) -> None:
        """
        Learn BPE merges from a text corpus.

        Args:
            corpus: Raw text string to learn from.
        """
        # Step 1: Pre-tokenise corpus into words and count word frequencies.
        # In this simplified version, a "word" is a whitespace-separated token.
        # GPT-2's actual BPE uses a regex to split on punctuation, numbers, etc.
        word_freq = self._count_words(corpus)

        # Step 2: Represent each word as a sequence of characters + EOW marker.
        # e.g., "hello" -> ("h", "e", "l", "l", "o</w>")
        # The EOW marker tells the tokeniser where words end.
        vocab_counts: dict[tuple[str, ...], int] = {
            self._word_to_chars(word): freq
            for word, freq in word_freq.items()
        }

        # Step 3: Build initial character-level vocabulary.
        base_vocab: set[str] = set()
        for char_seq in vocab_counts:
            for char in char_seq:
                base_vocab.add(char)

        # Add special tokens
        all_tokens = [self.unk_token] + sorted(base_vocab)
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

        # Step 4: Iteratively perform merges until vocab_size is reached.
        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            self._is_trained = True
            return

        for merge_idx in range(num_merges):
            # Count all adjacent pairs across the corpus
            pair_counts = self._count_pairs(vocab_counts)
            if not pair_counts:
                break  # No more pairs to merge

            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break  # No pair appears more than once; stop early

            # Merge this pair in all words
            new_token = best_pair[0] + best_pair[1]
            vocab_counts = self._apply_merge(vocab_counts, best_pair)

            # Record the merge and add the new token to vocabulary
            self.merges.append(best_pair)
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token

        self._is_trained = True
        self.word_cache = {}  # Clear cache after training

    def _count_words(self, corpus: str) -> Counter:
        """Split corpus into words and count frequencies."""
        # Lowercase and split on whitespace+punctuation (simplified)
        words = re.findall(r"[a-zA-Z']+|[^\s]", corpus)
        return Counter(words)

    def _word_to_chars(self, word: str) -> tuple[str, ...]:
        """
        Split a word into characters with EOW appended to the last character.
        "hello" -> ("h", "e", "l", "l", "o</w>")
        """
        if len(word) == 1:
            return (word + self.EOW,)
        return tuple(list(word[:-1]) + [word[-1] + self.EOW])

    def _count_pairs(
        self, vocab_counts: dict[tuple[str, ...], int]
    ) -> Counter:
        """
        Count all adjacent symbol pairs in the current vocabulary representation.

        For each word representation (tuple of symbols) with frequency f,
        every adjacent pair (symbol_i, symbol_{i+1}) gets f added to its count.
        """
        pair_counts: Counter = Counter()
        for symbols, freq in vocab_counts.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    def _apply_merge(
        self,
        vocab_counts: dict[tuple[str, ...], int],
        pair: tuple[str, str],
    ) -> dict[tuple[str, ...], int]:
        """
        Apply a merge operation to all word representations.

        Replaces all occurrences of adjacent (pair[0], pair[1]) with
        the concatenated token pair[0]+pair[1].
        """
        a, b = pair
        merged = a + b
        new_vocab: dict[tuple[str, ...], int] = {}

        for symbols, freq in vocab_counts.items():
            # Replace each occurrence of (a, b) with merged
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab[tuple(new_symbols)] = freq

        return new_vocab

    # -----------------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            text: Input text string.
        Returns:
            List of integer token IDs.
        """
        assert self._is_trained, "Call train() before encode()"

        words = re.findall(r"[a-zA-Z']+|[^\s]", text)
        token_ids: list[int] = []

        for word in words:
            tokens = self._encode_word(word)
            for token in tokens:
                token_ids.append(
                    self.vocab.get(token, self.vocab.get(self.unk_token, 0))
                )

        return token_ids

    def _encode_word(self, word: str) -> list[str]:
        """
        Encode a single word into a list of BPE tokens.

        Uses the learned merge priority list: apply merges in order.
        The cache avoids recomputing the same word twice.
        """
        if word in self.word_cache:
            return self.word_cache[word]

        # Start with character-level representation
        symbols = list(self._word_to_chars(word))

        # Apply merges greedily in priority order (earlier merges = higher priority)
        merge_index = {pair: idx for idx, pair in enumerate(self.merges)}

        while len(symbols) > 1:
            # Find the highest-priority merge that can be applied
            best_pair: Optional[tuple[str, str]] = None
            best_priority = float('inf')
            best_pos = -1

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in merge_index and merge_index[pair] < best_priority:
                    best_priority = merge_index[pair]
                    best_pair = pair
                    best_pos = i

            if best_pair is None:
                break  # No applicable merges remaining

            # Apply the merge at the found position
            a, b = best_pair
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i == best_pos:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        self.word_cache[word] = symbols
        return symbols

    # -----------------------------------------------------------------------
    # Decoding
    # -----------------------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        The EOW marker signals a word boundary: the following token starts
        a new word (separated by a space).

        Args:
            token_ids: List of integer token IDs.
        Returns:
            Decoded text string.
        """
        tokens = [self.id_to_token.get(tid, self.unk_token) for tid in token_ids]
        text = ""
        for token in tokens:
            if token == self.unk_token:
                text += token
            elif token.endswith(self.EOW):
                # Remove EOW marker and add trailing space (word boundary)
                text += token[:-len(self.EOW)] + " "
            else:
                text += token

        return text.rstrip()  # Remove trailing space from last word

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def token_to_id(self, token: str) -> int:
        """Look up a token string's ID."""
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))

    def id_to_token_str(self, token_id: int) -> str:
        """Look up an ID's token string."""
        return self.id_to_token.get(token_id, self.unk_token)

    def tokenise(self, text: str) -> list[str]:
        """Return the list of token strings (not IDs) for a given text."""
        assert self._is_trained, "Call train() before tokenise()"
        words = re.findall(r"[a-zA-Z']+|[^\s]", text)
        tokens: list[str] = []
        for word in words:
            tokens.extend(self._encode_word(word))
        return tokens

    def vocab_size_actual(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# A small training corpus to demonstrate BPE
CORPUS = """
the cat sat on the mat the cat is on the mat
the dog sat on the log the dog is on the log
cats and dogs are common pets
sitting on the mat is comfortable
the sitting cat is comfortable
tokenization is the process of splitting text into tokens
byte pair encoding is a subword tokenization algorithm
"""


def test_training():
    """Verify that training produces the expected vocabulary size."""
    print("Test 1: BPE training")
    tokeniser = BPETokeniser(vocab_size=100)
    tokeniser.train(CORPUS)

    print(f"  Target vocab size: 100")
    print(f"  Actual vocab size: {tokeniser.vocab_size_actual()}")
    print(f"  Number of merges learned: {len(tokeniser.merges)}")
    assert tokeniser.vocab_size_actual() <= 100 + 5, \
        f"Vocab too large: {tokeniser.vocab_size_actual()}"
    assert len(tokeniser.merges) > 0, "No merges learned"

    # Show the first 10 merges
    print("  First 10 merges:")
    for i, merge in enumerate(tokeniser.merges[:10]):
        print(f"    {i+1:2d}: '{merge[0]}' + '{merge[1]}' -> '{merge[0]+merge[1]}'")
    print("  PASSED")


def test_encode_decode_roundtrip():
    """Verify that encode followed by decode recovers the original text."""
    print("Test 2: Encode-decode roundtrip")
    tokeniser = BPETokeniser(vocab_size=150)
    tokeniser.train(CORPUS)

    test_sentences = [
        "the cat sat on the mat",
        "dogs are common pets",
        "sitting is comfortable",
    ]

    for sentence in test_sentences:
        token_ids = tokeniser.encode(sentence)
        decoded = tokeniser.decode(token_ids)
        print(f"  Original: '{sentence}'")
        print(f"  Token IDs: {token_ids}")
        print(f"  Decoded:   '{decoded}'")
        # After BPE, decoding should reproduce the original text
        assert decoded == sentence, \
            f"Roundtrip failed: '{decoded}' != '{sentence}'"
        print()

    print("  PASSED")


def test_tokenisation():
    """Verify that common words are tokenised as expected."""
    print("Test 3: Tokenisation inspection")
    tokeniser = BPETokeniser(vocab_size=200)
    tokeniser.train(CORPUS)

    words_to_check = ["the", "cat", "tokenization", "algorithm"]
    for word in words_to_check:
        tokens = tokeniser.tokenise(word)
        print(f"  '{word}' -> {tokens}")

    # Common frequent words should be single tokens
    the_tokens = tokeniser.tokenise("the")
    print(f"\n  'the' tokenised into {len(the_tokens)} token(s): {the_tokens}")
    # "the" appears very frequently, so it should be a single token after training
    assert len(the_tokens) == 1, \
        f"'the' should be a single token but got: {the_tokens}"
    print("  PASSED")


def test_unknown_words():
    """Verify graceful handling of words with unseen characters."""
    print("Test 4: Unknown character handling")
    tokeniser = BPETokeniser(vocab_size=100)
    tokeniser.train(CORPUS)

    # Chinese characters are not in the training corpus
    ids = tokeniser.encode("hello 你好")
    print(f"  'hello 你好' -> token IDs: {ids}")
    # Should not raise an error; unknown chars get unk_token
    assert len(ids) > 0
    print("  PASSED")


def test_subword_splitting():
    """
    Demonstrate that BPE splits rare words into subword units.
    Train on corpus without "tokenization" appearing often, then tokenise it.
    """
    print("Test 5: Subword splitting")
    small_corpus = "the cat sat on the mat the cat is on the mat " * 50
    tokeniser = BPETokeniser(vocab_size=80)
    tokeniser.train(small_corpus)

    # "tokenization" is not in the training corpus
    word = "tokenization"
    tokens = tokeniser.tokenise(word)
    print(f"  '{word}' with limited vocab -> {tokens}")
    # Should be split into multiple subword tokens
    assert len(tokens) > 1, "Expected subword splitting for unseen word"
    print("  PASSED")


def test_vocabulary_coverage():
    """Check that frequent substrings appear in the vocabulary."""
    print("Test 6: Vocabulary coverage")
    tokeniser = BPETokeniser(vocab_size=200)
    tokeniser.train(CORPUS)

    # "the</w>" should be in vocab — it's the most frequent word
    eow_token = "the" + BPETokeniser.EOW
    in_vocab = eow_token in tokeniser.vocab
    print(f"  'the</w>' in vocab: {in_vocab}")
    assert in_vocab, "'the</w>' should be in vocabulary after training on this corpus"

    # "on" and "is" are also frequent
    for word in ["on", "is", "the", "cat"]:
        token = word + BPETokeniser.EOW
        print(f"  '{token}' in vocab: {token in tokeniser.vocab}")

    print("  PASSED")


def demo_bpe_merge_process():
    """
    Walk through the BPE merge process step-by-step on a tiny corpus.
    This is the clearest explanation of how BPE works.
    """
    print("\nDemo: BPE merge process on tiny corpus")
    print("-" * 50)

    # Tiny corpus to trace
    tiny_corpus = "low lower newest widest"
    print(f"Corpus: '{tiny_corpus}'")
    print()

    # Manual character-level representation
    words = {
        ("l", "o", "w</w>"): 1,
        ("l", "o", "w", "e", "r</w>"): 1,
        ("n", "e", "w", "e", "s", "t</w>"): 1,
        ("w", "i", "d", "e", "s", "t</w>"): 1,
    }
    print("Initial char-level vocabulary:")
    for symbols, freq in words.items():
        print(f"  {list(symbols)} : {freq}")
    print()

    # Count initial pairs
    pair_counts: Counter = Counter()
    for symbols, freq in words.items():
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i+1])] += freq

    print("Top pair counts:")
    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {pair}: {count}")
    print()

    # First merge: ('e', 's') or similar most frequent
    best = max(pair_counts, key=pair_counts.get)
    print(f"Merge 1: {best[0]} + {best[1]} -> '{best[0]+best[1]}'")
    print()
    print("(Continue: the tokeniser above automates all merge steps)")


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge 05: BPE Tokeniser")
    print("=" * 60)
    print()

    test_training()
    print()
    test_encode_decode_roundtrip()
    print()
    test_tokenisation()
    print()
    test_unknown_words()
    print()
    test_subword_splitting()
    print()
    test_vocabulary_coverage()
    print()
    demo_bpe_merge_process()
    print()
    print("All tests passed.")
