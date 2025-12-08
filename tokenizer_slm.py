from typing import List
import re
from datasets import load_dataset

class SimpleTokenizer:

    UNKNOWN_TOKEN = "<UNK>"

    def __init__(self, corpus: List[str], vocabulary: List[str] | None = None):
        """Initializes the tokenizer with texts in the dataset or with a vocabulary."""

        if vocabulary is None:
            if isinstance(corpus, str):
                corpus = [corpus]

            # Convert text sequence to tokens.
            tokens = []
            for text in corpus:
                for token in self.space_tokenize(text):
                    tokens.append(token)

            # Create a vocabulary comprising of unique tokens.
            self.vocabulary = self.build_vocabulary(tokens)

        else:
            self.vocabulary = vocabulary

        self.vocabulary = vocabulary + self.UNKNOWN_TOKEN
        self.vocabulary_size = len(self.vocabulary)

        # Create token and token IDs mappings.
        self.token_to_index = {}
        self.index_to_token = {}

        for index, token in enumerate(self.vocabulary):
            self.token_to_index[token] = index
            self.index_to_token[index] = token

    def space_tokenizer(text: str) -> List[str]:
        """Splits a string into a list of tokens and removes punctuation."""

        # Replace exclamation marks with a space before splitting
        text = re.sub(r'[!;:()"\[\]{}<>/\\`~@#$%^&*\_+=|\n“”]', ' ', text)
        tokens = re.split(r'\s+', text)
        tokens = [token for token in tokens if token]
        return tokens

    def build_vocabulary(self, tokens: List[str]) -> List[str]:
        """Builds a vocabulary from a list of tokens."""
        return list(set(tokens))

    def encode(self, text: str) -> List[int]:
        """Encodes a text sequence into a list of indices."""

        # Convert tokens into indices.
        indices = []
        unk_index = self.token_to_index[self.UNKNOWN_TOKEN]
        for token in self.space_tokenize(text):
            token_index = self.token_to_index.get(token, unk_index)
            indices.append(token_index)

        return indices

    def decode(self, indices: int | List[int]) -> str:
        """Decodes a list (or single index) of integers back into tokens."""

        # If a single integer is passed, convert it into a list.
        if isinstance(indices, int):
            indices = [indices]

        # Map indices to tokens.
        tokens = []
        unk_index = self.token_to_index[self.UNKNOWN_TOKEN]
        for index in indices:
            token = self.index_to_token.get(index, unk_index)
            tokens.append(token)

        return " ".join(tokens)

if __name__ == "__main__":
    prompt = "This is why"
    naija_web = load_dataset("saheedniyi/naijaweb")
    dataset = naija_web["train"]["text"][:500]
    
    tokenizer = SimpleTokenizer(dataset)
    print(tokenizer.encode(prompt))
    print(tokenizer.decode([102]))