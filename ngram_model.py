from typing import List, Dict
from collections import Counter, defaultdict
import random
from datasets import load_dataset


class NGramModel:
    def __init__(self, dataset, n=3):
        self.dataset = dataset
        self.n = n
        self.ngram_model = self.build_ngram_model()


    def space_tokenizer(self, text: str) -> List[str]:
        """Splits a string into a list of words (tokens)"""
        return text.split()
    
    def generate_ngrams(self, text: str) -> List[tuple]:
        """Generates n-grams from a paragraph"""

        tokens = self.space_tokenizer(text)

        num_of_tokens = len(tokens)

        ngrams = [tuple(tokens[i:i+self.n]) for  i in range(0, num_of_tokens - self.n + 1)]


        return ngrams
    
    def generate_ngram_counts(self) -> Dict[str, Counter]:
        """Generates n-gram counts for the dataset"""

        ngram_counts = defaultdict(Counter)

        for paragraph in self.dataset:

            ngrams_list = self.generate_ngrams(paragraph)

            for ngram in ngrams_list:
                context = " ".join(ngram[:-1])
                ngram_counts[context][ngram[-1]] += 1

        return dict(ngram_counts)

    def build_ngram_model(self) -> Dict[str, Dict[str, float]]:
        """Build an n-gram language model."""

        ngram_model = defaultdict(dict)

        ngram_counts = self.generate_ngram_counts()

        for context, next_tokens in ngram_counts.items():
            total = sum(next_tokens.values())
            for next_token, counts in next_tokens.items():
                ngram_model[context][next_token] = counts / total

        return ngram_model

    def generate_output(
        self,
        prompt: str,
        num_of_tokens: int,
    ) -> str:
        """Generates new token output for ngram model"""

        generated_tokens = self.space_tokenizer(prompt)

    
        for _ in range(num_of_tokens):
            context = generated_tokens[-(self.n - 1):]
            context = " ".join(context)
            if context in self.ngram_model:
                next_word = random.choices(
                    list(self.ngram_model[context].keys()),
                    weights=self.ngram_model[context].values()
                )[0]

                generated_tokens.append(next_word)
            else:
                print("No valid continuation found")
                break

        return " ".join(generated_tokens)


if __name__ == "__main__":
    prompt = "This is why"
    naija_web = load_dataset("saheedniyi/naijaweb")
    dataset = naija_web["train"]["text"][:500]
    
    ngram_model = NGramModel(dataset=dataset)
    print(ngram_model.generate_output(prompt, 10))