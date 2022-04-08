import itertools
from typing import List

import numpy as np

from artificial_language.languages.language import Language

from .random_sampling_language import RandomSamplingLanguage


def softmax(x, temperature: float):

    temperatured_x = x / temperature

    y = np.exp(temperatured_x - np.max(temperatured_x))
    f_x = y / y.sum()
    return f_x


@Language.register("random_walk")
class RandomWalkLanguage(RandomSamplingLanguage):
    """
    A Latent Variable Model Approach to PMI-based Word Embeddings
    https://aclanthology.org/Q16-1028/

    A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS
    https://openreview.net/pdf?id=SyK00v5xx
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 10,
        alpha: float = 0.0,
        walk_stride: float = 0.1,
        temperature: float = 1.0,
        reset_every_sentence: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self.embeddings = np.random.normal(size=(vocab_size, embedding_dim))

        self.embedding_dim = embedding_dim

        self.context_vector = None
        self.walk_stride = walk_stride
        self.temperature = temperature
        self.reset()

        norms = np.linalg.norm(self.embeddings, axis=1)
        vec_norm_order = np.argsort(-norms)

        sorting_index = np.argsort(vec_norm_order)
        unigram_prob = np.array([1 / i for i in np.arange(1, self.vocab_size + 1)])[sorting_index]
        self.unigram_prob = unigram_prob / unigram_prob.sum()
        self.alpha = alpha

        self.reset_every_sentence = reset_every_sentence

    def walk(self):
        direction = np.random.normal(size=self.embedding_dim)
        magnitude = self.walk_stride / np.sqrt(self.embedding_dim)
        self.context_vector += direction * magnitude

    def batch_generate_sentences(self, num_sentences: int):
        length_list = [self.sentence_length_sampler.sample_length() for _ in range(num_sentences)]
        assert all([l > 0 for l in length_list])
        p = self._compute_probs()
        sentences = np.random.choice(np.arange(self._vocab_size), p=p, size=sum(length_list))
        sentences = [str(i) for i in sentences]
        if self.reset_every_sentence:
            self.reset()
        iter_sentences = iter(sentences)
        return [list(itertools.islice(iter_sentences, 0, l)) for l in length_list]

    def _sample_tokens(self) -> List[str]:
        length = self.sentence_length_sampler.sample_length()
        assert length > 0
        p = self._compute_probs()
        sentence = np.random.choice(np.arange(self._vocab_size), p=p, size=length)
        sentence = [str(i) for i in sentence]
        if self.reset_every_sentence:
            self.reset()
        return sentence

    @property
    def vocab_size(self):
        return self._vocab_size

    def _compute_probs(self):
        scores = self.embeddings.dot(self.context_vector[:, None]).flatten()
        p = softmax(scores, self.temperature)
        prob = self.alpha * self.unigram_prob + (1 - self.alpha) * p
        return prob

    def get_token(self):
        idx = np.random.choice(np.arange(self._vocab_size), p=self._compute_probs())
        self.walk()
        yield str(idx)

    def reset(self):
        # the stationary distribution of the random walk is uniform over the unit sphere
        self.context_vector = np.random.normal(size=self.embedding_dim)
