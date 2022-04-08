import itertools
from typing import List

import numpy as np

from artificial_language.languages.language import Language
from artificial_language.languages.util import sample_index_with_weights

from .random_sampling_language import RandomSamplingLanguage


@Language.register("random_zipf")
class RandomZipfLanguage(RandomSamplingLanguage):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(**kwargs)

        self._vocab_size = vocab_size
        unigram_freq = np.array([int(vocab_size / i) for i in np.arange(1, vocab_size + 1)])
        self.unigram_freq = unigram_freq

    @property
    def vocab_size(self):
        return self._vocab_size

    def _sample_tokens(self) -> List[str]:
        length = self.sentence_length_sampler.sample_length()
        assert length > 0
        indices = sample_index_with_weights(self.unigram_freq, size=length)
        return [str(idx) for idx in indices]

    def batch_generate_sentences(self, num_sentences: int):
        length_list = [self.sentence_length_sampler.sample_length() for _ in range(num_sentences)]
        assert all([l > 0 for l in length_list])
        indices = sample_index_with_weights(self.unigram_freq, size=sum(length_list))
        indices = [str(i) for i in indices]
        iter_indices = iter(indices)
        return [list(itertools.islice(iter_indices, 0, l)) for l in length_list]
