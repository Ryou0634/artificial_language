import random
from typing import List

from artificial_language.languages.language import Language

from .random_sampling_language import RandomSamplingLanguage


@Language.register("random_uniform")
class RandomUniformLanguage(RandomSamplingLanguage):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(**kwargs)

        self._vocab_size = vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size

    def _sample_tokens(self) -> List[str]:
        length = self.sentence_length_sampler.sample_length()
        assert length > 0
        indices = [random.randint(0, self.vocab_size - 1) for _ in range(length)]
        return [str(idx) for idx in indices]
