from abc import abstractmethod
from typing import List

from artificial_language.languages import Language

from .sentence_length_sampler import SentenceLengthSampler


class RandomSamplingLanguage(Language):
    def __init__(self, sentence_length_sampler: SentenceLengthSampler):
        self.sentence_length_sampler = sentence_length_sampler

    @abstractmethod
    def _sample_tokens(self) -> List[str]:
        raise NotImplementedError

    def generate_sentence(self) -> List[str]:
        return self._sample_tokens()

    def get_vocabulary(self):
        return [str(t) for t in range(self.vocab_size)]
