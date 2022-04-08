from abc import ABCMeta, abstractmethod
from typing import List

from artificial_language.utils.registrable import Registrable


class Language(Registrable, metaclass=ABCMeta):
    def batch_generate_sentences(self, num_sentences: int) -> List[List[str]]:
        return [self.generate_sentence() for _ in range(num_sentences)]

    @abstractmethod
    def generate_sentence(self) -> List[str]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def get_vocabulary(self) -> List[str]:
        raise NotImplementedError
