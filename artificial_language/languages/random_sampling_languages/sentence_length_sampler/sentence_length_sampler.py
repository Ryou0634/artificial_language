from abc import ABCMeta, abstractmethod

from artificial_language.utils.registrable import Registrable


class SentenceLengthSampler(Registrable, metaclass=ABCMeta):
    @abstractmethod
    def sample_length(self) -> int:
        raise NotImplementedError
