import random
from typing import List

from scipy.stats import truncnorm

from .sentence_length_sampler import SentenceLengthSampler


@SentenceLengthSampler.register("trunc_norm")
class TruncNormSampler(SentenceLengthSampler):
    def __init__(self, trunc_norm_params: List[float]):
        self.trunc_norm_params = trunc_norm_params

    def sample_length(self) -> int:
        length = truncnorm.rvs(*self.trunc_norm_params)
        if random.random() < 0.5:
            length += 1
        return int(length)
