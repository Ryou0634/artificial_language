from typing import List

from artificial_language.languages.util import sample_index_with_weights

from .sentence_length_sampler import SentenceLengthSampler


@SentenceLengthSampler.register("empirical_distribution")
class EmpiricalDistributionSampler(SentenceLengthSampler):
    def __init__(self, length_count_list: List[int]):
        self.length_count_list = length_count_list

    def sample_length(self) -> int:
        return sample_index_with_weights(self.length_count_list, size=1)[0]
