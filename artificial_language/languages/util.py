from typing import List

import numpy as np


def sample_index_with_weights(weights: List[int], size: int) -> np.ndarray:
    cum_weights = np.cumsum(weights)
    random_value = np.random.randint(cum_weights[-1], size=(size, 1))
    return np.argmax(cum_weights > random_value, axis=1)
