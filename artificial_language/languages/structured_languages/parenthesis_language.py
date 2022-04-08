import copy
import itertools
import random
from typing import List, Tuple

from artificial_language.languages import Language


def build_flat_parenthesis(token_ids: List[Tuple[str, str]]) -> List[str]:

    token_pairs_stack = copy.deepcopy(token_ids)

    to_close_stack = []

    random.shuffle(token_pairs_stack)

    sentence = []

    while len(token_pairs_stack) > 0:
        prob_to_choice_open = len(token_pairs_stack) / (len(to_close_stack) + len(token_pairs_stack))
        if len(to_close_stack) == 0 or prob_to_choice_open > random.random():
            open_token, close_token = token_pairs_stack.pop()
            sentence.append(open_token)
            to_close_stack.append(close_token)
            random.shuffle(to_close_stack)
        else:
            close_token = to_close_stack.pop()
            sentence.append(close_token)

    while len(to_close_stack) > 0:
        close_token = to_close_stack.pop()
        sentence.append(close_token)

    return sentence


def build_nesting_parenthesis(
    token_pairs: List[Tuple[str, str]],
    flat: bool = False,
    open_prob: float = 0.4,
) -> List[str]:
    opened_stack = []
    sentence = []

    token_pairs_stack = copy.deepcopy(token_pairs)

    while len(token_pairs_stack) > 0:
        if len(opened_stack) == 0 or random.random() < open_prob:
            h, d = token_pairs_stack.pop()
            opened_stack.append(d)
            sentence.append(h)
        else:
            if flat:
                random.shuffle(opened_stack)
            d = opened_stack.pop()
            sentence.append(d)

    while len(opened_stack) > 0:
        d = opened_stack.pop()
        sentence.append(d)
    return sentence


@Language.register("parenthesis")
class ParenthesisLanguage(Language):

    VALID_TYPE = {"flat", "nesting"}

    def __init__(self, parenthesis_type: str, language: Language, same_open_and_close_token: bool = False, **kwargs):
        super().__init__(**kwargs)
        if parenthesis_type not in self.VALID_TYPE:
            raise ValueError(parenthesis_type)
        self.parenthesis_type = parenthesis_type
        self.same_open_and_close_token = same_open_and_close_token

        self.language = language

    @property
    def vocab_size(self):
        if self.same_open_and_close_token:
            return self.language.vocab_size
        else:
            return self.language.vocab_size * 2

    def batch_generate_sentences(self, num_sentences: int = 64):
        sentences = self.language.batch_generate_sentences(num_sentences)
        return [self._tokens_to_parenthesis(s) for s in sentences]

    def _tokens_to_parenthesis(self, tokens: List[str]):

        half_length = len(tokens) // 2
        if random.random() < 0.5:
            half_length += 1
        if half_length == 0:
            half_length = 1

        tokens = tokens[:half_length]

        token_pairs = [self._get_parenthesis(t) for t in tokens]

        if self.parenthesis_type == "flat":
            return build_flat_parenthesis(token_pairs)
        elif self.parenthesis_type == "nesting":
            return build_nesting_parenthesis(token_pairs)
        elif self.parenthesis_type not in self.VALID_TYPE:
            raise ValueError(self.parenthesis_type)

    def generate_sentence(self) -> List[str]:
        tokens = self.language.generate_sentence()
        return self._tokens_to_parenthesis(tokens)

    def _get_parenthesis(self, token: str) -> Tuple[str, str]:
        if self.same_open_and_close_token:
            return f"{token}", f"{token}"
        else:
            return f"({token}", f"{token})"

    def get_vocabulary(self):
        token_pairs = [self._get_parenthesis(t) for t in self.language.get_vocabulary()]
        return list(itertools.chain.from_iterable(token_pairs))


def calculate_dependency_distance(sentence: List[str]) -> List[int]:
    from collections import defaultdict

    position_dict = defaultdict(list)
    distance_list = []
    for i, t in enumerate(sentence):
        if t.startswith("("):
            token_id = int(t[1:])
            position_dict[token_id].append(i)
        elif t.endswith(")"):
            token_id = int(t[:-1])
            start_position = position_dict[token_id].pop()
            distance = i - start_position
            distance_list.append(distance)
        else:
            raise ValueError(t)
    return distance_list
