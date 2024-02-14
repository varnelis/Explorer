from collections import defaultdict
from dataclasses import dataclass
import math
from uuid import uuid4, UUID

import numpy as np

from Explorer.tf_idf.tokenizer import Tokenizer
from Explorer.tf_idf.types import Similarity, Term, Token, Entry, Frequency, Importance


@dataclass(init=True, frozen=True)
class Index:
    tokenizer: Tokenizer
    # uuid can be 
    data: dict[UUID, Term]
    token_index: dict[Token, Entry]
    # TODO: Add term index, I don't know why I did not do that in the first place, oh well.
    # Then find_all_to_all_match can be implemented to use this, and it will be much quicker

    @classmethod
    def index_data(cls, data: list[Term] | dict[UUID, Term], tokenizer: Tokenizer) -> "Index":
        # don't look at this unless you want your brain melted away. Love xoxo!

        uuid_data = {}
        if isinstance(data, list):
            uuid_data = {uuid4(): d for d in data}
        elif isinstance(data, dict):
            uuid_data = data
        else:
            raise ValueError(f"Cannot index data of type {type(data)}")

        term_index: dict[UUID, dict[Token, Frequency]] = {}
        token_index: dict[Token, Entry] = defaultdict(Entry)

        # Count all the tokens
        for uuid, term in uuid_data.items():
            term_index[uuid] = defaultdict(Frequency)
            for token in tokenizer.apply(term):
                term_index[uuid][token] += 1
                token_index[token].count += 1

        for uuid, term in uuid_data.items():
            term_weigth = cls._term_weight(token_index, term_index[uuid])
            # Calculate token Importance in term, and save reference to the term in token Entry
            for token, frequency in term_index[uuid].items():
                token_importance = math.sqrt(frequency) / (term_weigth * token_index[token].count)
                token_index[token].reference[term] = token_importance

        return Index(
            tokenizer=tokenizer,
            data=data,
            token_index=token_index,
        )
    
    @classmethod
    def _term_weight(cls, tokens_index: dict[Token, Entry], term_tokens_count: dict[Token, Frequency]) -> float:
        # Root of sum of squared inverse frequencies of tokens, a f***ing mouthfull
        return math.sqrt(
            sum(
                [tokens_index[token].count**(-2) * frequency for token, frequency in term_tokens_count.items()]
            )
        )
    
    def outside_term_weight(self, term_tokens_count: dict[Token, Frequency]) -> float:
        # same as _term_weight, but considers that token might not exist in the index
        # Used for search_terms after the index was constructed
        sum: int = 0

        for token, frequency in term_tokens_count.items():
            try:
                sum += self.token_index[token].count**(-2) * frequency
            except:
                sum += 1

        return math.sqrt(sum)

    def get_token_frequency(self, token: Token) -> Frequency:
        try:
            return self.index[token].count
        except KeyError:
            return 0
    
    def find_matches_for(self, term: Term, depth: int = -1) -> dict[Term, Similarity]:
        tokens = self.tokenizer.apply(term)

        if depth > len(tokens) or depth == -1:
            depth = len(tokens)

        # Count tokens and get term_weight
        tokens_index: dict[Token, Frequency] = defaultdict(Frequency)
        for token in tokens:
            tokens_index[token] += 1
        term_weight = self.outside_term_weight(tokens_index)

        # Sort tokens in descending importance
        tokens.sort(key = lambda key: self.get_token_frequency(key))
        matches: dict[Term, Similarity] = defaultdict(Similarity)

        # Only search the most important <depth> tokens.
        # We might have millions of different tokens, which do nothing to the score
        # This might need to be tuned specifically for different data sets
        for token in tokens[:depth]:
            for term_match, indexed_weight in self.token_index[token].reference.items():
                matches[term_match] += indexed_weight / (term_weight * self.token_index[token].count)

        return matches
    
    def find_best_match_for(self, term: Term, count: int = 1, depth: int=-1) -> list[tuple[Term, Similarity]]:
        matches = self.find_matches_for(term, depth)

        # Sort match score descending
        ordered_term_matches = sorted(matches.keys(), key=lambda key: matches[key], reverse=True)

        return [(match, matches[match]) for match in ordered_term_matches[:count]]
    
    def find_all_to_all_match(self, depth: int=-1) -> dict[UUID, dict[UUID, Similarity]]:
        match_matrix = np.array()
        for uuid, term in self.data.items():
            matches = self.find_matches_for(term, depth)
            normalization_const = max(list(matches.values()))
            for k, v in matches.items():
                matches[k] = v / normalization_const
            match_matrix[uuid] = matches
        return match_matrix
