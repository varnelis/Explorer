from copy import deepcopy
from Explorer.tf_idf.types import Term, Token
from Explorer.tf_idf.filters import Filter

class Tokenizer:
    def __init__(self, separators: list[str], filters: list[Filter], gram: int) -> None:
        self.separators: list[str] = separators
        self.filters: list[Filter] = filters
        self.gram: int = gram
    
    def _separate(self, tokens: list[Token]) -> list[Token]:
        for separator in self.separators:
            new_tokens: list[Token] = []
            for token in tokens:
                new_tokens += token.split(separator)
            tokens = deepcopy(new_tokens)
        return tokens

    def _filter(self, tokens: list[Token]) -> list[Token]:
        for filter in self.filters:
            tokens = filter.apply(tokens)
        return tokens

    def _gramiffy(self, tokens: list[Token]) -> list[Token]:
        if self.gram == 1:
            return tokens

        new_tokens: list[Token] = []
        for token in tokens:
            if len(token) <= self.gram:
                new_tokens.append(token)
            else:
                new_tokens += self._spliter(token)
        return new_tokens

    def _spliter(self, token: Token) -> list[Token]:
        if len(token) == self.gram:
            return [token]
        return [token[:self.gram]] + self._spliter(token[1:])
    
    def apply(self, term: Term) -> list[Token]:
        tokens: list[Token] = [term]

        tokens = self._separate(tokens)
        tokens = self._filter(tokens)
        tokens = self._gramiffy(tokens)

        return tokens
    