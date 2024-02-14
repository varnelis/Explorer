
from Explorer.tf_idf.types import Term


class Filter:
    @classmethod
    def apply(term: list[Term]) -> list[Term]:
        raise NotImplementedError

class LowerCaseFilter(Filter):
    @classmethod
    def apply(terms: list[Term]) -> list[Term]:
        return [term.lower() for term in terms]