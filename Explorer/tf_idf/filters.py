
from Explorer.tf_idf.types import Term
from spellchecker import SpellChecker

class Filter:
    @classmethod
    def apply(cls, term: list[Term]) -> list[Term]:
        raise NotImplementedError

class LowerCaseFilter(Filter):
    @classmethod
    def apply(cls, terms: list[Term]) -> list[Term]:
        return [term.lower() for term in terms]
    
class SpellingFilterWithReplacement(Filter):

    speller = SpellChecker()
    history = {}

    @classmethod
    def apply(cls, terms: list[Term]) -> list[Term]:
        new_terms = []
        for t in terms:
            if t not in cls.history:
                new_t = cls.speller.correction(t)
                cls.history[t] = new_t if new_t is not None else t
            new_t = cls.history[t]
            if new_t is None:
                continue
            new_terms.append(cls.history[t])
        return new_terms
    
class SpellingFilterWithoutReplacement(Filter):

    speller = SpellChecker()
    history = {}

    @classmethod
    def apply(cls, terms: list[Term]) -> list[Term]:
        new_terms = []
        for t in terms:
            if t not in cls.history:
                new_t = cls.speller.correction(t)
                cls.history[t] = new_t
            new_t = cls.history[t]
            if new_t is None:
                continue
            new_terms.append(cls.history[t])
        return new_terms
