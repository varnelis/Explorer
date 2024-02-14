from dataclasses import dataclass, field
from uuid import UUID


Term = str
Token = str
Frequency = int
Importance = float
Similarity = float

@dataclass(init=True)
class Entry:
    reference: dict[UUID, Importance] = field(default_factory = lambda: {})
    count: Frequency = 0