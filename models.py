from enum import Enum


class SummaryType(Enum):
    STORYLINES = "storylines"
    SINGLE_NEWS = "single_news"


class EmbeddingSource(Enum):
    FTMLM = "ft+mlm"
    OPENAI = "openai"
    MLM = "mlm"


class LinkingMethod(Enum):
    DBSCAN = "dbscan"
    BM25 = "bm25"
    NO_LINKER = "no_linker"


class SummaryMethod(Enum):
    OPENAI = "openai"
    BART = "bart"


class Density(Enum):
    SMALL = "small"
    LARGE = "large"

    def next(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) + 1
        if index >= len(members):
            raise StopIteration("end of enumeration reached")
        return members[index]

    def prev(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) - 1
        if index < 0:
            raise StopIteration("beginning of enumeration reached")
        return members[index]
