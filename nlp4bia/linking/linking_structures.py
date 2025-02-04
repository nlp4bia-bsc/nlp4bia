from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class Candidate:
    code: str
    score: float
    term: str

    def __repr__(self):
        return f"Candidate(code={self.code}, score={self.score:.4f}, term={self.term})"

@dataclass
class TermResult:
    term: str
    candidates: List[Candidate]

    def __repr__(self):
        return f"TermResult(term={self.term}, candidates={self.candidates})"



# TODO: Sería bueno tener unaa estructura de datos de este tipo que carge el gold standard para
# utilizarlo directamente para evaluar los modelos con un método "evaluation()" dentro de cada una
# de las clases de linking y rerankeo.

@dataclass
class Mention:
    span: str
    mention_class: str
    code: str
    source: str
    filename_id: str
    span_ini: int
    span_end: int

    @staticmethod
    def from_row(row) -> 'Mention':
        span_ini, span_end = map(int, row['filenameid'].split('#')[1:3])
        return Mention(
            span=row['span'],
            mention_class=row['mention_class'],
            code=row['code'],
            source=row['source'],
            filename_id=row['filenameid'].split('#')[0],
            span_ini=span_ini,
            span_end=span_end
        )
    
    @staticmethod
    def from_row_test(row) -> 'Mention':
        span_ini, span_end = map(int, row['filenameid'].split('#')[1:3])
        return Mention(
            span=row['span'],
            mention_class=row['mention_class'],
            code=None,
            source=row['source'],
            filename_id=row['filenameid'].split('#')[0],
            span_ini=span_ini,
            span_end=span_end
        )

class GoldStandardLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.mentions = self._load_mentions()

    def _load_mentions(self) -> List[Mention]:
        self.df = pd.read_csv(self.filepath, sep='\t', dtype=str)
        return [Mention.from_row(row) for _, row in self.df.iterrows()]

    def filter_mentions(self, sources: Optional[List[str]] = None, mention_classes: Optional[List[str]] = None) -> List[Mention]:
        filtered_mentions = self.mentions
        
        if sources:
            filtered_mentions = [mention for mention in filtered_mentions if mention.source in sources]
        
        if mention_classes:
            filtered_mentions = [mention for mention in filtered_mentions if mention.mention_class in mention_classes]
        
        return filtered_mentions

    def __repr__(self):
        return f"GoldStandardLoader(mentions={self.mentions})"

class TestLoader(GoldStandardLoader):
    def __init__(self, filepath: str):
        super().__init__(filepath)

    def _load_mentions(self) -> List[Mention]:
        self.df = pd.read_csv(self.filepath, sep='\t', dtype=str)
        return [Mention.from_row_test(row) for _, row in self.df.iterrows()]
    
    def __repr__(self):
        return f"TestLoader(mentions={self.mentions})"