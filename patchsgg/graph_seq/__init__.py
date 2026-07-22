from patchsgg.graph_seq.vocab import GraphVocab, TokenType, VG_VOCAB
#My comment: GraphVocab is a dataclass that defines the mapping between sematic graph concepts-> int token ids
#My comment: TokenType is the obj defining the role of each tokens
#My comment: VG_VOCAB is the default vocabulary instance.
from patchsgg.graph_seq.linearize import (
    graph_to_sequence,
    build_train_pair,
    sequence_to_graph,
    canonical_tuple,
)
#My comment: The purpose is Graph representation <-> Token sequence representation
__all__ = [
    "GraphVocab",
    "TokenType",
    "VG_VOCAB",
    "graph_to_sequence",
    "build_train_pair",
    "sequence_to_graph",
    "canonical_tuple",
]
