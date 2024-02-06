from typing import List
from loops import LoopSpanInSeq
from Bio.Phylo.BaseTree import Clade

class HOR:
    clade_seq: List[Clade]

    def __init__(self, clade_seq):
        self.clade_seq = clade_seq

    def __str__(self):
        return ''.join([clade.id if clade.id is not None else '*' for clade in self.clade_seq])

class HORInSeq:
    hor: HOR
    spans_in_seq: List[LoopSpanInSeq]
    super_hor: any
    sub_hors: List[any]

    def __init__(self, hor, spans_in_seq = []):
        self.hor = hor
        self.spans_in_seq = spans_in_seq

    def add_span(self, span_in_seq):
        self.spans_in_seq.append(span_in_seq)

    def __str__(self):
        return (
            f'{self.loop}' +
            (
                f' in {",".join([str(span) for span in self.spans_in_seq])}'
                    if len(self.spans_in_seq) > 0 else ''
            )
        )

def loop_to_HOR(loop_in_seq, clades):
    hor = HOR([clades[clade_index] for clade_index in loop_in_seq.loop.loop_seq])
    return HORInSeq(hor, spans_in_seq=loop_in_seq.spans_in_seq)

def loops_to_HORs(loops_in_seq, clades):
    return [loop_to_HOR(loop_in_seq, clades) for loop_in_seq in loops_in_seq]
