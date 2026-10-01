"""Deterministic byte BPE for teaching, not a production-speed tokenizer.
No pretokenizer, normalization or special tokens. Training is O(tokens * merges).
Fit only on the training split. Any UTF-8 input is encodable through 256 bytes.
"""
from collections import Counter
from dataclasses import dataclass


def replace_pair(ids, pair, new_id):
    out=[]; i=0
    while i<len(ids):
        if i+1<len(ids) and (ids[i],ids[i+1])==pair:
            out.append(new_id); i+=2
        else:
            out.append(ids[i]); i+=1
    return out


@dataclass(frozen=True)
class ByteBPE:
    merges: tuple[tuple[int,int], ...] = ()

    def __post_init__(self):
        for rank,pair in enumerate(self.merges):
            if len(pair)!=2 or any(not isinstance(x,int) or x<0 or x>=256+rank for x in pair):
                raise ValueError('a merge may only reference previously defined tokens')
        if len(set(self.merges))!=len(self.merges):
            raise ValueError('duplicate merge')

    @classmethod
    def fit(cls,text,max_merges=128):
        if max_merges<0: raise ValueError('max_merges must be nonnegative')
        ids=list(text.encode('utf-8')); merges=[]
        for _ in range(max_merges):
            counts=Counter(zip(ids,ids[1:]))
            if not counts: break
            pair,count=min(counts.items(),key=lambda item:(-item[1],item[0]))
            if count<2: break
            new_id=256+len(merges); merges.append(pair)
            ids=replace_pair(ids,pair,new_id)
        return cls(tuple(merges))

    @property
    def vocab(self):
        pieces=[bytes([i]) for i in range(256)]
        for a,b in self.merges: pieces.append(pieces[a]+pieces[b])
        return tuple(pieces)

    def encode(self,text):
        ids=list(text.encode('utf-8'))
        for rank,pair in enumerate(self.merges): ids=replace_pair(ids,pair,256+rank)
        return ids

    def decode(self,ids,errors='strict'):
        pieces=self.vocab; result=[]
        for index in ids:
            index=int(index)
            if index<0 or index>=len(pieces): raise ValueError('token id out of vocabulary')
            result.append(pieces[index])
        return b''.join(result).decode('utf-8',errors=errors)

    def state(self):
        return {'kind':'byte_bpe','merges':[list(pair) for pair in self.merges]}

    @classmethod
    def from_state(cls,state):
        if state.get('kind')!='byte_bpe': raise ValueError('unsupported tokenizer format')
        return cls(tuple(tuple(pair) for pair in state['merges']))
