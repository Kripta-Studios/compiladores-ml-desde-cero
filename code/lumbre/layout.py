"""Zero-copy layout model for the movement-operations laboratory.
This is a separate oracle/lab module, not a claim that Program accepts arbitrary strides.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class View:
    shape: tuple[int,...]
    strides: tuple[int,...]
    offset: int=0

    def __post_init__(self):
        if len(self.shape)!=len(self.strides) or any(d<0 for d in self.shape): raise ValueError("invalid view")

    @staticmethod
    def contiguous(shape):
        shape=tuple(shape)
        return View(shape,tuple(math.prod(shape[i+1:]) for i in range(len(shape))))

    def address(self,index):
        if len(index)!=len(self.shape) or any(not 0<=i<d for i,d in zip(index,self.shape)): raise IndexError(index)
        return self.offset+sum(i*s for i,s in zip(index,self.strides))

    def permute(self,axes):
        if sorted(axes)!=list(range(len(self.shape))): raise ValueError("invalid permutation")
        return View(tuple(self.shape[a] for a in axes),tuple(self.strides[a] for a in axes),self.offset)

    def expand(self,shape):
        shape=tuple(shape)
        if len(shape)<len(self.shape): raise ValueError("cannot remove axes")
        old=(1,)*(len(shape)-len(self.shape))+self.shape
        strides=(0,)*(len(shape)-len(self.shape))+self.strides
        if any(a!=b and a!=1 for a,b in zip(old,shape)): raise ValueError("invalid broadcast")
        return View(shape,tuple(0 if a==1 else s for a,s in zip(old,strides)),self.offset)

    def flip(self,axis):
        axis%=len(self.shape);s=list(self.strides)
        off=self.offset+(self.shape[axis]-1)*s[axis] if self.shape[axis] else self.offset
        s[axis]=-s[axis]
        return View(self.shape,tuple(s),off)

    def contiguous_reshape(self,shape):
        shape=tuple(shape)
        if math.prod(shape)!=math.prod(self.shape): raise ValueError("size changes")
        expected=1
        for d,s in reversed(list(zip(self.shape,self.strides))):
            if d>1 and s!=expected: raise ValueError("noncontiguous reshape needs general indexing or materialization")
            expected*=d
        v=View.contiguous(shape)
        return View(v.shape,v.strides,self.offset)
