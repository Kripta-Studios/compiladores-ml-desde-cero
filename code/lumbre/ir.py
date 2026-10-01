"""Lumbre: static tensor IR and symbolic reverse-mode differentiation.
Original teaching implementation. No PyTorch/tinygrad computation is called.
All floating tensor operations are lowered by compiler.py to C/CUDA/HIP.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import itertools
import ctypes
import math

Shape = tuple[int, ...]
_counter = itertools.count()


def size(shape: Shape) -> int:
    return math.prod(shape)


def valid_shape(shape) -> Shape:
    s = tuple(int(x) for x in shape)
    if any(x < 0 for x in s):
        raise ValueError(f"negative dimension: {s}")
    if size(s) > 2**40:
        raise ValueError("teaching implementation limits tensor size to 2**40")
    return s


def broadcast(a: Shape, b: Shape) -> Shape:
    a, b = (1,) * max(0, len(b)-len(a)) + a, (1,) * max(0, len(a)-len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x != y and x != 1 and y != 1:
            raise ValueError(f"cannot broadcast {a} with {b}")
        out.append(y if x == 1 else x)
    return tuple(out)


@dataclass(frozen=True, eq=False)
class UOp:
    op: str
    shape: Shape
    src: tuple[UOp, ...] = ()
    arg: object = None
    dtype: str = "float32"
    uid: int = -1

    def __add__(self, other): return binary("add", self, other)
    def __radd__(self, other): return binary("add", other, self)
    def __sub__(self, other): return binary("sub", self, other)
    def __rsub__(self, other): return binary("sub", other, self)
    def __mul__(self, other): return binary("mul", self, other)
    def __rmul__(self, other): return binary("mul", other, self)
    def __truediv__(self, other): return binary("div", self, other)
    def __rtruediv__(self, other): return binary("div", other, self)
    def __neg__(self): return self * -1.0
    def __matmul__(self, other): return matmul(self, other)
    def exp(self): return unary("exp", self)
    def log(self): return unary("log", self)
    def sqrt(self): return unary("sqrt", self)
    def tanh(self): return unary("tanh", self)
    def detach(self): return node("detach", self.shape, (self,), dtype=self.dtype)
    def eq(self, other): return binary("eq", self, other)
    def lt(self, other): return binary("lt", self, other)
    def where(self, yes, no):
        yes, no = tensor(yes), tensor(no)
        s = broadcast(self.shape, broadcast(yes.shape, no.shape))
        return node("where", s, (self, yes, no))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        s = valid_shape(shape)
        if size(s) != size(self.shape):
            raise ValueError(f"reshape changes size: {self.shape} -> {s}")
        return node("reshape", s, (self,), dtype=self.dtype)

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if sorted(axes) != list(range(len(self.shape))):
            raise ValueError(f"invalid permutation {axes}")
        return node("permute", tuple(self.shape[i] for i in axes), (self,), tuple(axes), self.dtype)

    def transpose(self):
        if len(self.shape) < 2:
            raise ValueError("transpose needs at least two axes")
        p = list(range(len(self.shape)))
        p[-1], p[-2] = p[-2], p[-1]
        return self.permute(p)

    def expand(self, shape):
        shape = valid_shape(shape)
        if broadcast(self.shape, shape) != shape:
            raise ValueError("expand target is not the broadcast result")
        return node("expand", shape, (self,), dtype=self.dtype)

    def sum(self, axis=None, keepdim=False): return reduce("sum", self, axis, keepdim)
    def max(self, axis=None, keepdim=False): return reduce("max", self, axis, keepdim)
    def mean(self, axis=None, keepdim=False):
        axes = normalize_axes(axis, len(self.shape))
        n = math.prod(self.shape[i] for i in axes)
        if n == 0: raise ValueError("mean of an empty reduction is undefined here")
        return self.sum(axes, keepdim) / float(n)

    def slice(self, axis, start, stop):
        axis %= len(self.shape)
        if not 0 <= start <= stop <= self.shape[axis]:
            raise ValueError("slice requires 0 <= start <= stop <= dimension")
        shape = list(self.shape)
        shape[axis] = stop-start
        return node("slice", tuple(shape), (self,), (axis, start, stop), self.dtype)

    def pad(self, pads):
        pads = tuple(tuple(int(x) for x in p) for p in pads)
        if len(pads) != len(self.shape) or any(len(p) != 2 or min(p) < 0 for p in pads):
            raise ValueError("pad expects one nonnegative (left,right) pair per axis")
        shape = tuple(d+l+r for d, (l,r) in zip(self.shape, pads))
        return node("pad", shape, (self,), pads, self.dtype)


@lru_cache(maxsize=None)
def node(op, shape, src=(), arg=None, dtype="float32"):
    if dtype not in ("float32", "int32"):
        raise ValueError(f"unsupported dtype {dtype}")
    return UOp(op, valid_shape(shape), tuple(src), arg, dtype, next(_counter))


def param(name, shape, dtype="float32"):
    if not isinstance(name, str) or not name:
        raise ValueError("parameter name must be a nonempty string")
    return node("param", valid_shape(shape), arg=name, dtype=dtype)


def tensor(x) -> UOp:
    if isinstance(x, UOp): return x
    v = ctypes.c_float(float(x)).value
    if not math.isfinite(v):
        raise ValueError("use finite scalar constants; padding/masks need explicit semantics")
    return node("const", (), arg=v.hex())


def unary(op, x):
    if x.dtype != "float32": raise TypeError("floating operation on integer tensor")
    return node(op, x.shape, (x,))


def binary(op, a, b):
    a, b = tensor(a), tensor(b)
    if a.dtype != "float32" or b.dtype != "float32":
        raise TypeError("binary arithmetic only supports float32")
    return node(op, broadcast(a.shape, b.shape), (a, b))


def normalize_axes(axis, ndim):
    if axis is None: return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    if any(not -ndim <= x < ndim for x in axes): raise ValueError("axis out of bounds")
    axes = tuple(sorted(x % ndim for x in axes))
    if len(set(axes)) != len(axes): raise ValueError("duplicate reduction axis")
    return axes


def reduce(op, x, axis, keepdim):
    axes = normalize_axes(axis, len(x.shape))
    if x.dtype != "float32": raise TypeError("reductions require float32")
    if op == "max" and any(x.shape[i] == 0 for i in axes):
        raise ValueError("empty max is not defined by this API")
    if not axes: return x
    s = tuple(1 if i in axes else d for i,d in enumerate(x.shape))
    out = node(op, s, (x,), axes)
    return out if keepdim else out.reshape(tuple(d for i,d in enumerate(s) if i not in axes))


def matmul(a, b):
    if min(len(a.shape), len(b.shape)) < 2: raise ValueError("matmul requires rank >= 2")
    if a.shape[-1] != b.shape[-2]: raise ValueError("matmul K dimensions disagree")
    if a.dtype != "float32" or b.dtype != "float32": raise TypeError("matmul requires float32")
    batch = broadcast(a.shape[:-2], b.shape[:-2])
    return node("matmul", batch + (a.shape[-2], b.shape[-1]), (a,b))


def concat(a, b, axis=-1):
    if len(a.shape) != len(b.shape): raise ValueError("concat ranks disagree")
    axis %= len(a.shape)
    if any(x != y for i,(x,y) in enumerate(zip(a.shape,b.shape)) if i != axis):
        raise ValueError("concat dimensions disagree")
    if a.dtype != b.dtype: raise TypeError("concat dtypes disagree")
    shape = list(a.shape); shape[axis] += b.shape[axis]
    return node("concat", tuple(shape), (a,b), axis, a.dtype)


def gather(table, ids):
    if len(table.shape) != 2 or table.dtype != "float32" or ids.dtype != "int32":
        raise TypeError("gather requires float32[V,D] and int32 ids")
    return node("gather", ids.shape + (table.shape[1],), (table, ids))


def one_hot(ids, classes):
    if ids.dtype != "int32" or classes <= 0: raise ValueError("invalid one_hot")
    return node("onehot", ids.shape + (int(classes),), (ids,), int(classes))


def topo(outputs):
    """Iterative DFS; each input is emitted before each use."""
    done, out = set(), []
    for root in outputs:
        stack = [(root, False)]
        while stack:
            u, exit = stack.pop()
            if u in done: continue
            if exit:
                done.add(u); out.append(u)
            else:
                stack.append((u, True))
                stack.extend((s, False) for s in reversed(u.src) if s not in done)
    return out


def unbroadcast(g, shape):
    aligned = (1,) * (len(g.shape)-len(shape)) + shape
    axes = tuple(i for i,(a,b) in enumerate(zip(aligned,g.shape)) if a == 1 and b != 1)
    if axes: g = g.sum(axes, keepdim=True)
    return g.reshape(shape)


def gradients(loss, wrt):
    """Construct derivative graphs, not numerical finite differences."""
    grad = {loss: tensor(1.0).expand(loss.shape)}
    for u in reversed(topo((loss,))):
        if u not in grad: continue
        g = grad[u]; s = u.src; op = u.op
        gs = []
        if op == "add": gs = [g,g]
        elif op == "sub": gs = [g,-g]
        elif op == "mul": gs = [g*s[1], g*s[0]]
        elif op == "div": gs = [g/s[1], -g*s[0]/(s[1]*s[1])]
        elif op == "exp": gs = [g*u]
        elif op == "log": gs = [g/s[0]]
        elif op == "sqrt": gs = [g/(2.0*u)]
        elif op == "tanh": gs = [g*(1.0-u*u)]
        elif op == "where": gs = [None, s[0].where(g,0.0), s[0].where(0.0,g)]
        elif op == "reshape": gs = [g.reshape(s[0].shape)]
        elif op == "permute": gs = [g.permute(tuple(u.arg.index(i) for i in range(len(u.arg))))]
        elif op == "expand": gs = [unbroadcast(g, s[0].shape)]
        elif op == "sum": gs = [g.expand(s[0].shape)]
        elif op == "max":
            mask = s[0].eq(u.expand(s[0].shape))
            gs = [g.expand(s[0].shape)*mask/mask.sum(u.arg,keepdim=True)]
        elif op == "matmul": gs = [g @ s[1].transpose(), s[0].transpose() @ g]
        elif op == "slice":
            axis,start,stop = u.arg
            pads = tuple((start,s[0].shape[i]-stop) if i == axis else (0,0) for i in range(len(u.shape)))
            gs = [g.pad(pads)]
        elif op == "pad":
            back = g
            for i,(left,right) in enumerate(u.arg): back = back.slice(i,left,left+s[0].shape[i])
            gs = [back]
        elif op == "concat":
            axis = u.arg; split = s[0].shape[axis]
            gs = [g.slice(axis,0,split), g.slice(axis,split,u.shape[axis])]
        elif op == "gather":
            gs = [node("scatter", s[0].shape, (s[1],g)), None]
        elif op == "scatter": gs = [None, gather(g,s[0])]
        elif op == "attention":
            q,k,v = s; T,D = q.shape[-2:]
            p = softmax((q @ k.transpose())/(D**0.5) + node("causal",(T,T)))
            dp = g @ v.transpose()
            ds = p*(dp-(dp*p).sum(-1,keepdim=True))
            gs = [(ds @ k)/(D**0.5), (ds.transpose() @ q)/(D**0.5), p.transpose() @ g]
        elif op in ("param","const","detach","eq","lt","onehot","causal"): continue
        else: raise NotImplementedError(f"gradient for {op}")
        for x,gx in zip(s,gs):
            if gx is not None and x.dtype == "float32":
                gx = unbroadcast(gx,x.shape)
                grad[x] = grad[x]+gx if x in grad else gx
    return [grad.get(x,tensor(0.0).expand(x.shape)) for x in wrt]


def softmax(x, axis=-1):
    m = x.max(axis,keepdim=True).detach()
    e = (x-m).exp()
    return e/e.sum(axis,keepdim=True)


def cross_entropy(logits, labels):
    if labels.shape != logits.shape[:-1]: raise ValueError("label shape mismatch")
    m = logits.max(-1,keepdim=True).detach()
    z = logits-m
    logp = z-z.exp().sum(-1,keepdim=True).log()
    return -(logp*one_hot(labels,logits.shape[-1])).sum(-1).mean()


def online_attention(q,k,v):
    """Causal attention with a linear-memory forward and a dense symbolic VJP.
    This is an educational online-softmax implementation, not FlashAttention's
    complete tiled GPU algorithm. q/k/v must share a static [...,T,D] shape.
    """
    if q.shape != k.shape or q.shape != v.shape or len(q.shape)<2:
        raise ValueError("attention requires equal [...,T,D] shapes")
    if min(q.shape[-2:])<=0 or any(x.dtype!="float32" for x in (q,k,v)):
        raise ValueError("attention requires nonempty float32 rows")
    return node("attention",q.shape,(q,k,v))
