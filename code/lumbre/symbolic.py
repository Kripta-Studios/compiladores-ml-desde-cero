"""Exact integer expressions for index arithmetic; NOT float rewrite rules."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True)
class Expr:
    op: str
    src: tuple[Expr,...]=()
    arg: int|str|None=None
    def __add__(self,x): return make("add",self,expr(x))
    def __radd__(self,x): return expr(x)+self
    def __mul__(self,x): return make("mul",self,expr(x))
    def __rmul__(self,x): return expr(x)*self
    def __floordiv__(self,x): return make("div",self,expr(x))
    def __mod__(self,x): return make("mod",self,expr(x))
    def evaluate(self,values):
        if self.op=="const": return self.arg
        if self.op=="var": return int(values[self.arg])
        a,b=(x.evaluate(values) for x in self.src)
        if self.op=="add": return a+b
        if self.op=="mul": return a*b
        if b<=0: raise ValueError("index division requires a positive denominator")
        return a//b if self.op=="div" else a%b

@lru_cache(None)
def var(name): return Expr("var",arg=name)
@lru_cache(None)
def const(n): return Expr("const",arg=int(n))
def expr(x): return x if isinstance(x,Expr) else const(x)

@lru_cache(None)
def make(op,a,b):
    if op not in ("add","mul","div","mod"): raise ValueError(op)
    return Expr(op,(a,b))


def simplify(root):
    memo={}
    def walk(u):
        if u in memo: return memo[u]
        if not u.src: out=u
        else:
            a,b=map(walk,u.src)
            out=make(u.op,a,b)
            ac=a.op=="const"; bc=b.op=="const"
            if ac and bc:
                out=const(out.evaluate({}))
            elif u.op=="add" and bc and b.arg==0: out=a
            elif u.op=="add" and ac and a.arg==0: out=b
            elif u.op=="mul" and bc and b.arg==1: out=a
            elif u.op=="mul" and ac and a.arg==1: out=b
            elif u.op=="mul" and ((ac and a.arg==0) or (bc and b.arg==0)): out=const(0)
            elif u.op=="div" and bc and b.arg==1: out=a
            elif u.op=="mod" and bc and b.arg==1: out=const(0)
        memo[u]=out; return out
    return walk(root)


def interval(u,bounds):
    if u.op=="const": return u.arg,u.arg
    if u.op=="var": return bounds[u.arg]
    (lo,hi),(l2,h2)=[interval(s,bounds) for s in u.src]
    if u.op=="add": return lo+l2,hi+h2
    if u.op=="mul":
        vals=(lo*l2,lo*h2,hi*l2,hi*h2);return min(vals),max(vals)
    if l2!=h2 or l2<=0: raise ValueError("only positive constant divisor supported")
    if u.op=="div": return lo//l2,hi//l2
    return (lo%l2,hi%l2) if lo//l2==hi//l2 else (0,l2-1)


def render_c(u):
    if u.op=="const": return str(u.arg)
    if u.op=="var":
        if not str(u.arg).isidentifier() or not str(u.arg).isascii(): raise ValueError("unsafe variable name")
        return u.arg
    a,b=map(render_c,u.src)
    # Contract: numerator nonnegative, divisor positive; C / then matches floor.
    return f"({a}{ {'add':'+','mul':'*','div':'/','mod':'%'}[u.op] }{b})"
