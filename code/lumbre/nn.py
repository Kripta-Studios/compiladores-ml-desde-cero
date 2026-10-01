"""Decoder-only Transformer: RMSNorm, RoPE, GQA, SwiGLU, tied embeddings."""
from dataclasses import dataclass
import math
import numpy as np
from .ir import param, tensor, concat, gather, softmax, cross_entropy, gradients, online_attention


@dataclass(frozen=True)
class Config:
    vocab: int = 256
    batch: int = 2
    context: int = 16
    dim: int = 32
    heads: int = 4
    kv_heads: int = 2
    layers: int = 2
    hidden: int = 64
    rope_base: float = 10000.0
    eps: float = 1e-5
    online: bool = False

    def __post_init__(self):
        if min(self.vocab,self.batch,self.context,self.dim,self.heads,self.kv_heads,self.layers,self.hidden)<=0:
            raise ValueError("all dimensions must be positive")
        if self.dim%self.heads or self.heads%self.kv_heads or (self.dim//self.heads)%2:
            raise ValueError("dim/heads must be even; heads must be divisible by kv_heads")


class Decoder:
    def __init__(self,config=Config(),seed=7):
        self.c=config; self.weights={}; self.initial={}
        self.rng=np.random.default_rng(seed)
        c=config; B,T,D,H,HK=c.batch,c.context,c.dim,c.heads,c.kv_heads
        dh=D//H
        self.tokens=param("tokens",(B,T),"int32")
        self.labels=param("labels",(B,T),"int32")
        inv=c.rope_base**(-np.arange(0,dh,2,dtype=np.float32)/dh)
        angle=np.arange(T,dtype=np.float32)[:,None]*inv[None,:]
        self.cos=self.fixed("rope_cos",np.cos(angle).reshape(1,1,T,dh//2))
        self.sin=self.fixed("rope_sin",np.sin(angle).reshape(1,1,T,dh//2))
        mask=np.where(np.arange(T)[None,:] <= np.arange(T)[:,None],0.0,-1e9).astype("float32")
        self.mask=self.fixed("causal_mask",mask.reshape(1,1,T,T))
        embed=self.weight("embedding",(c.vocab,D),scale=0.02)
        x=gather(embed,self.tokens)
        for i in range(c.layers):
            tag=f"layer{i}"
            z=self.norm(x,self.weight(tag+".attn_norm",(D,),ones=True))
            q=z@self.weight(tag+".wq",(D,H*dh))
            k=z@self.weight(tag+".wk",(D,HK*dh))
            v=z@self.weight(tag+".wv",(D,HK*dh))
            q=q.reshape(B,T,H,dh).permute(0,2,1,3)
            k=k.reshape(B,T,HK,dh).permute(0,2,1,3)
            v=v.reshape(B,T,HK,dh).permute(0,2,1,3)
            q,k=self.rope(q),self.rope(k)
            if HK!=H:
                k=k.reshape(B,HK,1,T,dh).expand((B,HK,H//HK,T,dh)).reshape(B,H,T,dh)
                v=v.reshape(B,HK,1,T,dh).expand((B,HK,H//HK,T,dh)).reshape(B,H,T,dh)
            if c.online:
                attended=online_attention(q,k,v)
            else:
                a=softmax((q@k.transpose())/math.sqrt(dh)+self.mask)
                attended=a@v
            y=attended.permute(0,2,1,3).reshape(B,T,D)
            x=x+y@self.weight(tag+".wo",(D,D),scale=1/math.sqrt(D*c.layers))
            z=self.norm(x,self.weight(tag+".ffn_norm",(D,),ones=True))
            gate=z@self.weight(tag+".gate",(D,c.hidden))
            up=z@self.weight(tag+".up",(D,c.hidden))
            silu=gate/(1.0+(-gate).exp())
            x=x+(silu*up)@self.weight(tag+".down",(c.hidden,D),scale=1/math.sqrt(c.hidden*c.layers))
        x=self.norm(x,self.weight("final_norm",(D,),ones=True))
        self.logits=x@embed.transpose()
        self.loss=cross_entropy(self.logits,self.labels)

    def weight(self,name,shape,scale=None,ones=False):
        u=param(name,shape); self.weights[name]=u
        if ones: a=np.ones(shape,dtype=np.float32)
        else: a=(self.rng.standard_normal(shape)*(scale or 1/math.sqrt(shape[0]))).astype(np.float32)
        self.initial[name]=a
        return u

    def fixed(self,name,value):
        value=np.asarray(value,dtype=np.float32); self.initial[name]=value
        return param(name,value.shape)

    def norm(self,x,w): return x/(x*x).mean(-1,keepdim=True).__add__(self.c.eps).sqrt()*w

    def rope(self,x):
        half=x.shape[-1]//2
        a=x.slice(-1,0,half); b=x.slice(-1,half,2*half)
        return concat(a*self.cos-b*self.sin,a*self.sin+b*self.cos,-1)

    def parameter_count(self): return sum(math.prod(w.shape) for w in self.weights.values())


def adamw(model,weight_decay=0.01,clip_norm=1.0,beta1=0.9,beta2=0.999,eps=1e-8):
    """Return compiled update outputs, assignment map, and initial optimizer state."""
    weights=list(model.weights.values())
    gs=gradients(model.loss,weights)
    norm2=sum((g*g).sum() for g in gs)
    norm=(norm2+1e-12).sqrt()
    ratio=clip_norm/(norm+1e-6)
    scale=ratio.lt(1.0).where(ratio,1.0)
    lr=param("learning_rate",())
    bc1=param("adam_bc1",()); bc2=param("adam_bc2",())
    updates={}; initial={}
    for (name,w),g in zip(model.weights.items(),gs):
        g=g*scale
        m=param("m."+name,w.shape); v=param("v."+name,w.shape)
        m1=beta1*m+(1-beta1)*g
        v1=beta2*v+(1-beta2)*g*g
        new=w*(1-lr*weight_decay)-lr*(m1/bc1)/((v1/bc2).sqrt()+eps)
        updates[name]=new; updates[m.arg]=m1; updates[v.arg]=v1
        initial[m.arg]=np.zeros(w.shape,dtype=np.float32)
        initial[v.arg]=np.zeros(w.shape,dtype=np.float32)
    return [model.loss,norm,*updates.values()],updates,initial
