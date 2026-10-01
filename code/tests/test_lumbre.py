import math
import numpy as np
import pytest
from lumbre import *
from lumbre.ir import one_hot, broadcast
from lumbre.nn import Decoder,Config,adamw
from lumbre.layout import View
from lumbre.symbolic import var,simplify,interval

rng=np.random.default_rng(23)

@pytest.mark.parametrize("shape_a,shape_b", [((2,3),(3,)),((2,1,4),(1,3,1)),((),(3,)),((0,3),(1,3))])
@pytest.mark.parametrize("fuse",[True,False])
def test_elementwise(shape_a,shape_b,fuse):
    a,b=param("a",shape_a),param("b",shape_b)
    x=rng.normal(size=shape_a).astype("float32");y=rng.normal(size=shape_b).astype("float32")
    out=(a+b)*a-b
    got=Program([out],fuse=fuse).run({"a":x,"b":y})[0]
    np.testing.assert_allclose(got,(x+y)*x-y,rtol=1e-6,atol=1e-6)

@pytest.mark.parametrize("a_shape,b_shape",[((2,3),(3,4)),((2,2,3),(1,3,5)),((1,3,2,4),(2,1,4,3)),((2,0),(0,3))])
def test_matmul(a_shape,b_shape):
    a,b=param("ma",a_shape),param("mb",b_shape)
    x=rng.normal(size=a_shape).astype("float32");y=rng.normal(size=b_shape).astype("float32")
    got=Program([a@b]).run({"ma":x,"mb":y})[0]
    np.testing.assert_allclose(got,x@y,rtol=2e-5,atol=1e-6)

@pytest.mark.parametrize("axis",[0,1,(0,2),None])
def test_reduce(axis):
    a=param("r",(2,3,4));x=rng.normal(size=a.shape).astype("float32")
    p=Program([a.sum(axis),a.max(axis),a.mean(axis)])
    for got,want in zip(p.run({"r":x}),[x.sum(axis=axis),x.max(axis=axis),x.mean(axis=axis)]):
        np.testing.assert_allclose(got,want,rtol=2e-5,atol=1e-6)

@pytest.mark.parametrize("reuse",[True,False])
def test_movement(reuse):
    a=param("v",(2,3));x=np.arange(6,dtype="float32").reshape(2,3)
    z=a.permute(1,0).slice(0,1,3).pad(((1,2),(2,0)))
    y=concat(z,z,-1)
    got=Program([y],reuse=reuse).run({"v":x})[0]
    expected=np.pad(x.T[1:3],((1,2),(2,0)));expected=np.concatenate((expected,expected),axis=-1)
    np.testing.assert_equal(got,expected)


def test_gather():
    a=param("table",(4,3));ids=param("ids",(2,2),"int32")
    x=np.arange(12,dtype="float32").reshape(4,3);i=np.array([[1,1],[0,3]],dtype="int32")
    y=gather(a,ids);g=gradients(y.sum(),[a])[0]
    got,grad=Program([y,g]).run({"table":x,"ids":i})
    np.testing.assert_equal(got,x[i]);np.testing.assert_equal(grad,np.array([[1]*3,[2]*3,[0]*3,[1]*3]))

@pytest.mark.parametrize("op",["exp","log","sqrt","tanh"])
def test_unary_gradient(op):
    a=param("u",(3,));x=np.array([0.5,0.9,1.4],dtype="float32")
    y=getattr(a,op)().sum();g=gradients(y,[a])[0]
    p=Program([y,g]);base,analytic=p.run({"u":x});numeric=np.empty_like(x)
    for i in range(3):
        xp=x.copy();xm=x.copy();xp[i]+=1e-3;xm[i]-=1e-3
        numeric[i]=(p.run({"u":xp},read=[y])[0]-p.run({"u":xm},read=[y])[0])/(2e-3)
    np.testing.assert_allclose(analytic,numeric,rtol=2e-3,atol=2e-3)


def test_composite_gradient():
    a=param("g_a",(2,3));b=param("g_b",(3,4));bias=param("g_bias",(4,))
    y=(((a@b+bias).tanh())**1 if False else (a@b+bias).tanh()).sum()
    gs=gradients(y,[a,b,bias]);p=Program([y,*gs])
    feed={u.arg:(rng.normal(size=u.shape)*0.2).astype("float32") for u in (a,b,bias)}
    vals=p.run(feed)
    for u,analytic in zip((a,b,bias),vals[1:]):
        x=feed[u.arg];numeric=np.empty_like(x)
        for idx in np.ndindex(x.shape):
            plus=x.copy();minus=x.copy();plus[idx]+=1e-3;minus[idx]-=1e-3
            f1=p.run({**feed,u.arg:plus},read=[y])[0]
            f0=p.run({**feed,u.arg:minus},read=[y])[0]
            numeric[idx]=(f1-f0)/(2e-3)
        np.testing.assert_allclose(analytic,numeric,rtol=3e-3,atol=3e-3)


def test_softmax_and_cross_entropy():
    a=param("logits",(2,3));t=param("target",(2,),"int32")
    x=np.array([[1000,1001,999],[-1000,-1002,-999]],dtype="float32");labels=np.array([1,2],dtype="int32")
    loss=cross_entropy(a,t);g=gradients(loss,[a])[0]
    got,grad=Program([loss,g]).run({"logits":x,"target":labels})
    z=x-x.max(-1,keepdims=True);probs=np.exp(z)/np.exp(z).sum(-1,keepdims=True)
    target=np.eye(3)[labels]
    np.testing.assert_allclose(got,-np.log(probs[np.arange(2),labels]).mean(),rtol=1e-5)
    np.testing.assert_allclose(grad,(probs-target)/2,rtol=1e-5,atol=1e-6)


def test_arena_reuse_and_fusion():
    a=param("chain",(101,));u=a
    for _ in range(12): u=(u*0.9+0.1).tanh()
    x=rng.normal(size=a.shape).astype("float32")
    p=Program([u],reuse=True,fuse=True);q=Program([u],reuse=False,fuse=False)
    np.testing.assert_allclose(p.run({"chain":x})[0],q.run({"chain":x})[0],rtol=1e-5,atol=1e-6)
    assert len(p.kernels)<len(q.kernels);assert p.bytes<q.bytes


def test_views():
    v=View.contiguous((2,3)).permute((1,0))
    assert v.address((2,1))==5
    assert v.flip(0).address((0,1))==5
    with pytest.raises(ValueError): v.contiguous_reshape((6,))
    assert View.contiguous((1,3)).expand((4,3)).address((3,2))==2


def test_symbolic():
    i=var("i");e=(i*1+0)*3
    assert simplify(e)==i*3
    assert interval((i+3)*2,{"i":(0,7)})==(6,20)
    for n in range(100): assert e.evaluate({"i":n})==simplify(e).evaluate({"i":n})


def test_shapes_and_errors():
    with pytest.raises(ValueError): broadcast((2,3),(4,3))
    with pytest.raises(ValueError): param("bad",(-1,))
    with pytest.raises(ValueError): param("s",(2,3)).reshape(5)
    with pytest.raises(ValueError): Program([param("unset",(1,))+1]).run()


def test_decoder_causality():
    m=Decoder(Config(vocab=7,batch=1,context=4,dim=8,heads=2,kv_heads=1,layers=1,hidden=16))
    p=Program([m.logits])
    for name,value in m.initial.items():
        if name in p.params:p.set(name,value)
    x=np.array([[1,2,3,4]],dtype="int32");y=x.copy();y[0,3]=6
    a=p.run({"tokens":x})[0];b=p.run({"tokens":y})[0]
    np.testing.assert_allclose(a[:,:3],b[:,:3],rtol=1e-6,atol=1e-6)


def test_adam_learning_and_assignment():
    w=param("learn",());loss=(w-3)*(w-3)
    g=gradients(loss,[w])[0];new=w-0.1*g
    p=Program([loss,new]);p.set("learn",np.array(0,dtype="float32"))
    first=float(p.run(read=[loss])[0])
    for _ in range(25): p.run(read=[]);p.assign({"learn":new})
    last=float(p.run(read=[loss])[0]);assert last<first*1e-4

@pytest.mark.parametrize('shape',[(3,5,7),(8,16,64),(1,1,0),(9,17,65)])
def test_blocked_gemm(shape):
    M,N,K=shape; rng=np.random.default_rng(32)
    a=param('a',(2,M,K));b=param('b',(1,K,N));out=a@b
    av=rng.normal(size=a.shape).astype('float32');bv=rng.normal(size=b.shape).astype('float32')
    with Program([out],gemm='blocked') as p:
        np.testing.assert_allclose(p.run({'a':av,'b':bv})[0],av@bv,atol=2e-5,rtol=2e-5)

@pytest.mark.parametrize('T,D',[(1,2),(5,4),(17,7)])
def test_online_attention_and_gradients(T,D):
    from lumbre import online_attention,softmax
    from lumbre.ir import node
    rng=np.random.default_rng(63)
    q,k,v=[param(n,(1,2,T,D)) for n in ('q','k','v')]
    online=online_attention(q,k,v)
    dense=softmax((q@k.transpose())/(D**0.5)+node('causal',(T,T)))@v
    g1=gradients((online*online).sum(),[q,k,v]);g2=gradients((dense*dense).sum(),[q,k,v])
    values={n:rng.normal(size=q.shape).astype('float32') for n in ('q','k','v')}
    with Program([online,dense,*g1,*g2],gemm='blocked') as p:
        results=p.run(values)
    np.testing.assert_allclose(results[0],results[1],atol=2e-5,rtol=2e-5)
    for i in range(3): np.testing.assert_allclose(results[2+i],results[5+i],atol=4e-5,rtol=4e-5)


def test_assign_rejects_aliasing_parameter_outputs():
    a=param('a',(2,)); b=param('b',(2,))
    with Program([a,b]) as p:
        p.run({'a':np.array([1,2]),'b':np.array([3,4])})
        with pytest.raises(ValueError): p.assign({'a':b,'b':a})
