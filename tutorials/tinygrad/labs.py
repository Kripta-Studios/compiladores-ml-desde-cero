"""Original, deterministic lessons for tinygrad commit c3aec477b99d."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from tinygrad import Tensor, TinyJit, Device, Context, dtypes
from tinygrad.nn.optim import SGD

PIN = 'c3aec477b99d9bb87c54d91897cf60acd3f17441'

def check(actual, expected, atol=1e-5, rtol=1e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(actual-expected))) if actual.size else 0.0

def tensor(data, device):
    return Tensor(np.asarray(data, dtype=np.float32), device=device)

def arrays(device):
    a = np.arange(6, dtype=np.float32).reshape(2,3)
    x = tensor(a, device)
    bias = tensor([0.5, -1, 2], device)
    y = (x+bias).relu().sum(axis=1)
    error = check(y.numpy(), np.maximum(a+[0.5,-1,2],0).sum(axis=1))
    transposed = check(x.transpose().numpy(), a.T)
    for n in (1,127,128,129,257):
        v = np.arange(n,dtype=np.float32)/16
        check((tensor(v,device)*2-1).numpy(),v*2-1)
    return {'broadcast_error': error, 'transpose_error': transposed, 'device': str(x.device)}

def gradients(device):
    values = np.array([-2,-0.5,1,3],dtype=np.float32)
    x = tensor(values,device)
    loss = (x*x).mean()
    loss.backward()
    assert x.grad is not None
    analytical = 2*values/len(values)
    error = check(x.grad.numpy(),analytical)
    eps = 1e-3
    numerical=[]
    for i in range(len(values)):
        plus=values.astype(np.float64); minus=plus.copy()
        plus[i]+=eps; minus[i]-=eps
        numerical.append(((plus*plus).mean()-(minus*minus).mean())/(2*eps))
    fd_error = check(x.grad.numpy(),numerical,atol=1e-4)
    return {'loss': float(loss.item()), 'gradient': x.grad.numpy().tolist(),
            'analytical_error': error, 'finite_difference_error': fd_error}

def regression(device, output):
    inputs=np.linspace(-1,1,33,dtype=np.float32).reshape(-1,1)
    targets=3*inputs-0.5
    x,y=tensor(inputs,device),tensor(targets,device)
    w,b=tensor([[0]],device),tensor([0],device)
    optimizer=SGD([w,b],lr=0.2)
    history=[]
    with Context(TRAINING=1):
        for _ in range(100):
            optimizer.zero_grad()
            loss=((x@w+b-y)**2).mean()
            history.append(float(loss.item()))
            loss.backward()
            optimizer.step()
    prediction=(x@w+b).numpy()
    error=check(prediction,targets,atol=2e-4)
    output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'regression_weights.npz',w=w.numpy(),b=b.numpy())
    with np.load(output/'regression_weights.npz') as saved:
        restored=(x@tensor(saved['w'],device)+tensor(saved['b'],device)).numpy()
    restore_error=check(restored,prediction,atol=0,rtol=0)
    assert history[-1]<history[0]*1e-5
    return {'steps':100, 'initial_loss':history[0], 'final_loss':history[-1],
            'w':w.numpy().tolist(), 'b':b.numpy().tolist(), 'max_error':error,
            'weights_reload_error':restore_error, 'history':history}

def attention(device):
    rng=np.random.default_rng(17)
    q,k,v=[rng.normal(size=(1,2,5,4)).astype(np.float32) for _ in range(3)]
    def forward(value):
        tq,tk,tv=[tensor(a,device) for a in (q,k,value)]
        scores=(tq@tk.transpose(-1,-2))/2
        mask=Tensor(np.triu(np.ones((5,5),dtype=bool),1),device=device)
        return mask.where(float('-inf'),scores).softmax(axis=-1)@tv
    scores=q@k.swapaxes(-1,-2)/2
    scores=np.where(np.triu(np.ones((5,5),dtype=bool),1),-np.inf,scores)
    p=np.exp(scores-scores.max(axis=-1,keepdims=True))
    p/=p.sum(axis=-1,keepdims=True)
    got=forward(v).numpy()
    error=check(got,p@v,atol=2e-5)
    changed=v.copy(); changed[:,:,4,:]+=100
    future_error=check(forward(changed).numpy()[:,:,:4,:],got[:,:,:4,:],atol=2e-5)
    return {'shape':[1,2,5,4], 'numpy_error':error, 'future_value_error':future_error}

def jit(device):
    @TinyJit
    def step(x):
        return (x*2+1).relu().realize()
    # Each call gets fresh storage with the same shape and dtype.
    for i in range(5):
        data=np.arange(257,dtype=np.float32)/16+i
        check(step(tensor(data,device).realize()).numpy(),data*2+1)
    x=tensor(np.arange(257,dtype=np.float32),device).realize()
    samples=[]
    for _ in range(30):
        Device[device].synchronize()
        start=time.perf_counter()
        result=step(x)
        Device[device].synchronize()
        samples.append(time.perf_counter()-start)
    check(result.numpy(),np.arange(257,dtype=np.float32)*2+1)
    return {'changed_inputs_checked':5,'samples_seconds':samples,
            'median_seconds':float(np.median(samples)),
            'scope':'warm TinyJit call plus host synchronization; excludes input/output transfers'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['CPU','CUDA'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    results={'tinygrad_commit':PIN,'requested_device':args.device,'lessons':{}}
    for name,fn in [('arrays',arrays),('gradients',gradients),
                    ('regression',lambda d:regression(d,args.output)),
                    ('attention',attention),('jit',jit)]:
        results['lessons'][name]=fn(args.device)
        print(f'{args.device} {name}: PASS',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
