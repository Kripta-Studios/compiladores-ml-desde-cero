"""Correctness first; resident microbenchmarks, not claims of SOTA.
Compile, transfers and result copies excluded from the resident hot timing.
"""
import argparse,json,platform,time,statistics
from pathlib import Path
import numpy as np
from lumbre import param,Program,online_attention,softmax
from lumbre.ir import node


def measure(p,repeat):
    for _ in range(3): p.run(read=[])
    ts=[]
    for _ in range(repeat):
        t=time.perf_counter_ns();p.run(read=[]);ts.append((time.perf_counter_ns()-t)*1e-9)
    return {'median_seconds':statistics.median(ts),'min_seconds':min(ts),'max_seconds':max(ts),'samples_seconds':ts}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,default=20)
    ap.add_argument('--backend',choices=['cpu','cuda','hip'],default='cpu')
    ap.add_argument('--output',type=Path,default=Path('reports/kernels.json'));args=ap.parse_args()
    if args.repeat<3: raise ValueError('use at least 3 samples')
    rng=np.random.default_rng(17);records=[]
    for M,N,K in [(31,47,65),(128,128,128),(256,256,256)]:
        a=param('a',(M,K));b=param('b',(K,N));out=a@b
        data={'a':rng.normal(size=a.shape).astype('float32'),'b':rng.normal(size=b.shape).astype('float32')}
        reference=data['a']@data['b']
        for strategy in ['reference','blocked']:
            with Program([out],backend=args.backend,gemm=strategy) as p:
                got=p.run(data)[0];np.testing.assert_allclose(got,reference,atol=1e-4,rtol=1e-4)
                timing=measure(p,args.repeat)
                records.append({'op':'gemm','shape':[M,N,K],'strategy':strategy,'max_abs_error':float(np.max(np.abs(got-reference))),**timing,'gflops':2*M*N*K/timing['median_seconds']/1e9,'compiler':p.report()})
    for T in [16,64,128]:
        D=32;q,k,v=[param(n,(1,2,T,D)) for n in ('q','k','v')]
        dense=softmax((q@k.transpose())/(D**0.5)+node('causal',(T,T)))@v
        online=online_attention(q,k,v)
        data={n:rng.normal(size=q.shape).astype('float32') for n in ('q','k','v')}
        with Program([dense],backend=args.backend,gemm='blocked') as p: reference=p.run(data)[0]
        for name,out in [('dense',dense),('online',online)]:
            with Program([out],backend=args.backend,gemm='blocked') as p:
                got=p.run(data)[0];np.testing.assert_allclose(got,reference,atol=2e-5,rtol=2e-5)
                records.append({'op':'attention','shape':[1,2,T,D],'strategy':name,'max_abs_error':float(np.max(np.abs(got-reference))),**measure(p,args.repeat),'compiler':p.report()})
    result={'system':platform.platform(),'backend':args.backend,'records':records,'scope':f'local {args.backend} resident microbenchmark; no cross-system comparison'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    for r in records:print(r['op'],r['shape'],r['strategy'],r['median_seconds'],r['compiler']['arena_bytes'])

if __name__=='__main__':main()
