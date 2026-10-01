import numpy as np
import pytest
from lumbre import param,Program,gradients
from lumbre.nnops import conv2d

@pytest.mark.parametrize('groups,stride,pad,dilation',[(1,1,0,1),(2,2,1,1),(2,1,1,2)])
def test_composite_conv_and_gradient(groups,stride,pad,dilation):
    rng=np.random.default_rng(93)
    x=param('cx',(1,2,4,5));w=param('cw',(4,2//groups,2,2));b=param('cb',(4,))
    out=conv2d(x,w,b,stride=stride,padding=pad,dilation=dilation,groups=groups)
    loss=(out*out).mean();gx,gw,gb=gradients(loss,[x,w,b])
    feed={u.arg:(rng.normal(size=u.shape)*.2).astype('float32') for u in (x,w,b)}
    with Program([out,loss,gx,gw,gb],gemm='blocked') as p:
        result=p.run(feed);want=np.empty(out.shape,dtype='float64')
        for batch,o,y,z in np.ndindex(out.shape):
            acc=float(feed['cb'][o]);group=o//(4//groups)
            for ci,ky,kz in np.ndindex(w.shape[1:]):
                iy=y*stride-pad+ky*dilation;iz=z*stride-pad+kz*dilation
                if 0<=iy<4 and 0<=iz<5:
                    acc+=float(feed['cx'][batch,group*(2//groups)+ci,iy,iz])*float(feed['cw'][o,ci,ky,kz])
            want[batch,o,y,z]=acc
        np.testing.assert_allclose(result[0],want,rtol=2e-5,atol=2e-5)
        for u,analytic in zip((x,w,b),result[2:]):
            index=tuple(0 for _ in u.shape); plus=feed[u.arg].copy();minus=plus.copy()
            plus[index]+=1e-3;minus[index]-=1e-3
            hi=p.run({**feed,u.arg:plus},read=[loss])[0]
            lo=p.run({**feed,u.arg:minus},read=[loss])[0]
            np.testing.assert_allclose(analytic[index],(hi-lo)/2e-3,rtol=4e-3,atol=4e-4)
