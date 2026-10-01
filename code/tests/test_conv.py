"""Compile and validate the standalone C reference, not a claimed fast kernel."""
import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest

@pytest.fixture(scope='module')
def conv(tmp_path_factory):
    cc=shutil.which('cc') or shutil.which('gcc')
    if cc is None: pytest.skip('C compiler unavailable')
    source=Path(__file__).resolve().parents[1]/'kernels'/'conv.c'
    target=tmp_path_factory.mktemp('conv')/'conv.so'
    subprocess.run([cc,'-std=c11','-O2','-shared','-fPIC',str(source),'-o',str(target)],check=True)
    lib=ct.CDLL(str(target)); f=lib.conv2d_nchw
    f.argtypes=[ct.POINTER(ct.c_float)]*4+[ct.c_int]*14
    f.restype=ct.c_int
    return f

@pytest.mark.parametrize('groups,stride,dilation,pad',[(1,1,1,0),(1,2,1,1),(2,1,2,2),(4,1,1,1)])
def test_conv_reference(conv,groups,stride,dilation,pad):
    B,C,H,W,O,KH,KW=2,4,7,8,8,3,2
    OH=(H+2*pad-dilation*(KH-1)-1)//stride+1
    OW=(W+2*pad-dilation*(KW-1)-1)//stride+1
    rng=np.random.default_rng(7)
    x=rng.normal(size=(B,C,H,W)).astype('float32')
    w=rng.normal(size=(O,C//groups,KH,KW)).astype('float32')
    bias=rng.normal(size=O).astype('float32');out=np.empty((B,O,OH,OW),dtype='float32')
    ptr=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_float))
    assert conv(*map(ptr,(x,w,bias,out)),B,C,H,W,O,KH,KW,stride,stride,pad,pad,dilation,dilation,groups)==0
    want=np.zeros_like(out,dtype='float64')
    for b,o,y,z in np.ndindex(want.shape):
        want[b,o,y,z]=bias[o]
        group=o//(O//groups)
        for ci,ky,kx in np.ndindex((C//groups,KH,KW)):
            iy=y*stride-pad+ky*dilation;ix=z*stride-pad+kx*dilation
            if 0<=iy<H and 0<=ix<W:
                want[b,o,y,z]+=float(x[b,group*(C//groups)+ci,iy,ix])*float(w[o,ci,ky,kx])
    np.testing.assert_allclose(out,want,rtol=2e-5,atol=2e-5)
