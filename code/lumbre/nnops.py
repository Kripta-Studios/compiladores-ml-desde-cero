"""Composite CNN operation using only existing differentiable IR primitives.
This explicit patch construction is for small correctness laboratories. It grows
with the number of output positions; it is not an optimized im2col lowering.
"""
from functools import reduce
from .ir import concat


def join(values,axis):
    if not values: raise ValueError('cannot concatenate an empty sequence')
    return reduce(lambda a,b:concat(a,b,axis),values)


def pair(value):
    if isinstance(value,int): return (value,value)
    if len(value)!=2: raise ValueError('expected an integer or pair')
    return tuple(int(x) for x in value)


def conv2d(x,w,bias=None,stride=1,padding=0,dilation=1,groups=1):
    """NCHW x OIHW -> NCHW, grouped, symmetric padding, static FP32.
    Gradients come from composition; no handwritten convolution VJP is used.
    A 1024-position cap prevents accidentally building huge teaching graphs.
    """
    if len(x.shape)!=4 or len(w.shape)!=4: raise ValueError('conv2d needs rank 4')
    if x.dtype!='float32' or w.dtype!='float32': raise TypeError('conv2d needs FP32')
    B,C,H,W=x.shape;O,CG,KH,KW=w.shape
    sh,sw=pair(stride);ph,pw=pair(padding);dh,dw=pair(dilation)
    if min(B,C,H,W,O,CG,KH,KW,sh,sw,dh,dw,groups)<=0 or min(ph,pw)<0:
        raise ValueError('invalid convolution dimensions')
    if C%groups or O%groups or CG!=C//groups: raise ValueError('invalid groups')
    OH=(H+2*ph-dh*(KH-1)-1)//sh+1
    OW=(W+2*pw-dw*(KW-1)-1)//sw+1
    if min(OH,OW)<=0 or OH*OW>1024: raise ValueError('output outside teaching limits')
    if bias is not None and (bias.shape!=(O,) or bias.dtype!='float32'):
        raise ValueError('bias must be FP32[O]')
    xp=x.pad(((0,0),(0,0),(ph,ph),(pw,pw)))
    OG=O//groups;outputs=[]
    for group in range(groups):
        weight=w.slice(0,group*OG,(group+1)*OG).reshape(OG,CG*KH*KW).transpose()
        rows=[]
        for y in range(OH):
            columns=[]
            for z in range(OW):
                entries=[]
                for ci in range(CG):
                    for ky in range(KH):
                        for kx in range(KW):
                            iy=y*sh+ky*dh;ix=z*sw+kx*dw;c=group*CG+ci
                            entries.append(xp.slice(1,c,c+1).slice(2,iy,iy+1).slice(3,ix,ix+1).reshape(B,1))
                columns.append((join(entries,1)@weight).reshape(B,OG,1,1))
            rows.append(join(columns,3))
        outputs.append(join(rows,2))
    result=join(outputs,1)
    return result if bias is None else result+bias.reshape(1,O,1,1)
