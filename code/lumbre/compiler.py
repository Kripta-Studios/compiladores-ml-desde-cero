"""C/CUDA/HIP renderer, static scheduler, arena planner and JIT runtime.
Linux/POSIX host. CUDA/HIP routes need the corresponding toolkit and device.
Device validation records live in reports/validation_20261001.
"""
from __future__ import annotations
from collections import Counter
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import numpy as np
from .ir import UOp, topo, size

FUSIBLE = {"add","sub","mul","div","exp","log","sqrt","tanh","eq","lt","where",
           "reshape","expand","permute","slice","pad","concat","detach","onehot","causal"}
CTYPES = {"float32":"float", "int32":"int32_t"}


def coordinates(shape, q):
    return [f"((({q})/{max(1,size(shape[i+1:]))})%{max(1,d)})" for i,d in enumerate(shape)]


def flatten(shape, coords):
    return "(" + "+".join(f"({c})*{size(shape[i+1:])}" for i,c in enumerate(coords)) + ")" if shape else "0"


def bindex(out_shape, in_shape, q):
    cs = coordinates(out_shape,q)[len(out_shape)-len(in_shape):]
    return flatten(in_shape,["0" if d == 1 else c for d,c in zip(in_shape,cs)])


def align(x, a=64): return ((x+a-1)//a)*a


class Program:
    def __init__(self, outputs, backend="cpu", fuse=True, reuse=True, cache=None, gemm="reference"):
        self.outputs = tuple(outputs)
        if not self.outputs: raise ValueError("at least one output is required")
        if backend not in ("cpu","cuda","hip"): raise ValueError("backend must be cpu, cuda or hip")
        self.backend = backend
        if gemm not in ("reference","blocked"): raise ValueError("unknown GEMM strategy")
        self.gemm = gemm
        order = topo(self.outputs)
        names = [u.arg for u in order if u.op == "param"]
        if len(set(names)) != len(names): raise ValueError("same input name used with different shapes/dtypes")
        self.params = {u.arg:u for u in order if u.op == "param"}
        users = Counter(s for u in order for s in u.src)
        mats, depth = set(self.outputs), {}
        for u in order:
            if u.op == "const" and u not in mats:
                depth[u] = 0; continue
            d = 1+max((depth[s] for s in u.src), default=0)
            if not fuse or u.op not in FUSIBLE or users[u] > 1 or d >= 4 or u in mats:
                mats.add(u); depth[u] = 0
            else: depth[u] = d
        # Calls consume contiguous materialized operands. Their boundaries are explicit.
        for u in order:
            if u.op == "attention" or (u.op=="matmul" and gemm=="blocked"):
                mats.update(u.src)
        self.materialized = mats
        self.kernels = [u for u in order if u in mats and u.op != "param"]
        self.ids = {u:i for i,u in enumerate(order)}
        self.inputs = {u:self.dependencies(u) for u in self.kernels}
        self.offsets, self.bytes = self.plan_arena(reuse)
        self.source = self.render()
        self.cache = Path(cache or os.getenv("LUMBRE_CACHE", Path.home()/".cache/lumbre"))
        self.cache.mkdir(parents=True,exist_ok=True)
        compiler = {"cpu":os.getenv("CC","cc"),"cuda":"nvcc","hip":"hipcc"}[backend]
        path = shutil.which(compiler)
        if not path: raise RuntimeError(f"required compiler not found: {compiler}")
        version = subprocess.run([path,"--version"],capture_output=True,text=True,check=True).stdout
        if backend == "cpu": flags = ["-O3","-std=c11","-shared","-fPIC","-ffp-contract=off"]
        elif backend == "cuda":
            arch = os.getenv("LUMBRE_CUDA_ARCH")
            if not arch: raise RuntimeError("set LUMBRE_CUDA_ARCH, e.g. sm_80, for your actual device")
            flags = ["-O3","-std=c++17","-shared","-Xcompiler","-fPIC",f"-arch={arch}"]
        else: flags = ["-O3","-std=c++17","-shared","-fPIC"]
        identity = json.dumps([self.source,path,version,flags,platform.machine(),platform.system()])
        self.digest = hashlib.sha256(identity.encode()).hexdigest()
        self.source_path = self.cache/(self.digest+{"cpu":".c","cuda":".cu","hip":".cpp"}[backend])
        self.library_path = self.cache/(self.digest+".so")
        self.source_path.write_text(self.source,encoding="utf-8")
        self.compile_seconds = 0.0
        if not self.library_path.exists():
            start = time.perf_counter()
            with tempfile.TemporaryDirectory(dir=self.cache) as temp:
                target = Path(temp)/"module.so"
                command = [path,*flags,str(self.source_path),"-o",str(target)]
                if backend == "cpu": command += ["-lm"]
                result = subprocess.run(command,capture_output=True,text=True,timeout=180)
                if result.returncode:
                    raise RuntimeError(f"compiler failed ({self.source_path}):\n{result.stderr[-12000:]}")
                os.replace(target,self.library_path)
            self.compile_seconds = time.perf_counter()-start
        self.lib = C.CDLL(str(self.library_path))
        self.lib.run.argtypes = [C.c_void_p]; self.lib.run.restype = C.c_int
        self._closed = False
        if backend == "cpu":
            self._storage = np.zeros(self.bytes+64,dtype=np.uint8)
            self.address = align(self._storage.ctypes.data)
        else:
            self.lib.lm_alloc.argtypes=[C.c_size_t]; self.lib.lm_alloc.restype=C.c_void_p
            self.lib.lm_free.argtypes=[C.c_void_p]; self.lib.lm_free.restype=C.c_int
            self.lib.lm_copy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]; self.lib.lm_copy.restype=C.c_int
            self.lib.lm_sync.argtypes=[]; self.lib.lm_sync.restype=C.c_int
            self.address = self.lib.lm_alloc(max(self.bytes,64))
            if not self.address: raise RuntimeError("device allocation failed")
        self.initialized = set()
        self.ran = False

    def dependencies(self, root):
        result, seen = [], set()
        def visit(u):
            if u in seen: return
            seen.add(u)
            if u is not root and u in self.materialized:
                result.append(u)
            else:
                for s in u.src: visit(s)
        visit(root)
        return tuple(result)

    def plan_arena(self, reuse):
        offsets, end = {}, 0
        for u in self.params.values():
            offsets[u] = end; end += align(max(1,size(u.shape))*4)
        last = {u:i for i,u in enumerate(self.kernels)}
        for i,u in enumerate(self.kernels):
            for s in self.inputs[u]: last[s] = max(last.get(s,-1),i)
        for u in self.outputs: last[u] = len(self.kernels)
        active, free = [], []
        for i,u in enumerate(self.kernels):
            remaining = []
            for old,off,n in active:
                if reuse and last[old] < i: free.append((n,off))
                else: remaining.append((old,off,n))
            active = remaining
            need = align(max(1,size(u.shape))*4)
            candidates = sorted((n,off,j) for j,(n,off) in enumerate(free) if n >= need)
            if candidates:
                n,off,j = candidates[0]; free.pop(j)
                if n > need: free.append((n-need,off+need))
            else: off = end; end += need
            offsets[u] = off; active.append((u,off,need))
        return offsets, max(end,64)

    def pointer(self,u): return f"p{self.ids[u]}"

    def read(self,u,q,root):
        if u is not root and u in self.materialized: return f"{self.pointer(u)}[{q}]"
        op,s = u.op,u.src
        if op == "const": return float.fromhex(u.arg).hex()+"f"
        if op == "causal":
            T=u.shape[-1]
            return f"((({q})%{T})<=(({q})/{T}) ? 0.0f : -INFINITY)"
        if op == "param": return f"{self.pointer(u)}[{q}]"
        if op in ("reshape","detach"): return self.read(s[0],q,root)
        if op == "expand": return self.read(s[0],bindex(u.shape,s[0].shape,q),root)
        if op == "permute":
            cs = coordinates(u.shape,q)
            back = [cs[u.arg.index(i)] for i in range(len(u.shape))]
            return self.read(s[0],flatten(s[0].shape,back),root)
        if op == "slice":
            cs = coordinates(u.shape,q); axis,start,_ = u.arg
            cs[axis] = f"({cs[axis]}+{start})"
            return self.read(s[0],flatten(s[0].shape,cs),root)
        if op == "pad":
            cs = coordinates(u.shape,q)
            test = " && ".join(f"({c}>={l} && {c}<{l+d})" for c,d,(l,r) in zip(cs,s[0].shape,u.arg)) or "1"
            back = [f"({c}-{p[0]})" for c,p in zip(cs,u.arg)]
            v = self.read(s[0],flatten(s[0].shape,back),root)
            return f"(({test}) ? ({v}) : 0)"
        if op == "concat":
            cs = coordinates(u.shape,q); axis=u.arg; cut=s[0].shape[axis]
            a = self.read(s[0],flatten(s[0].shape,cs),root)
            bs = cs.copy(); bs[axis]=f"({bs[axis]}-{cut})"
            b = self.read(s[1],flatten(s[1].shape,bs),root)
            return f"(({cs[axis]}<{cut}) ? ({a}) : ({b}))"
        if op == "onehot":
            v = self.read(s[0],f"(({q})/{u.arg})",root)
            return f"(({v}) == (({q})%{u.arg}) ? 1.0f : 0.0f)"
        args = [self.read(x,bindex(u.shape,x.shape,q),root) for x in s]
        if op in ("add","sub","mul","div","eq","lt"):
            symbol={"add":"+","sub":"-","mul":"*","div":"/","eq":"==","lt":"<"}[op]
            return f"(({args[0]}){symbol}({args[1]}))"
        if op in ("exp","log","sqrt","tanh"): return f"{op}f({args[0]})"
        if op == "where": return f"(({args[0]}) ? ({args[1]}) : ({args[2]}))"
        raise NotImplementedError(f"cannot inline {op}")

    def body(self,u):
        out = self.pointer(u); op=u.op; s=u.src
        if op in FUSIBLE or op == "const": return f"{out}[q] = {self.read(u,'q',u)};"
        if op in ("sum","max"):
            shape = s[0].shape
            cs = coordinates(u.shape,"q")
            redshape = tuple(shape[i] for i in u.arg)
            rs = coordinates(redshape,"r")
            back = [rs[u.arg.index(i)] if i in u.arg else cs[i] for i in range(len(shape))]
            v = self.read(s[0],flatten(shape,back),u)
            initial = "0.0f" if op == "sum" else "-INFINITY"
            update = f"acc += ({v});" if op == "sum" else f"acc = fmaxf(acc,({v}));"
            return f"float acc={initial};\nfor(int64_t r=0;r<{size(redshape)};r++) {{ {update} }}\n{out}[q]=acc;"
        if op == "matmul":
            a,b=s; M,K=a.shape[-2:]; N=b.shape[-1]
            batch=u.shape[:-2]
            ab = bindex(batch,a.shape[:-2],f"q/{max(1,M*N)}")
            bb = bindex(batch,b.shape[:-2],f"q/{max(1,M*N)}")
            ai = f"(({ab})*{M*K}+((q/{max(1,N)})%{max(1,M)})*{K}+k)"
            bi = f"(({bb})*{K*N}+k*{N}+q%{max(1,N)})"
            av,bv=self.read(a,ai,u),self.read(b,bi,u)
            return f"float acc=0.0f;\nfor(int64_t k=0;k<{K};k++) acc+=({av})*({bv});\n{out}[q]=acc;"
        if op == "gather":
            table,ids=s; V,D=table.shape
            iv=self.read(ids,f"q/{max(1,D)}",u)
            tv=self.read(table,f"((int64_t)row*{D}+q%{max(1,D)})",u)
            return f"int32_t row={iv};\n{out}[q]=(row>=0 && row<{V}) ? {tv} : NAN;"
        if op == "scatter":
            ids,g=s; V,D=u.shape
            iv=self.read(ids,"r",u); gv=self.read(g,f"(r*{D}+q%{max(1,D)})",u)
            return f"float acc=0.0f;\nfor(int64_t r=0;r<{size(ids.shape)};r++) if(({iv})==q/{max(1,D)}) acc+=({gv});\n{out}[q]=acc;"
        raise NotImplementedError(op)

    def render(self):
        gpu = self.backend != "cpu"
        header = "#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n"
        prefix = "extern \"C\" " if gpu else ""
        if gpu:
            p="cuda" if self.backend == "cuda" else "hip"
            header += "#include <cuda_runtime.h>\n" if p=="cuda" else "#include <hip/hip_runtime.h>\n"
            header += f'''extern "C" void* lm_alloc(size_t n) {{ void* p=nullptr; return {p}Malloc(&p,n)=={p}Success ? p : nullptr; }}
extern "C" int lm_free(void* p) {{ return (int){p}Free(p); }}
extern "C" int lm_sync() {{ return (int){p}DeviceSynchronize(); }}
extern "C" int lm_copy(void* d,const void* s,size_t n,int kind) {{
  auto k=kind==1 ? {p}MemcpyHostToDevice : (kind==2 ? {p}MemcpyDeviceToHost : {p}MemcpyDeviceToDevice);
  return (int){p}Memcpy(d,s,n,k);
}}
'''
        header += self.call_helpers(gpu)
        definitions,run=[],[]
        for u,off in self.offsets.items():
            run.append(f"{CTYPES[u.dtype]}* {self.pointer(u)}=({CTYPES[u.dtype]}*)(base+{off});")
        for i,u in enumerate(self.kernels):
            count=size(u.shape)
            if count==0: continue
            if u.op == "attention":
                T,D=u.shape[-2:]; rows=size(u.shape)//D
                q,k,v=map(self.pointer,u.src); out=self.pointer(u)
                if not gpu:
                    run.append(f"for(int64_t r=0;r<{rows};r++) lm_attention_row({q},{k},{v},{out},r,{T},{D});")
                else:
                    definitions.append(f"__global__ void k{i}(const float* Q,const float* K,const float* V,float* O) {{ int64_t r=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(r<{rows}) lm_attention_row(Q,K,V,O,r,{T},{D}); }}")
                    run.append(f"k{i}<<<{(rows+127)//128},128>>>({q},{k},{v},{out});")
                continue
            if u.op == "matmul" and self.gemm=="blocked":
                a,b=u.src; M,K=a.shape[-2:]; N=b.shape[-1]
                B=size(u.shape[:-2]); ab=bindex(u.shape[:-2],a.shape[:-2],"batch")
                bb=bindex(u.shape[:-2],b.shape[:-2],"batch")
                A,BB,O=self.pointer(a),self.pointer(b),self.pointer(u)
                if not gpu:
                    run.append(f"for(int64_t batch=0;batch<{B};batch++) lm_gemm({A}+({ab})*{M*K},{BB}+({bb})*{K*N},{O}+batch*{M*N},{M},{N},{K});")
                else:
                    if B>65535: raise ValueError("tiled GPU GEMM batch exceeds this renderer's grid-z limit")
                    definitions.append(f"""__global__ void k{i}(const float* AA,const float* BB,float* CC) {{
  int64_t batch=blockIdx.z; const float* A=AA+({ab})*{M*K}; const float* B=BB+({bb})*{K*N}; float* C=CC+batch*{M*N};
  __shared__ float a[16][16], b[16][16];
  int tx=threadIdx.x,ty=threadIdx.y; int64_t row=(int64_t)blockIdx.y*16+ty,col=(int64_t)blockIdx.x*16+tx;
  float acc=0.0f;
  for(int64_t k0=0;k0<{K};k0+=16) {{
    a[ty][tx]=(row<{M} && k0+tx<{K}) ? A[row*{K}+k0+tx] : 0.0f;
    b[ty][tx]=(k0+ty<{K} && col<{N}) ? B[(k0+ty)*{N}+col] : 0.0f;
    __syncthreads();
    for(int t=0;t<16;t++) acc+=a[ty][t]*b[t][tx];
    __syncthreads();
  }}
  if(row<{M} && col<{N}) C[row*{N}+col]=acc;
}}""")
                    run.append(f"k{i}<<<dim3({(N+15)//16},{(M+15)//16},{B}),dim3(16,16)>>>({A},{BB},{O});")
                continue
            if not gpu:
                run.append(f"/* kernel {i}: {u.op} {u.shape} */\nfor(int64_t q=0;q<{count};q++) {{\n{self.body(u)}\n}}")
            else:
                args=(u,)+self.inputs[u]
                signature=", ".join(f"{CTYPES[x.dtype]}* {self.pointer(x)}" for x in args)
                definitions.append(f"__global__ void k{i}({signature}) {{\nint64_t q=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;\nif(q<{count}) {{\n{self.body(u)}\n}}\n}}")
                run.append(f"k{i}<<<{(count+127)//128},128>>>({','.join(self.pointer(x) for x in args)});")
        ret="0" if not gpu else ("(int)cudaGetLastError()" if self.backend=="cuda" else "(int)hipGetLastError()")
        return header+"\n".join(definitions)+f"\n{prefix}int run(unsigned char* base) {{\n"+"\n".join(run)+f"\nreturn {ret};\n}}\n"

    def call_helpers(self,gpu):
        prefix="__device__ " if gpu else ""
        parts=[]
        if any(u.op=="attention" for u in self.kernels):
            parts.append(prefix+"""static void lm_attention_row(const float* Q,const float* K,const float* V,float* O,int64_t r,int64_t T,int64_t D) {
  int64_t i=r%T,base=(r/T)*T*D;
  const float* q=Q+r*D; float* o=O+r*D;
  float m=-INFINITY,den=0.0f,scale=1.0f/sqrtf((float)D);
  for(int64_t d=0;d<D;d++) o[d]=0.0f;
  for(int64_t j=0;j<=i;j++) {
    float score=0.0f;
    for(int64_t d=0;d<D;d++) score+=q[d]*K[base+j*D+d];
    score*=scale;
    float nm=fmaxf(m,score),alpha=expf(m-nm),beta=expf(score-nm);
    for(int64_t d=0;d<D;d++) o[d]=alpha*o[d]+beta*V[base+j*D+d];
    den=alpha*den+beta; m=nm;
  }
  for(int64_t d=0;d<D;d++) o[d]/=den;
}
""")
        if self.gemm=="blocked" and not gpu:
            parts.append("""static void lm_gemm(const float* A,const float* B,float* C,int64_t M,int64_t N,int64_t K) {
  for(int64_t ii=0;ii<M;ii+=4) for(int64_t jj=0;jj<N;jj+=8) {
    float acc[4][8]={{0}};
    for(int64_t kk=0;kk<K;kk+=64) {
      int64_t end=kk+64<K ? kk+64 : K;
      for(int64_t k=kk;k<end;k++) for(int i=0;i<4 && ii+i<M;i++) {
        float av=A[(ii+i)*K+k];
        for(int j=0;j<8 && jj+j<N;j++) acc[i][j]+=av*B[k*N+jj+j];
      }
    }
    for(int i=0;i<4 && ii+i<M;i++) for(int j=0;j<8 && jj+j<N;j++) C[(ii+i)*N+jj+j]=acc[i][j];
  }
}
""")
        return "".join(parts)

    def _view(self,u):
        buf=(C.c_uint8*(size(u.shape)*4)).from_address(self.address+self.offsets[u])
        return np.frombuffer(buf,dtype=u.dtype).reshape(u.shape)

    def set(self,name,value):
        if self._closed: raise RuntimeError("program is closed")
        u=self.params[name]; a=np.asarray(value,dtype=u.dtype)
        if a.shape != u.shape: raise ValueError(f"{name}: expected {u.shape}, got {a.shape}")
        a=np.array(a,dtype=u.dtype,order="C",copy=True)
        if self.backend=="cpu": self._view(u)[...]=a
        else: self._check(self.lib.lm_copy(self.address+self.offsets[u],a.ctypes.data,a.nbytes,1))
        self.initialized.add(name)

    def run(self,feed=None,read=None):
        if self._closed: raise RuntimeError("program is closed")
        for name,value in (feed or {}).items(): self.set(name,value)
        missing=set(self.params)-self.initialized
        if missing: raise ValueError(f"uninitialized parameters: {sorted(missing)}")
        self._check(self.lib.run(self.address)); self.ran=True
        if self.backend!="cpu": self._check(self.lib.lm_sync())
        selected=self.outputs if read is None else tuple(read)
        return [self.read_output(u) for u in selected]

    def read_output(self,u):
        if not self.ran or u not in self.outputs: raise ValueError("read requires an evaluated declared output")
        if self.backend=="cpu": return self._view(u).copy()
        out=np.empty(u.shape,dtype=u.dtype)
        self._check(self.lib.lm_copy(out.ctypes.data,self.address+self.offsets[u],out.nbytes,2))
        return out

    def assign(self,updates):
        """Commit only after the complete graph has read the old parameter state."""
        if not self.ran: raise RuntimeError("run before assign")
        for name,u in updates.items():
            target=self.params[name]
            if u not in self.outputs or u.op=="param" or u.shape!=target.shape or u.dtype!=target.dtype:
                raise ValueError(f"invalid update for {name}")
        # Input tensors live outside the scratch region: updates cannot overwrite outputs.
        for name,u in updates.items():
            target=self.params[name]
            if self.backend=="cpu": self._view(target)[...]=self._view(u)
            else:
                self._check(self.lib.lm_copy(self.address+self.offsets[target],self.address+self.offsets[u],size(u.shape)*4,3))
            self.initialized.add(name)

    @staticmethod
    def _check(status):
        if status: raise RuntimeError(f"device/runtime error code {status}")

    def close(self):
        if not self._closed and self.backend!="cpu": self._check(self.lib.lm_free(self.address))
        self._closed=True

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def report(self):
        return {"backend":self.backend,"kernels":len(self.kernels),"arena_bytes":self.bytes,
                "gemm_strategy":self.gemm,"compile_seconds":self.compile_seconds,"sha256":self.digest,"source":str(self.source_path)}
