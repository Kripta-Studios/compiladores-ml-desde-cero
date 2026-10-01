"""Train a small real decoder with Lumbre-generated C; CUDA/HIP are optional.
The supplied original corpus is a debugging corpus, not evidence of useful LLM quality.
"""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from lumbre import Program
from lumbre.nn import Decoder,Config,adamw
from lumbre.tokenizer import ByteBPE

CORPUS = """el sol sale y la luna descansa.\nla casa tiene una puerta azul.\nel gato mira al rio y escucha el viento.\nuna idea se convierte en un programa.\nprimero comprobamos el resultado y despues medimos el tiempo.\nla memoria guarda datos y el compilador organiza el trabajo.\n"""


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",choices=["cpu","cuda","hip"],default="cpu")
    ap.add_argument("--steps",type=int,default=60)
    ap.add_argument("--dim",type=int,default=16)
    ap.add_argument("--layers",type=int,default=1)
    ap.add_argument("--context",type=int,default=8)
    ap.add_argument("--text",type=Path)
    ap.add_argument("--byte-tokens",action="store_true")
    ap.add_argument("--bpe-merges",type=int,default=0)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--heads",type=int,default=2)
    ap.add_argument("--kv-heads",type=int,default=1)
    ap.add_argument("--hidden",type=int,default=0)
    ap.add_argument("--online",action="store_true")
    ap.add_argument("--gemm",choices=["reference","blocked"],default="reference")
    ap.add_argument("--resume",type=Path)
    ap.add_argument("--output",type=Path,default=Path("reports/decoder"))
    args=ap.parse_args()
    if args.steps < 1: raise ValueError("steps must be positive")
    text=args.text.read_text(encoding="utf-8") if args.text else CORPUS*20
    if args.bpe_merges<0: raise ValueError("bpe-merges must be nonnegative")
    if args.byte_tokens or args.bpe_merges:
        split=int(0.9*len(text)); raw_train,raw_val=text[:split],text[split:]
        codec=ByteBPE.fit(raw_train,args.bpe_merges)
        alphabet=[piece.hex() for piece in codec.vocab]
        encode_text=codec.encode
        decode_tokens=lambda tokens:codec.decode(tokens,errors="replace")
        tokenizer_state=codec.state()
        train=np.asarray(codec.encode(raw_train),dtype=np.int32)
        val=np.asarray(codec.encode(raw_val),dtype=np.int32)
    else:
        alphabet=sorted(set(text)); encode={c:i for i,c in enumerate(alphabet)}
        encode_text=lambda value:[encode.get(c,0) for c in value]
        decode_tokens=lambda tokens:"".join(alphabet[int(t)] for t in tokens)
        tokenizer_state={"kind":"character","alphabet":alphabet}
        ids=np.asarray([encode[c] for c in text],dtype=np.int32)
        split=int(0.9*len(ids)); train,val=ids[:split],ids[split:]
    if min(len(train),len(val)) <= args.context+1: raise ValueError("corpus is too small for train/validation")
    cfg=Config(vocab=len(alphabet),dim=args.dim,heads=args.heads,kv_heads=args.kv_heads,layers=args.layers,hidden=args.hidden or 2*args.dim,context=args.context,batch=args.batch,online=args.online)
    metadata={"model":asdict(cfg),"alphabet":alphabet,"tokenizer":tokenizer_state,
              "corpus_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest()}
    model=Decoder(cfg)
    outputs,updates,optinit=adamw(model)
    print("Compiling",model.parameter_count(),"parameters...",flush=True)
    program=Program(outputs,backend=args.backend,gemm=args.gemm)
    rng=np.random.default_rng(19)
    def batch(data):
        starts=rng.integers(0,len(data)-cfg.context,size=cfg.batch)
        x=np.stack([data[s:s+cfg.context] for s in starts])
        y=np.stack([data[s+1:s+cfg.context+1] for s in starts])
        return x,y
    for name,value in {**model.initial,**optinit}.items():
        if name in program.params: program.set(name,value)
    start_step=0
    if args.resume:
        archive=np.load(args.resume,allow_pickle=False)
        expected=json.dumps(metadata,sort_keys=True)
        if "configuration" not in archive or str(archive["configuration"])!=expected:
            raise ValueError("checkpoint configuration or vocabulary mismatch")
        for name in list(model.weights)+list(optinit): program.set(name,archive[name])
        start_step=int(archive["step"])
        rng.bit_generator.state=json.loads(str(archive["rng_state"]))
    args.output.mkdir(parents=True,exist_ok=True)
    history=[]; start=time.perf_counter()
    for step in range(start_step+1,start_step+args.steps+1):
        x,y=batch(train)
        feed={"tokens":x,"labels":y,"learning_rate":np.array(0.004,dtype="float32"),
              "adam_bc1":np.array(1-0.9**step,dtype="float32"),"adam_bc2":np.array(1-0.999**step,dtype="float32")}
        loss,norm=program.run(feed,read=outputs[:2])
        if not np.isfinite(loss) or not np.isfinite(norm): raise RuntimeError("nonfinite training state")
        program.assign(updates)
        row={"step":step,"loss":float(loss),"grad_norm":float(norm)};history.append(row)
        if step==1 or step%10==0: print(row,flush=True)
    elapsed=time.perf_counter()-start
    # Reading persistent weights is an explicit checkpoint operation, not training through NumPy.
    saved={}
    for name in list(model.weights)+list(optinit):
        u=program.params[name]
        if args.backend=="cpu": saved[name]=program._view(u).copy()
        else:
            import ctypes
            a=np.empty(u.shape,dtype=u.dtype)
            program._check(program.lib.lm_copy(a.ctypes.data,program.address+program.offsets[u],a.nbytes,2))
            saved[name]=a
    saved["step"]=np.array(start_step+args.steps)
    saved["configuration"]=np.array(json.dumps(metadata,sort_keys=True))
    saved["rng_state"]=np.array(json.dumps(rng.bit_generator.state))
    np.savez(args.output/"checkpoint.npz",**saved)
    # A separate inference graph has no optimizer and does not modify weights.
    inference=Program([model.loss,model.logits],backend=args.backend,gemm=args.gemm)
    for name,value in model.initial.items():
        if name in inference.params: inference.set(name,saved.get(name,value))
    vals=[]
    for _ in range(4):
        x,y=batch(val)
        v=inference.run({"tokens":x,"labels":y},read=[model.loss])[0]; vals.append(float(v))
    prompt=encode_text("el sol ")
    for _ in range(80):
        ctx=([0]*cfg.context+prompt)[-cfg.context:]
        x=np.tile(np.asarray(ctx,dtype=np.int32),(cfg.batch,1))
        logits=inference.run({"tokens":x,"labels":x},read=[model.logits])[0][0,-1]
        p=np.exp((logits-logits.max())/0.8); p/=p.sum()
        prompt.append(int(rng.choice(len(alphabet),p=p)))
    report={"config":asdict(cfg),"parameters":model.parameter_count(),"backend":args.backend,
            "steps":args.steps,"train_seconds_excluding_compile":elapsed,"history":history,
            "validation_loss":float(np.mean(vals)),"sample":decode_tokens(prompt),
            "compiler":program.report(),"corpus":"external UTF-8" if args.text else "original repeated teaching corpus",
            "warning":"debugging-scale training; not SOTA quality, not an independent language benchmark"}
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.output/"config.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Validation",report["validation_loss"],"seconds",elapsed,flush=True)
    print(report["sample"],flush=True)
    inference.close();program.close()

if __name__=="__main__": main()
