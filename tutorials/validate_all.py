"""Sequential tutorial campaign; run from the pinned tinygrad environment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=root/'tutorials/reports')
parser.add_argument('--arch',default='sm_120')
args=parser.parse_args()
out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
commands=[('cuda',[sys.executable,str(root/'tutorials/validate_cuda.py'),'--arch',args.arch,'--output',str(out/'cuda')])]
for device in ('CPU','CUDA'):
    commands.append(('tinygrad_'+device,[sys.executable,str(root/'tutorials/tinygrad/labs.py'),
                     '--device',device,'--output',str(out/'tinygrad'/device)]))
report={'python':sys.version,'platform':platform.platform(),
        'tinygrad_version':importlib.metadata.version('tinygrad'),
        'numpy_version':importlib.metadata.version('numpy'),
        'environment':{k:os.environ.get(k) for k in ('DEV','CC','CUDA_PATH')},
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True),
        'clang':subprocess.check_output([os.environ.get('CC','clang'),'--version'],text=True).splitlines()[0],
        'sequence':[], 'sources':{}}
for name,command in commands:
    start=time.perf_counter()
    run=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/(name+'.log')).write_text(run.stdout,encoding='utf-8')
    report['sequence'].append({'name':name,'returncode':run.returncode,'seconds':time.perf_counter()-start})
    print(run.stdout,flush=True)
    if run.returncode: raise RuntimeError(f'{name} failed; inspect its log')
for path in sorted((root/'tutorials').rglob('*')):
    if path.suffix in ('.py','.cu','.cuh') and '__pycache__' not in path.parts:
        report['sources'][path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
report['measurement_scope']='Sequential programs on the same host. No fixed clocks, power isolation or desktop-load isolation.'
(out/'campaign.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
