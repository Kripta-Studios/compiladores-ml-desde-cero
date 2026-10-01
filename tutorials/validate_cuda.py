"""Compile and run all original CUDA tutorial programs in temporary storage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--arch',default='sm_120')
parser.add_argument('--output',type=Path,default=root/'reports/cuda')
args=parser.parse_args()
nvcc=shutil.which('nvcc')
if not nvcc: raise SystemExit('nvcc must be on PATH')
args.output.mkdir(parents=True,exist_ok=True)
report={'arch':args.arch,'nvcc':subprocess.check_output([nvcc,'--version'],text=True),'programs':[]}
with tempfile.TemporaryDirectory(prefix='cuda-lessons-') as directory:
    for source in sorted((root/'cuda').glob('*.cu')):
        exe=Path(directory)/source.stem
        command=[nvcc,'-O2','-lineinfo','-std=c++17',f'-arch={args.arch}',str(source),'-o',str(exe)]
        subprocess.run(command,check=True)
        run=subprocess.run([str(exe)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (args.output/(source.stem+'.log')).write_text(run.stdout,encoding='utf-8')
        if run.returncode: raise RuntimeError(run.stdout)
        report['programs'].append({'file':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                   'returncode':run.returncode,'passing_cases':run.stdout.count('PASS')})
        print(source.name,run.stdout.count('PASS'),'cases PASS',flush=True)
report['temporary_binaries_removed']=not Path(directory).exists()
(args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
