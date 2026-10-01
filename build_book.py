from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
ROOT=Path(__file__).parent
SPECIAL={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
'→':r'\ensuremath{\rightarrow}','←':r'\ensuremath{\leftarrow}','↔':r'\ensuremath{\leftrightarrow}','×':r'\ensuremath{\times}','≤':r'\ensuremath{\leq}','≥':r'\ensuremath{\geq}','≠':r'\ensuremath{\neq}','∈':r'\ensuremath{\in}','∞':r'\ensuremath{\infty}','−':'-','–':'--','—':'---','·':r'\ensuremath{\cdot}','²':r'\ensuremath{^2}','³':r'\ensuremath{^3}','…':r'\ldots{}','μ':r'\ensuremath{\mu}','ε':r'\ensuremath{\varepsilon}'}
def esc(x):return ''.join(SPECIAL.get(c,c) for c in x)
def inline(s):
    pattern=r'(\[\@[^\]]+\]|`[^`]+`|\$[^$\n]+\$|\*\*[^*]+\*\*)'
    pieces=re.split(pattern,s);out=[]
    for x in pieces:
        if x.startswith('[@') and x.endswith(']'):
            out.append(r'\cite{'+x[2:-1].replace(';@',',').replace(';',',').replace(' ','')+'}')
        elif x.startswith('`') and x.endswith('`'):
            txt=x[1:-1]
            if len(txt)>20:
                out.append(r'\texttt{'+r'\allowbreak{}'.join(esc(txt[i:i+12]) for i in range(0,len(txt),12))+'}')
            else:out.append(r'\texttt{'+esc(txt)+'}')
        elif x.startswith('$') and x.endswith('$'):out.append(x)
        elif x.startswith('**') and x.endswith('**'):out.append(r'\textbf{'+esc(x[2:-2])+'}')
        else:out.append(esc(x))
    return ''.join(out)
def parse(text):
    lines=text.splitlines();out=[];para=[];i=0;boxes=[]
    def flush():
        if para:out.append(inline(' '.join(para))+'\n\n');para.clear()
    while i<len(lines):
        line=lines[i];i+=1
        if not line.strip():flush();continue
        if line.startswith('```'):
            flush();lang=line[3:].strip();body=[]
            while i<len(lines) and not lines[i].startswith('```'):
                body.append(lines[i]);i+=1
            i+=1;b='\n'.join(body)
            if lang=='math':out.append('\\[\n'+b+'\n\\]\n')
            elif lang=='tikz':out.append(b+'\n')
            elif lang=='diagram':
                items=[x.strip() for x in b.split('|')]
                # Three horizontal stages fit the A4 text block.
                out.append('\\begin{center}\n\\begin{tikzpicture}\n')
                for j,item in enumerate(items):
                    out.append(f'\\node[box,text width=4.15cm] (n{j}) at ({j*5.15},0) {{{inline(item)}}};\n')
                    if j:out.append(f'\\draw[flow] (n{j-1}) -- (n{j});\n')
                out.append('\\end{tikzpicture}\n\\end{center}\n')
            else:
                languages={'python':'Python','c':'C','cpp':'C++','cuda':'C++','bash':'bash','json':'','text':'','toml':''}
                ll=languages.get(lang,'')
                out.append('\\begin{lstlisting}'+ ('[language='+ll+']' if ll else '')+'\n'+b+'\n\\end{lstlisting}\n')
            continue
        if line.startswith(':::'):
            flush();spec=line[3:].strip()
            if not spec:
                out.append('\\end{'+boxes.pop()+'}\n')
            else:
                typ,_,title=spec.partition(' ');boxes.append(typ)
                out.append('\\begin{'+typ+'}['+inline(title)+']\n')
            continue
        if line.startswith('#'):
            flush();m=re.match(r'(#+)(\*)?\s+(.*)',line);level=len(m[1]);star=m[2] or '';title=m[3]
            label=re.search(r'\s*\{#([^}]+)\}$',title)
            if label:title=title[:label.start()]
            cmd={1:'chapter',2:'section',3:'subsection',4:'paragraph'}.get(level,'paragraph')
            out.append('\\'+cmd+star+'{'+inline(title)+'}\n')
            if star:out.append('\\addcontentsline{toc}{'+cmd+'}{'+inline(title)+'}\n')
            if label:out.append('\\label{'+label[1]+'}\n')
            continue
        if line.startswith('|'):
            flush();rows=[line]
            while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
            cells=[[x.strip() for x in r.strip('|').split('|')] for r in rows]
            cells=[r for r in cells if not all(re.fullmatch(r'[:\- ]+',x or '-') for x in r)]
            n=len(cells[0]);widths={2:[.3,.65],3:[.28,.32,.34],4:[.22,.23,.23,.23],5:[.17,.18,.18,.18,.17]}.get(n,[.9/n]*n)
            spec='@{}'+''.join('P{'+str(w)+r'\textwidth}' for w in widths)+'@{}'
            out.append('{\\small\n\\begin{longtable}{'+spec+'}\n\\toprule\n')
            header=' & '.join('\\textbf{'+inline(x)+'}' for x in cells[0])+r'\\'
            out.append(header+'\n\\midrule\n\\endfirsthead\n\\toprule\n'+header+'\n\\midrule\n\\endhead\n')
            for row in cells[1:]:out.append(' & '.join(inline(x) for x in row)+r'\\'+'\n')
            out.append('\\bottomrule\n\\end{longtable}\n}\n')
            continue
        if line.startswith('- '):
            flush();items=[line[2:]]
            while i<len(lines) and lines[i].startswith('- '):items.append(lines[i][2:]);i+=1
            out.append('\\begin{itemize}\n'+''.join('\\item '+inline(x)+'\n' for x in items)+'\\end{itemize}\n');continue
        para.append(line)
    flush()
    if boxes:raise ValueError(f'Unclosed boxes {boxes}')
    return ''.join(out)

def build():
    tex=(ROOT/'preamble.tex').read_text(encoding='utf-8')
    for p in sorted((ROOT/'chapters').glob('*.md')):tex+='\n% SOURCE: '+p.name+'\n'+parse(p.read_text(encoding='utf-8'))
    # Appendices are self-contained: listings are embedded, not external inputs.
    tex+='\n\\appendix\n\\chapter{Código completo de Lumbre y sus pruebas}\\label{app:codigo}\n'
    tex+='El código siguiente coincide con los archivos del paquete de esta edición. Los comentarios del software se conservan en inglés para facilitar trabajo técnico; el desarrollo didáctico del libro está en español. La revisión local valida CUDA en una RTX 5070 Ti Laptop; HIP sigue pendiente de hardware compatible.\\par\n'
    files=list((ROOT/'code/lumbre').glob('*.py'))+list((ROOT/'code/examples').glob('*.py'))+list((ROOT/'code/kernels').glob('*'))+list((ROOT/'code/tests').glob('*.py'))
    for p in sorted(files):
        if not p.is_file():continue
        rel=p.relative_to(ROOT/'code').as_posix();s=p.read_text(encoding='utf-8')
        if len(s.splitlines()) <= 26:
            tex+='\\needspace{'+str((len(s.splitlines())+7)*12)+'pt}\n'
        tex+='\\section{'+inline(str(rel))+'}\n'
        lang='Python' if p.suffix=='.py' else 'C++' if p.suffix in('.cu','.cpp','.hip') else 'C'
        tex+='\\begin{lstlisting}[language='+lang+']\n'+s+'\n\\end{lstlisting}\n'
    if (ROOT/'appendices.tex').exists():tex+=(ROOT/'appendices.tex').read_text(encoding='utf-8')
    if (ROOT/'bibliography.tex').exists():tex+=(ROOT/'bibliography.tex').read_text(encoding='utf-8')
    tex+='\n\\clearpage\n\\renewcommand{\\contentsname}{Índice detallado}\n\\tableofcontents\n\\end{document}\n'
    (ROOT/'Compiladores_ML_Desde_Cero_2026.tex').write_text(tex,encoding='utf-8',newline='\n')
    print(len(tex), 'characters;',len(files),'embedded source files')
def compile_pdf(engine='pdflatex', report=None, source=None):
    """Compile a self-contained snapshot and remove its temporary auxiliaries."""
    executable = shutil.which(engine)
    if not executable:
        raise RuntimeError(f'LaTeX engine not found: {engine}')
    source = Path(source) if source is not None else ROOT/'Compiladores_ML_Desde_Cero_2026.tex'
    source_bytes = source.read_bytes()
    passes = []
    with tempfile.TemporaryDirectory(prefix='lumbre-book-') as directory:
        work = Path(directory).resolve()
        if not work.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise RuntimeError('Unexpected temporary build directory')
        (work/source.name).write_bytes(source_bytes)
        for number in range(1, 4):
            start = time.perf_counter()
            result = subprocess.run([executable, '-interaction=nonstopmode', '-halt-on-error', source.name],
                                    cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace')
            passes.append({'pass': number, 'returncode': result.returncode, 'seconds': time.perf_counter()-start})
            print(f'LaTeX pass {number}: exit {result.returncode}', flush=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-12000:])
        log = (work/source.with_suffix('.log').name).read_text(encoding='utf-8', errors='replace')
        unresolved = ('There were undefined references', 'There were undefined citations',
                      'Label(s) may have changed', 'Rerun to get cross-references right',
                      'There were multiply-defined labels')
        if any(message in log for message in unresolved):
            raise RuntimeError('LaTeX references did not converge after three passes')
        if 'Overfull' in log:
            details = '\n'.join(re.findall(r'Overfull[^\n]*(?:\n[^\n]*){0,3}', log))
            raise RuntimeError('LaTeX reported overflowing boxes; revise the layout before publishing\n'+details)
        pdf = work/source.with_suffix('.pdf').name
        target = source.with_suffix('.pdf')
        shutil.copyfile(pdf, target)
        summary = {'passes': passes, 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
                   'pdf_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                   'engine': subprocess.check_output([executable, '--version'], text=True).splitlines()[0],
                   'references_converged': True, 'overfull_boxes': log.count('Overfull'),
                   'underfull_boxes': log.count('Underfull'),
                   'duplicate_page_destinations': log.count('destination with the same identifier')}
    summary['temporary_directory_removed'] = not work.exists()
    if report:
        report = Path(report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Regenerate the book; optionally compile in a temporary directory.')
    parser.add_argument('--pdf', action='store_true', help='Compile three LaTeX passes and clean all temporary files')
    parser.add_argument('--engine', default='pdflatex', help='LaTeX executable (default: pdflatex)')
    parser.add_argument('--report', type=Path, help='Optional structured build report in JSON')
    args = parser.parse_args()
    build()
    if args.pdf:
        compile_pdf(args.engine, args.report)
