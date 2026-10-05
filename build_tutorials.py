"""Build standalone tutorials using the typography and parser of the main book."""
import argparse
from pathlib import Path
import build_book

ROOT=Path(__file__).resolve().parent
TITLES={'cuda':('Tutorial de CUDA','Del índice al kernel comprobado','Tutorial_CUDA_Desde_Cero_2026'),
        'tinygrad':('Tutorial de tinygrad','Del tensor al programa ejecutado','Tutorial_Tinygrad_Desde_Cero_2026')}

def build(name):
    title,subtitle,stem=TITLES[name]
    preamble=(ROOT/'preamble.tex').read_text(encoding='utf-8').split('\\begin{document}')[0]
    preamble=preamble.replace('Compiladores de ML: desde cero',title)
    preamble=preamble.replace('pdftitle={Compiladores para aprendizaje automático: desde cero}',
                              'pdftitle={'+title+'}')
    tex=preamble+'\\begin{document}\n\\hypersetup{pageanchor=false}\n'
    tex+=r'\begin{titlepage}\centering\vspace*{1.5cm}'+'\n'
    tex+=r'{\large CUADERNOS DE COMPILADORES PARA APRENDIZAJE AUTOMÁTICO\par}\vspace{1.4cm}'+'\n'
    tex+=r'{\Huge\bfseries '+title+r'\par}\vspace{.8cm}{\Large\color{azul}'+subtitle+r'\par}\vspace{1cm}'+'\n'
    tex+=r'\begin{minipage}{.85\textwidth}Material para quien ha completado \emph{Compiladores de ML desde cero}. Explicaciones paso a paso, programas completos, ejercicios con solución y validación local en CPU o GPU según la práctica.\end{minipage}\vfill'+'\n'
    tex+=r'{\large Kripta Studios\par}\vspace{.4cm}Edición del 5 de octubre de 2026\par\vspace{.5cm}'+'\n'
    tex+=r'{\small Material didáctico independiente. Los programas originales y las evidencias acompañan al PDF en el repositorio. Las APIs externas se fijan a las versiones indicadas.\par}\end{titlepage}'+'\n'
    tex+=r'\pagenumbering{arabic}\hypersetup{pageanchor=true}'+'\n'
    for part in ['tutorial.md','syllabus.md','results.md']:
        tex+=build_book.parse((ROOT/'tutorials'/name/part).read_text(encoding='utf-8'))
    tex+=r'\clearpage\appendix\chapter{Código completo de las prácticas}'+'\n'
    files=sorted((ROOT/'tutorials'/name).glob('*.cu'))+sorted((ROOT/'tutorials'/name).glob('*.cuh')) if name=='cuda' else [ROOT/'tutorials/tinygrad/labs.py']
    if name=='cuda': files=[ROOT/'tutorials/cuda/common.cuh']+sorted((ROOT/'tutorials/cuda').glob('*.cu'))+[ROOT/'tutorials/validate_cuda.py']
    for path in files:
        relative=path.relative_to(ROOT).as_posix()
        tex+='\\section{'+build_book.esc(relative)+'}\n% EMBEDDED: '+relative+'\n'
        language='Python' if path.suffix=='.py' else 'C++'
        tex+='\\begin{lstlisting}[language='+language+']\n'+path.read_text(encoding='utf-8')+'\n\\end{lstlisting}\n'
    tex+=(ROOT/'tutorials'/name/'sources.tex').read_text(encoding='utf-8')
    tex+=r'\clearpage\renewcommand{\contentsname}{Índice detallado}\tableofcontents\end{document}'+'\n'
    target=ROOT/(stem+'.tex')
    target.write_text(tex,encoding='utf-8',newline='\n')
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pdf',action='store_true')
    parser.add_argument('--only',choices=TITLES)
    args=parser.parse_args()
    for name in [args.only] if args.only else TITLES:
        source=build(name)
        print(source.name,flush=True)
        if args.pdf:
            build_book.compile_pdf(report=ROOT/'tutorials/reports'/f'{name}_build.json',source=source)
