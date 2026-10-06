#!/usr/bin/env python3
"""Build the current manuscript and synchronized main/SI PDFs."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import fitz
ROOT=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='dose-combination-bo-paper-') as build_dir:
    build=Path(build_dir)
    for _ in range(4):
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',
                        f'-output-directory={build}','PAPER_revised_singlefile.tex'],
                       cwd=ROOT,check=True)
        log=(build/'PAPER_revised_singlefile.log').read_text()
        if not any(message in log for message in (
                'undefined references', 'undefined citations',
                'Rerun to get cross-references right',
                'Rerun to get outlines right')):
            break
    else:
        raise RuntimeError('References did not stabilize after four passes; inspect the build log.')
    with fitz.open(build/'PAPER_revised_singlefile.pdf') as doc:
        starts=[i for i,p in enumerate(doc) if p.get_text().lstrip().startswith('A\nImplementation and simulation details')]
        if len(starts)!=1:
            raise RuntimeError('Cannot unambiguously identify Appendix A; inspect PDF before splitting.')
        cut=starts[0]
        for lo,hi,name in [(0,cut-1,'Main_Manuscript.pdf'),
                           (cut,len(doc)-1,'Supporting_Information.pdf')]:
            with fitz.open() as out:
                out.insert_pdf(doc,from_page=lo,to_page=hi)
                out.save(build/name)
            shutil.copy2(build/name,ROOT.parent/name)
        shutil.copy2(build/'PAPER_revised_singlefile.pdf',ROOT.parent/'Complete_Manuscript.pdf')
        print(f'Combined: {len(doc)} pages; main: {cut}; SI: {len(doc)-cut}')
