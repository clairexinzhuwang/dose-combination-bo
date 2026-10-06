Editable manuscript source

PAPER_revised_singlefile.tex is the current single source for the main article
and supporting information. The figure PDFs are unchanged vector originals.

With a LaTeX distribution (pdflatex) and PyMuPDF installed, run:
    python build_manuscript.py

This rebuilds Complete_Manuscript.pdf, Main_Manuscript.pdf, and
Supporting_Information.pdf one folder above, using a temporary build directory.
The SI retains
the complete manuscript's numbering and refers to its main reference list.

Revision dated 6 October 2026. The source and PDFs contain the same revised
text, without revision highlighting.
