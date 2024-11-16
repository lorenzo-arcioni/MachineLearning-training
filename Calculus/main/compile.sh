#!/bin/bash

# PdfLatex compile script
rm main.toc; bibtex main; pdflatex main.tex; #pdflatex main.tex