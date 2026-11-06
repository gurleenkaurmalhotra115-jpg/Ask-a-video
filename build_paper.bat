@echo off
REM =========================================================================
REM One-click LaTeX Compilation Script for VidQuery Conference Paper (IEEEtran)
REM =========================================================================

echo Checking for pdflatex...
where pdflatex >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pdflatex was not found on your system PATH.
    echo Please compile paper.tex on Overleaf (https://www.overleaf.com/) using pdfLaTeX.
    pause
    exit /b 1
)

echo Compiling paper.tex (Pass 1)...
pdflatex -interaction=nonstopmode paper.tex > nul

echo Generating citations via BibTeX (if needed)...
bibtex paper > nul 2>nul

echo Compiling paper.tex (Pass 2 - Cross-references)...
pdflatex -interaction=nonstopmode paper.tex > nul

echo Compiling paper.tex (Pass 3 - Final Layout)...
pdflatex -interaction=nonstopmode paper.tex > nul

if exist paper.pdf (
    echo.
    echo =========================================================================
    echo [SUCCESS] paper.pdf has been generated successfully!
    echo =========================================================================
) else (
    echo.
    echo [WARNING] Compilation completed, but paper.pdf was not detected. 
    echo Check paper.log for details or compile online via Overleaf.
)

pause
