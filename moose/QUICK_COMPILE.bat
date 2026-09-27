@echo off
REM ============================================================================
REM SolSight Presentation Quick Compiler (Windows)
REM ============================================================================
REM This script compiles the LaTeX presentation and cleans up temporary files
REM Usage: Double-click or run: QUICK_COMPILE.bat
REM ============================================================================

setlocal enabledelayedexpansion

cls
echo ============================================================================
echo SolSight LaTeX Presentation Compiler
echo ============================================================================
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>nul
if errorlevel 1 (
    echo ERROR: pdflatex not found!
    echo.
    echo Install LaTeX using one of these methods:
    echo   1. MiKTeX: https://miktex.org/download
    echo   2. TeX Live: https://tug.org/texlive/
    echo.
    echo After installation, restart your terminal and try again.
    pause
    exit /b 1
)

REM Check if presentation.tex exists
if not exist "presentation.tex" (
    echo ERROR: presentation.tex not found in current directory!
    pause
    exit /b 1
)

echo. LaTeX environment found
echo. Input file: presentation.tex
echo.

REM Compilation
echo Compiling LaTeX presentation...
pdflatex -interaction=nonstopmode presentation.tex >nul 2>&1
if errorlevel 1 (
    echo ERROR: Compilation failed!
    echo Run this command for detailed errors:
    echo   pdflatex -interaction=nonstopmode presentation.tex
    pause
    exit /b 1
)

echo. Compilation complete

REM Cleanup temporary files
echo Cleaning up temporary files...
del /q presentation.aux presentation.log presentation.nav presentation.out ^
        presentation.snm presentation.toc presentation.dvi 2>nul

echo. Cleanup complete
echo.

REM Check if PDF was created
if exist "presentation.pdf" (
    for %%A in (presentation.pdf) do set PDF_SIZE=%%~zA
    echo ============================================================================
    echo SUCCESS: presentation.pdf created
    echo ============================================================================
    echo.
    echo Next Steps:
    echo   View:    presentation.pdf
    echo   Print:   Right-click presentation.pdf ^> Print
    echo   Edit:    Edit presentation.tex with your text editor
    echo   Present: Right-click presentation.pdf ^> Open with Adobe Reader
    echo            Press F5 to start presentation mode
    echo.
    echo For more information, see PRESENTATION_GUIDE.md
    echo.
) else (
    echo ERROR: PDF file was not created
    pause
    exit /b 1
)

pause
