#!/bin/bash

# ============================================================================
# SolSight Presentation Quick Compiler
# ============================================================================
# This script compiles the LaTeX presentation and cleans up temporary files
# Usage: ./QUICK_COMPILE.sh
# ============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_FILE="presentation.tex"
OUTPUT_FILE="presentation.pdf"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 SolSight LaTeX Presentation Compiler"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if LaTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo "❌ ERROR: pdflatex not found!"
    echo ""
    echo "Install LaTeX with one of these commands:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra"
    echo "  macOS: brew install mactex"
    echo "  Windows: Download from https://miktex.org/download"
    exit 1
fi

# Check if presentation.tex exists
if [ ! -f "$TEX_FILE" ]; then
    echo "❌ ERROR: $TEX_FILE not found in $PROJECT_DIR"
    exit 1
fi

echo "✓ LaTeX environment found"
echo "✓ Input file: $TEX_FILE"
echo ""

# Compilation
echo "📝 Compiling LaTeX presentation..."
pdflatex -interaction=nonstopmode -output-directory=. "$TEX_FILE" > /dev/null 2>&1 || {
    echo "❌ Compilation failed!"
    echo "Run 'pdflatex -interaction=nonstopmode presentation.tex' for detailed errors"
    exit 1
}

echo "✓ Compilation complete"

# Cleanup temporary files
echo "🧹 Cleaning up temporary files..."
rm -f presentation.aux presentation.log presentation.nav presentation.out \
      presentation.snm presentation.toc presentation.dvi 2>/dev/null || true

echo "✓ Cleanup complete"

# Check if PDF was created
if [ -f "$OUTPUT_FILE" ]; then
    PDF_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ SUCCESS: $OUTPUT_FILE created ($PDF_SIZE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📊 Next Steps:"
    echo "  View:    evince $OUTPUT_FILE"
    echo "  Print:   lp $OUTPUT_FILE"
    echo "  Edit:    nano presentation.tex (then re-run this script)"
    echo "  Present: Open $OUTPUT_FILE in fullscreen (F5 key)"
    echo ""
else
    echo "❌ ERROR: PDF file was not created"
    exit 1
fi
