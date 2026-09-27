# SolSight Presentation Guide

## Overview

This guide helps you compile, present, and customize the **SolSight LaTeX Powerdot presentation** for your Solder Defect Detection System project.

---

## 📋 Presentation Structure

### 17 Slides Organized into Key Sections:

1. **Title Slide** - Project branding
2. **Problem Statement** - Why automated defect detection matters
3. **Solution Overview** - Core value proposition
4. **Architecture Design** - Technical foundation (Fully Convolutional CAE)
5. **Multi-Scale Training** - Resolution-agnostic approach
6. **Synthetic Benchmark Results** - Model performance metrics
7. **Real-World PCBA Results** - Zero-shot domain transfer validation
8. **Failure Boundaries** - Known limitations and safe operating ranges
9. **Deployment Workflow** - 3-step production integration
10. **Comparative Advantages** - Why SolSight vs competitors
11. **Technical Stack** - Dependencies and frameworks
12. **Quick Start (3 Steps)** - Installation and verification
13. **Key Metrics Summary** - Performance overview
14. **Use Cases** - Real-world applications
15. **Future Roadmap** - Enhancement areas
16. **Contact & Resources** - Links and documentation
17. **Closing Slide** - Call to action

---

## 🚀 Compilation Instructions

### Requirements

- **TeX Distribution**: Install MiKTeX (Windows), MacTeX (macOS), or TeX Live (Linux)
- **Required Packages**: `powerdot`, `tikz`, `xcolor`, `booktabs`, `hyperref`

### Step 1: Install LaTeX

**On Ubuntu/Debian:**
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

**On macOS (via Homebrew):**
```bash
brew install mactex
```

**On Windows:**
Download and install [MiKTeX](https://miktex.org/download)

### Step 2: Compile to PDF

Navigate to the project directory and run:

```bash
cd /home/keedastro/SolderDefectDetectionSystem/
pdflatex -interaction=nonstopmode presentation.tex
```

Or use a one-liner with automatic cleanup:

```bash
pdflatex -interaction=nonstopmode presentation.tex && \
rm -f presentation.aux presentation.log presentation.nav presentation.out presentation.snm presentation.toc && \
echo "✓ presentation.pdf created successfully"
```

**Output:** `presentation.pdf` (ready for viewing and presenting)

### Step 3: View & Present

- **View:** `evince presentation.pdf` (Linux) or `open presentation.pdf` (macOS)
- **Present:** Use PDF viewer in fullscreen mode (e.g., Adobe Reader, Preview)
  - Press `F5` or `Ctrl+Shift+F` for presentation mode
  - Use arrow keys or mouse to navigate

---

## 🎨 Customization Guide

### 1. **Colors**

Modify the color scheme in lines 17–20:

```latex
\definecolor{primary}{RGB}{31, 78, 121}      % Deep blue
\definecolor{accent}{RGB}{192, 0, 0}         % Red accent
\definecolor{light}{RGB}{242, 242, 242}      % Light gray
\definecolor{dark}{RGB}{51, 51, 51}          % Dark gray
```

**Popular Alternatives:**

| Scheme | Primary | Accent | Hex Primary |
|--------|---------|--------|-------------|
| Corporate | `31, 78, 121` | `192, 0, 0` | `#1F4E79` |
| Modern | `46, 116, 181` | `237, 125, 49` | `#2E74B5` |
| Tech | `0, 102, 204` | `255, 102, 0` | `#0066CC` |
| Bold | `25, 25, 112` | `220, 20, 60` | `#191970` |

### 2. **Header/Footer Text**

Edit lines 24–29 (Powerdot configuration):

```latex
\pdsetup{
    lf=SolSight,                    % Left footer
    rf=Solder Defect Detection System,  % Right footer
    cp=\theslide,                   % Center: slide number
    trans=Fade,                     % Transition effect (Fade, Wipe, Push, etc.)
    themecolor=primary
}
```

### 3. **Title and Subtitle**

Lines 33–36:

```latex
\title{SolSight}
\subtitle{Solder Defect Detection System}
\author{Your Name/Organization}
\date{\today}  % or specify: {September 27, 2026}
```

### 4. **Add Your Own Slides**

Template for a new slide:

```latex
%==================== SLIDE XX: YOUR SLIDE TITLE ====================
\begin{slide}
    \frametitle{\emphred{Your Slide Title}}
    
    \vspace{1cm}
    
    \begin{itemize}
        \item \emphhighlight{Main Point 1}
        \item \emphred{Important Detail}
        \item Regular content point
    \end{itemize}
\end{slide}
```

### 5. **Add Images**

To include images in a slide:

```latex
\begin{slide}
    \frametitle{\emphred{Slide with Image}}
    
    \begin{center}
        \includegraphics[width=0.7\textwidth]{path/to/image.png}
    \end{center}
    
    \small
    Caption goes here.
\end{slide}
```

---

## 📊 Transition Effects

Available in Powerdot:
- `Fade` (default)
- `Wipe` (left to right)
- `Push` (new slide pushes old off)
- `Replace` (instant switch)
- `GroundFlip` (3D flip)
- `UnCover` (slide out)

Change in line 27:

```latex
\pdsetup{
    ...
    trans=GroundFlip,  % Try different effects
    ...
}
```

---

## 🎯 Presentation Tips

### For a Pitch Deck Style:

1. **Keep It Visual**: Use the color scheme liberally (primary + accent)
2. **Less Text, More Impact**: Bullet points, not paragraphs
3. **Numbers Over Words**: Use metrics, benchmarks, and data
4. **Hierarchy Clear**: Use `\emphhighlight{}` for key points, `\emphred{}` for emphasis
5. **Smooth Transitions**: Maintain consistent pacing (45–60 seconds per slide)

### Typical Presentation Flow:

- **Slides 1–3**: Problem, Solution, Overview (2 minutes)
- **Slides 4–7**: Technical Depth (Architecture, Results, Validation) (3 minutes)
- **Slides 8–11**: Robustness, Deployment, Advantages (2 minutes)
- **Slides 12–14**: Implementation, Use Cases (1 minute)
- **Slides 15–17**: Future, Contact, Closing (1 minute)

**Total Duration:** ~9 minutes (adjust as needed)

---

## 🔧 Advanced Customization

### 1. **Add Footnotes**

```latex
Some content\footnote{This is a footnote that appears at the bottom}
```

### 2. **Two-Column Layout**

```latex
\begin{slide}
    \frametitle{\emphred{Two-Column Example}}
    
    \begin{columns}
        \column{0.5\textwidth}
        Left column content here
        
        \column{0.5\textwidth}
        Right column content here
    \end{columns}
\end{slide}
```

### 3. **Code Snippets with Syntax Highlighting**

The presentation already includes `listings` package. Customize:

```latex
\lstset{
    language=Python,
    basicstyle=\tiny\ttfamily,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{red},
    breaklines=true,
    backgroundcolor=\color{light}
}
```

### 4. **Custom Title Pages per Section**

```latex
\begin{slide}[toc=,bm=]{} % Blank slide
    \centering
    \vspace*{2cm}
    {\Huge\bfseries\color{primary} Section 2: Technical Details}
\end{slide}
```

---

## 📝 Editing Workflow

### Recommended Editors:

- **VSCode** with LaTeX Workshop extension
- **Overleaf.com** (online, no setup needed)
- **TeXstudio** (dedicated LaTeX IDE)
- **Vim** with vim-latex

### Quick Edit → Compile Cycle:

```bash
# Watch for changes and auto-compile
watch -n 2 'pdflatex -interaction=nonstopmode presentation.tex'
```

---

## 🐛 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `Undefined control sequence` | Missing LaTeX package; install via `texlive-latex-extra` |
| `! Package powerdot Error` | Ensure `powerdot` is installed: `tlmgr install powerdot` |
| `PDF not updating` | Delete `.aux`, `.log`, `.pdf` and recompile |
| Fonts look wrong | Update fonts: `tlmgr update --all` |
| Compilation too slow | Use `draft` mode: `pdflatex -draft` (for iteration) |

---

## 📦 Delivery Formats

### PDF (Recommended for Presentations)
```bash
pdflatex presentation.tex
```

### Handout (Print-Friendly)

Add to preamble:
```latex
\documentclass[handout]{powerdot}
```

Then compile as usual. This creates a print-optimized version.

### Quick Navigation PDF

Ensure bookmarks are enabled (already included via `hyperref`). PDFs will have clickable navigation.

---

## 🌐 Remote Presentation Mode

### Using Google Meet / Zoom:

1. Open `presentation.pdf` in fullscreen
2. Use PDF viewer's presentation mode (usually `F5`)
3. Share screen (select "PDF window")
4. Navigate with arrow keys or `Page Down`

**Pro Tip:** Use a second monitor for speaker notes (or print notes separately)

---

## 📞 Support & Resources

### LaTeX Resources:
- [Overleaf Tutorials](https://www.overleaf.com/learn)
- [Powerdot Documentation](https://www.ctan.org/pkg/powerdot)
- [TikZ Graphics Manual](https://www.ctan.org/pkg/pgf)

### For This Project:
- GitHub: [SolderDefectDetectionSystem](https://github.com/stanmaluki/SolderDefectDetectionSystem)
- Issues: Open an issue for presentation feedback
- Customize as needed for your audience

---

## ✅ Checklist Before Presenting

- [ ] Compile presentation successfully (no errors)
- [ ] Review all 17 slides for accuracy
- [ ] Check color scheme and theme consistency
- [ ] Test PDF in fullscreen presentation mode
- [ ] Verify hyperlinks work (if included)
- [ ] Backup PDF in multiple locations
- [ ] Test projection on target display (resolution/colors)
- [ ] Practice transitions and pacing
- [ ] Have backup printed notes
- [ ] Test audio/video if including multimedia

---

## 📄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-09-27 | Initial presentation with 17 slides |

---

**Enjoy your presentation! 🚀**

*For updates or customizations, refer to the inline LaTeX comments in `presentation.tex`.*
