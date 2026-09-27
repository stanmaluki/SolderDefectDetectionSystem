# SolSight Presentation Package

Professional LaTeX Powerdot presentation for the **SolSight Solder Defect Detection System** project. This package includes a fully-designed 17-slide pitch-deck style presentation, ready for conference talks, investor pitches, or technical seminars.

---

## 📦 Package Contents

```
SolderDefectDetectionSystem/
├── presentation.tex                    ← Main LaTeX file (17 slides)
├── presentation.pdf                    ← Compiled PDF (generated after compilation)
├── PRESENTATION_README.md              ← This file
├── PRESENTATION_GUIDE.md               ← Detailed compilation & customization guide
├── PRESENTATION_PROMPT.md              ← AI prompt for slide refinement
├── QUICK_COMPILE.sh                    ← Linux/macOS compiler script
└── QUICK_COMPILE.bat                   ← Windows compiler script
```

---

## 🚀 Quick Start (60 seconds)

### Option 1: Linux / macOS

```bash
cd SolderDefectDetectionSystem/
./QUICK_COMPILE.sh
# Opens: presentation.pdf
```

### Option 2: Windows

```batch
cd SolderDefectDetectionSystem
QUICK_COMPILE.bat
REM Opens: presentation.pdf
```

### Option 3: Manual Compilation

```bash
pdflatex -interaction=nonstopmode presentation.tex
```

---

## ✨ Presentation Highlights

### 17 Engaging Slides

| # | Title | Content |
|---|-------|---------|
| 1 | Title Slide | Project branding & tagline |
| 2 | The Problem | Why automated AOI matters |
| 3 | Our Solution | Unsupervised anomaly detection pitch |
| 4 | Architecture Design | Fully convolutional CAE + DSSIM |
| 5 | Multi-Scale Training | Resolution-agnostic approach (16–160px) |
| 6 | Synthetic Benchmarks | AUROC 0.963 @ 64px, generalization to 32px |
| 7 | Real-World Validation | 0.879 AUROC on physical PCBAs (SolDef-AI) |
| 8 | Failure Boundaries | Known limitations & safe operating ranges |
| 9 | Deployment Workflow | 3-step production integration process |
| 10 | Competitive Advantages | Why SolSight beats traditional AOI & supervised DL |
| 11 | Technical Stack | Python, PyTorch, OpenCV, SSIM |
| 12 | Quick Start | 3-step installation & verification |
| 13 | Key Metrics | Training time, inference speed, model size |
| 14 | Use Cases | Manufacturing, QA, R&D applications |
| 15 | Future Roadmap | Multi-modal fusion, edge deployment, explainability |
| 16 | Contact & Resources | GitHub, docs, MIT license |
| 17 | Closing Slide | "Questions?" call-to-action |

### Design Features

✓ **Modern Pitch Deck Style**
- Clean, minimalist layout with strategic color use
- Primary: Deep blue (#1F4E79) | Accent: Red (#C00000)
- High visual hierarchy (emphasis on data, metrics, results)

✓ **Data-Driven Content**
- All metrics from actual benchmark results
- Comparison tables for competitive positioning
- Real-world validation (0.879 AUROC on physical hardware)

✓ **Production-Ready**
- 16:9 widescreen format
- Smooth fade transitions between slides
- Hyperlinked GitHub repository
- Print-friendly handout mode available

✓ **Customizable**
- Color scheme adjustable in seconds
- All 17 slides editable
- TikZ diagrams for custom graphics
- Code examples with syntax highlighting

---

## 📊 Benchmark Data Included

### Synthetic Tier Results
```
16px (Trained):     AUROC 0.940 | Cold Joint: 96% | Bridge: 84% | FPR: 10%
32px (Zero-Shot):   AUROC 0.862 | Cold Joint: 100% | Bridge: 44% | FPR: 15%
64px (Trained):     AUROC 0.963 | Cold Joint: 100% | Bridge: 90% | FPR: 0%
128px (Trained):    AUROC 0.850 | Cold Joint: 100% | Bridge: 88% | FPR: 0%
```

### Real-World Results (SolDef-AI Benchmark)
```
Real-World AUROC:         0.879
Normal Joint FPR:         0.0%
Overall Defect Recall:    100.0%
Model Size:               2.4 MB
Inference Speed:          <50ms per patch
GPU Memory:               280 MB (CPU-compatible)
```

---

## 🎯 Use Cases

### Conference Talks
- Technical program venues (SEMITECH, ECTC, IPC)
- Presentation duration: 9–10 minutes + Q&A

### Investor Pitches
- Emphasizes unsupervised learning advantage (faster deployment)
- Real-world validation (0.879 AUROC on physical hardware)
- Market positioning vs traditional AOI systems

### Technical Seminars
- Deep dive into architecture, training strategy, failure boundaries
- Deployment workflow for manufacturing integration

### Academic Presentations
- Peer-reviewed publication accompanying material
- Stress test results and empirical breaking points

---

## 🛠️ Customization Examples

### Change Color Scheme

Open `presentation.tex` and modify lines 17–20:

```latex
% Corporate blue + gold
\definecolor{primary}{RGB}{25, 46, 89}        % Navy
\definecolor{accent}{RGB}{255, 153, 0}        % Gold

% Tech company vibe
\definecolor{primary}{RGB}{0, 102, 204}       % Bright blue
\definecolor{accent}{RGB}{255, 102, 0}        % Orange
```

Then recompile:
```bash
./QUICK_COMPILE.sh  # or QUICK_COMPILE.bat on Windows
```

### Add a New Slide

After slide 10, add:

```latex
%==================== SLIDE 11: YOUR TITLE ====================
\begin{slide}
    \frametitle{\emphred{Your Slide Title}}
    
    \vspace{1cm}
    
    \begin{itemize}
        \item \emphhighlight{Key Point 1}
        \item \emphred{Important Finding}
        \item Regular bullet point
    \end{itemize}
\end{slide}
```

Then update slide numbering and recompile.

### Include Your Own Images

```latex
\begin{slide}
    \frametitle{\emphred{Model Performance}}
    
    \begin{center}
        \includegraphics[width=0.75\textwidth]{outputs/roc_curves.png}
    \end{center}
    
    \small
    ROC curves for all four training tiers.
\end{slide}
```

**See PRESENTATION_GUIDE.md for 20+ more customization examples.**

---

## 📋 Requirements

### Minimum LaTeX Setup
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: `powerdot`, `tikz`, `xcolor`, `booktabs`

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

**macOS (with Homebrew):**
```bash
brew install mactex
```

**Windows:**
Download & install [MiKTeX](https://miktex.org/download)

### Verification

```bash
pdflatex -version  # Should return version info
```

---

## 💻 Viewing & Presenting

### View Presentation
```bash
# Linux
evince presentation.pdf

# macOS
open presentation.pdf

# Windows
start presentation.pdf
```

### Present in Fullscreen
1. Open `presentation.pdf` in your PDF viewer
2. Press **F5** (most viewers) or **Ctrl+Shift+F**
3. Use arrow keys or mouse to navigate

### Remote Presentation
- Share your screen in Google Meet, Zoom, or Teams
- Use PDF fullscreen mode
- Use a second monitor for speaker notes (optional)

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| **PRESENTATION_README.md** | This file — overview & quick start |
| **PRESENTATION_GUIDE.md** | Detailed guide: compilation, customization, troubleshooting |
| **PRESENTATION_PROMPT.md** | Master prompt for AI-generated slide refinements |
| **presentation.tex** | LaTeX source code (17 slides) |

---

## 🎨 Style Guide

### Color Usage
- **Primary (Blue)**: Headers, major section dividers
- **Accent (Red)**: Emphasis, important metrics, call-to-action
- **Light Gray**: Backgrounds for code/tables
- **Dark Gray**: Body text

### Typography
- **Headers**: Large/Huge, bold, primary color
- **Body**: normalsize, dark gray
- **Emphasis**: Use `\emphred{}` for accent or `\emphhighlight{}` for primary

### Layout Rules
- ✓ Whitespace > Clutter
- ✓ Data tables over paragraphs
- ✓ 3–5 bullet points per slide maximum
- ✓ One idea per slide
- ✓ Consistent spacing (1cm vertical gaps)

---

## 🚨 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `Command \powerdot not found` | Run: `tlmgr install powerdot` |
| PDF won't compile | Delete `.aux`, `.log` files and try again |
| Colors don't show | Ensure `xcolor` package is installed |
| Slide numbers wrong after edits | LaTeX auto-updates; recompile |
| Fonts look blurry | Update fonts: `tlmgr update --all` |

**For more troubleshooting, see PRESENTATION_GUIDE.md § Troubleshooting.**

---

## 📝 Editing Workflow

### Recommended Editors
- **VSCode** with [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) extension
- **Overleaf** (free, browser-based, no setup)
- **TeXstudio** (dedicated LaTeX IDE)
- **Vim** with vim-latex plugin

### Edit → Compile Cycle
```bash
# Terminal: auto-compile on file save
watch -n 2 'pdflatex -interaction=nonstopmode presentation.tex'

# Then view in another terminal
evince presentation.pdf
```

---

## ✅ Pre-Presentation Checklist

- [ ] Presentation compiled successfully (no errors)
- [ ] Reviewed all 17 slides for accuracy
- [ ] Verified metrics match README.md and actual results
- [ ] Tested PDF in fullscreen presentation mode
- [ ] Confirmed colors look good on target display
- [ ] Checked hyperlinks (GitHub, documentation)
- [ ] Backed up PDF in multiple locations
- [ ] Practiced slide transitions and pacing (9–10 min total)
- [ ] Have printed speaker notes or slide outline
- [ ] Tested on actual projection setup if possible

---

## 🔄 Version Control

This presentation package is under version control. To track changes:

```bash
git add presentation.tex PRESENTATION_*.md QUICK_COMPILE.*
git commit -m "refactor: update presentation slides for v2 metrics"
git push
```

---

## 📞 Support & Resources

### LaTeX Documentation
- [Overleaf LaTeX Tutorials](https://www.overleaf.com/learn) — Excellent beginner resource
- [Powerdot Manual](https://www.ctan.org/pkg/powerdot) — Official documentation
- [TikZ Graphics](https://www.ctan.org/pkg/pgf) — Diagram creation guide

### For This Project
- **GitHub**: [SolderDefectDetectionSystem](https://github.com/stanmaluki/SolderDefectDetectionSystem)
- **Issues**: Open a GitHub issue for presentation feedback
- **Documentation**: See `docs/` folder in repository

### AI-Powered Refinement
Use `PRESENTATION_PROMPT.md` with ChatGPT, Claude, or similar to:
- Generate new slides
- Refine existing slides
- Adapt to different audiences
- Convert to other formats (PowerPoint, Beamer, etc.)

---

## 📄 License

This presentation is part of the **SolderDefectDetectionSystem** project.

**License**: MIT (see repository for full text)

**You can**:
- ✓ Use for presentations, conferences, and teaching
- ✓ Modify slides and colors
- ✓ Share with colleagues and collaborators
- ✓ Create derivative works

**You should**:
- ✓ Credit the original project in presentation intro
- ✓ Keep MIT license attribution
- ✓ Link to GitHub repository

---

## 🎯 Next Steps

1. **Compile Now**: Run `./QUICK_COMPILE.sh` to generate `presentation.pdf`
2. **Customize**: Edit `presentation.tex` to match your branding
3. **Practice**: Review slides and time your presentation (aim for 9–10 min)
4. **Present**: Open PDF in fullscreen (F5 key) and share with your audience

**Happy presenting! 🚀**

---

## 📊 Presentation Metrics

- **Total Slides**: 17
- **Recommended Duration**: 9–10 minutes (slides only)
- **With Q&A**: 15–20 minutes
- **Audience**: Technical, investor, or academic
- **Format**: 16:9 widescreen
- **File Size**: ~2–5 MB (depending on included images)

---

## 🏆 Tips for Effective Delivery

### Opening (1 min)
- Start with the problem statement (emotional hook)
- Show real-world defect images if possible

### Middle (6–7 min)
- Walk through architecture and results quickly
- Emphasize the 0.879 AUROC real-world validation
- Highlight no-labeling advantage vs competitors

### Closing (1–2 min)
- Summarize key metrics (AUROC, speed, cost)
- Ask for questions
- Offer to demo the system if time permits

### General Tips
- ✓ Pause for questions after major sections
- ✓ Use presenter notes if available
- ✓ Point to specific numbers and graphs
- ✓ Vary pacing and tone to maintain engagement
- ✓ Have a backup plan (printed slides, internet link)

---

**Created**: September 27, 2026  
**Version**: 1.0  
**Last Updated**: September 27, 2026  
**Maintained by**: SolSight Project Team
