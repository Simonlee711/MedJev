"""Shared figure style: Google brand colours, and Product Sans when its .ttf is placed in fonts/ (not bundled;
otherwise matplotlib's default sans-serif)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from medjev.config import ROOT

FONT_DIR = ROOT / "fonts"

# Google brand colours, plus Google's neutral greys for text, grid and reference lines.
BLUE, RED, YELLOW, GREEN = "#4285F4", "#EA4335", "#FBBC05", "#34A853"
GREY_TEXT, GREY_MID, GREY_LIGHT = "#5F6368", "#9AA0A6", "#DADCE0"
INK = "#202124"

MODEL_STYLE = {
    "medjev": {"label": "MedJev (zero-shot)", "short": "MedJev (zero-shot)", "color": BLUE, "marker": "o"},
    "medjev-recal": {"label": "MedJev + recalibration", "short": "MedJev + recalibration", "color": BLUE, "marker": "o", "hollow": True},
    "qwen3.5-9b-base": {"label": "Qwen3.5-9B base (zero-shot)", "short": "Qwen3.5-9B (zero-shot)", "color": GREY_TEXT, "marker": "v"},
    "qwen3.5-2b": {"label": "Qwen3.5-2B (zero-shot)", "short": "Qwen3.5-2B (zero-shot)", "color": GREEN, "marker": "D"},
    "medgemma-1.5-4b": {"label": "MedGemma-1.5-4B (zero-shot)", "short": "MedGemma-1.5-4B (zero-shot)", "color": YELLOW, "marker": "h"},
    "gemma-3-4b": {"label": "Gemma-3-4B (zero-shot)", "short": "Gemma-3-4B (zero-shot)", "color": YELLOW, "marker": "h", "hollow": True},
    "pythia-2.8b": {"label": "Pythia-2.8B (zero-shot)", "short": "Pythia-2.8B (zero-shot)", "color": GREY_MID, "marker": "P"},
    "xgboost": {"label": "XGBoost (supervised)", "short": "XGBoost (supervised)", "color": RED, "marker": "s"},
    "tfidf-lr": {"label": "TF-IDF + logistic regression", "short": "TF-IDF + LR (supervised)", "color": RED, "marker": "s", "hollow": True},
}

TEXT_WIDTH_IN = 6.5  # arxiv.sty text width
DPI = 300  # minimum resolution for every saved figure (PNG previews, and any rasterised element inside the PDFs)


def setup() -> None:
    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    have = any(f.name == "Product Sans" for f in font_manager.fontManager.ttflist)
    plt.rcParams.update({
        "font.family": "Product Sans" if have else "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.edgecolor": GREY_TEXT,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": GREY_TEXT,
        "ytick.color": GREY_TEXT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "text.color": INK,
        "pdf.fonttype": 42,  # embed TrueType so text stays text in the PDF
        "savefig.bbox": "tight",
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
        "savefig.pad_inches": 0.03,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    fig.savefig(path.with_suffix(".png"), dpi=DPI)
    plt.close(fig)

# One short, unique name per benchmark, used in every table, figure and the text.
TASK_NAMES = {
    "medqa": "MedQA", "usmle4": "USMLE", "nejm_medqa": "NEJM", "diagnosisarena": "DiagnosisArena",
    "ed_triage_admission": "ED Triage Admission", "ed_triage_disposition": "ED Triage Disposition", "meds_ed_disposition": "ED Admission at 2 h",
    "meds_mortality": "In-Hospital Mortality", "meds_long_los": "Length of Stay > 7 Days", "meds_readmission_30d": "30-Day Readmission",
}
