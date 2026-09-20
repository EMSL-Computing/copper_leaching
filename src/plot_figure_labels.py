"""Shared axis/title labels for accessibility_ensemble and ensemble transport figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes

# Manuscript main-text figure typography (Figs. 4--7).
MANUSCRIPT_AXIS_FS = 14.0
MANUSCRIPT_TICK_FS = 12.0
MANUSCRIPT_LEGEND_FS = 13.0
MANUSCRIPT_PANEL_FS = 14.0

LBL_CU_RECOVERY = "Cu Recovery [%]"
LBL_CU_RECOVERY_SOLID = r"$R_{\mathrm{s}}$ [%]"
LBL_CU_RECOVERY_OUTLET = r"$R_{\mathrm{out}}$ [%]"
LBL_THETA_FCU = r"$\Theta_{\mathrm{acc}}\,f_{\mathrm{Cu,acc}}$"
LBL_PE_R = r"$Pe_r$"
LBL_DA_R = r"$Da_r$"


def recovery_axis_label(metric: str) -> str:
    """Y-axis label for domain (solid-phase) or outlet recovery."""
    if metric == "outlet":
        return LBL_CU_RECOVERY_OUTLET
    return LBL_CU_RECOVERY_SOLID
LBL_DELTA_P = r"$\Delta p$ [Pa]"
LBL_K_R = r"$k_r$ [mol cm$^{-2}$ s$^{-1}$]"
LBL_LR = r"$l_r$ [$\mu$m]"
LBL_LR_MEDIAN = r"Median $l_r$ [$\mu$m]"
LBL_THETA_48H = r"$\Theta$ at 48 h [PV]"


def format_k_r_latex(rate: float) -> str:
    """LaTeX math string for a surface rate constant."""
    if abs(rate - 1e-10) < 1e-15:
        return r"$10^{-10}$"
    if abs(rate - 5e-10) < 1e-15:
        return r"$5\times10^{-10}$"
    if abs(rate - 1e-9) < 1e-15:
        return r"$10^{-9}$"
    exp = int(np.round(np.log10(rate)))
    mant = rate / (10**exp)
    if abs(mant - 1.0) < 1e-9:
        return rf"$10^{{{exp}}}$"
    return rf"${mant:g}\times10^{{{exp}}}$"


def dp_rate_title(dp: float, rate: float) -> str:
    """Panel title with LaTeX Delta p and k_r."""
    kr = format_k_r_latex(rate).strip("$")
    return rf"$\Delta p={dp:g}$ Pa, $k_r={kr}$"


def anchor_title(dp: float, rate: float) -> str:
    """Anchor-case title with Delta p and k_r."""
    kr = format_k_r_latex(rate).strip("$")
    return rf"(a) Anchor: $\Delta p={dp:g}$ Pa, $k_r={kr}$ mol cm$^{{-2}}$ s$^{{-1}}$"


def add_panel_label(
    ax: "matplotlib.axes.Axes",
    label: str,
    *,
    fontsize: float = MANUSCRIPT_PANEL_FS,
) -> None:
    """Lower-left panel tag, e.g. (a), (b)."""
    if not label:
        return
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )