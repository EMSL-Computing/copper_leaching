"""Fig. 2: simulation-scenario schematic (Set A factorial, rescaling, Set B, protocol).

Hand-built SVG from the archived Set A / Set B metric tables; rendered to PNG (2x) and
PDF with headless Chromium. Canvas is 1600 px wide for a ~7.5 in figure, so the
smallest text (20 px) prints at ~6.8 pt.

Example:
    python src/plot_fig02_scenario_schematic.py
    python src/plot_fig02_scenario_schematic.py --out-dir manuscript/draft/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_hex

import _repo  # noqa: F401 - path bootstrap
from _repo import REPO_ROOT
from svg_render import render_svg

DEFAULT_OUT = REPO_ROOT / 'accessibility_ensemble/analysis/combined'


def build_svg(TABLES: Path = REPO_ROOT / "accessibility_ensemble" / "analysis" / "tables") -> str:
    """Return the figure as an SVG string."""
    W, H = 1600, 1070
    FONT = "Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
    INK, MUTED, RULE = "#1e2430", "#4f5866", "#c3c9d2"
    A, B = "#1f4e8c", "#2c6e49"
    B_T = "#edf5ef"
    ACID, BG = "#d9822b", "#dfe3e8"
    FS, FL, FT = 20, 22, 26  # small, label, panel title

    a = pd.read_csv(TABLES / "factorial_enriched.csv")
    b = pd.read_csv(TABLES / "matched_theta_metrics.csv")
    ref = a[(a.dP_Pa == 1.0) & np.isclose(a.rate_mol_cm2_s, 1e-9, rtol=1e-6, atol=0)]
    ref = ref.sort_values(["l_r_median_um", "geom_seed"]).reset_index(drop=True)
    lr = ref.l_r_median_um.to_numpy()
    theta_ref = ref.theta_48h.to_numpy()
    theta_tgt = float(b.theta_target_48h.iloc[0])
    dp_lo, dp_hi = b.dP_Pa.min(), b.dP_Pa.max()
    th_med = a.groupby("dP_Pa").theta_48h.median()
    cmap = colormaps["viridis"]
    col = [to_hex(cmap((v - lr.min()) / (lr.max() - lr.min()))) for v in lr]

    S = []
    add = S.append


    def text(x, y, s, size=FS, anchor="start", weight="normal", fill=INK, style="normal", extra=""):
        add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" {extra}>{s}</text>')


    def sub(base, lo, size):
        """Base text with a subscript; baseline restored afterwards."""
        k = 0.7 * size
        return f'{base}<tspan font-size="{k:.1f}" dy="{0.28 * size:.1f}">{lo}</tspan><tspan dy="{-0.28 * size:.1f}"> </tspan>'


    def subsup(base, lo, hi, size):
        """Base text with stacked superscript and subscript."""
        k = 0.66 * size
        return (f'{base}<tspan font-size="{k:.1f}" dy="{-0.42 * size:.1f}">{hi}</tspan>'
                f'<tspan font-size="{k:.1f}" dx="{-0.6 * k * len(hi):.1f}" dy="{0.72 * size:.1f}">{lo}</tspan>'
                f'<tspan dx="{max(0.0, (len(hi) - len(lo)) * 0.6 * k):.1f}" dy="{-0.30 * size:.1f}"> </tspan>')


    def theta(size, sup=None):
        return subsup("Θ", "acc", sup, size) if sup else sub("Θ", "acc", size)


    def it(s):
        return f'<tspan font-style="italic">{s}</tspan>'


    def kr(size):
        return sub(it("k"), it("r"), size)


    def dpx(size, lo):
        return sub(it("Δp"), lo, size)


    DP = it("Δp")


    def dots(cx, cy, r=7.6, step=21.0):
        """5 x 5 glyph: one dot per realization, row-major in l_r order."""
        x0, y0 = cx - 2 * step, cy - 2 * step
        for i in range(25):
            add(f'<circle cx="{x0 + (i % 5) * step:.1f}" cy="{y0 + (i // 5) * step:.1f}" r="{r}" '
                f'fill="{col[i]}" stroke="#ffffff" stroke-width="1.2"/>')


    def arrow(x1, y1, x2, y2, color=INK, w=2.0):
        add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" '
            f'marker-end="url(#ah-{color[1:]})"/>')


    def panel(x, y, w, h, color, letter, title, subtitle):
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        add(f'<path d="M{x},{y + 10} a10,10 0 0 1 10,-10 h{w - 20} a10,10 0 0 1 10,10 v66 h{-w} z" fill="{color}"/>')
        text(x + 18, y + 33, f"({letter}) {title}", FT, weight="bold", fill="#ffffff")
        text(x + 18, y + 62, subtitle, FS, fill="#ffffff")


    # ---------------------------------------------------------------- defs
    add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    add("<defs>")
    for c in (INK, A, B, MUTED):
        add(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    add('<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<rect width="8" height="8" fill="{ACID}" fill-opacity="0.18"/><line x1="0" y1="0" x2="0" y2="8" '
        f'stroke="{ACID}" stroke-width="2.4"/></pattern>')
    add("</defs>")

    # ---------------------------------------------------------------- shared input strip
    sq, gap = 26, 4
    x0 = 590
    xe = x0 + 25 * (sq + gap) - gap
    text(30, 44, "25 screened pore networks", FL, weight="bold")
    text(30, 74, "8 × 2 mm, near-constant φ", FS, fill=MUTED)
    for i in range(25):
        add(f'<rect x="{x0 + i * (sq + gap):.1f}" y="24" width="{sq}" height="{sq}" rx="4" fill="{col[i]}"/>')
    text(x0 + 30, 84, f"{lr.min():.0f} µm", FS, fill=MUTED)
    text(xe - 30, 84, f"{lr.max():.0f} µm", FS, anchor="end", fill=MUTED)
    text((x0 + xe) / 2, 84, sub(it("l"), it("r"), FS) + "increasing →", FS, anchor="middle", fill=MUTED)
    arrow(x0 + 13, 104, x0 + 13, 146, MUTED, 1.8)
    arrow(xe - 13, 104, xe - 13, 146, MUTED, 1.8)

    # ---------------------------------------------------------------- (a) Set A
    PY, PH = 150, 690
    panel(20, PY, 700, PH, A, "a", "Set A: factorial", "RQ1 · n = 225 = 3 " + DP + " × 3 " + kr(FS) + "× 25")
    CW, CH, G = 150, 128, 14
    GX, GY = 232, PY + 176
    rows = [1e-9, 5e-10, 1e-10]
    rlab = ["10⁻⁹", "5 × 10⁻¹⁰", "10⁻¹⁰"]
    dps = [0.001, 0.1, 1.0]
    GWD, GHT = 3 * CW + 2 * G, 3 * CH + 2 * G
    text(GX + GWD / 2, PY + 118, DP + " (Pa)", FL, anchor="middle", weight="bold", fill=A)
    text(50, GY + GHT / 2, kr(FL) + "(mol cm⁻² s⁻¹)", FL, anchor="middle", weight="bold", fill=A,
         extra=f'transform="rotate(-90 50 {GY + GHT / 2})"')
    text(GX - 14, GY + GHT + 34, "median " + theta(FS), FS, anchor="end", fill=MUTED)
    shade = ["#f6f8fc", "#e9eff8", "#d9e4f3"]
    for j, dp in enumerate(dps):
        cx = GX + j * (CW + G) + CW / 2
        text(cx, PY + 154, f"{dp:g}", FL, anchor="middle", fill=A)
        text(cx, GY + GHT + 34, f"{th_med[dp]:.3g} PV", FS, anchor="middle", fill=MUTED)
    for i, lab in enumerate(rlab):
        cy = GY + i * (CH + G) + CH / 2
        text(GX - 14, cy + 7, lab, FL, anchor="end", fill=A)
        for j in range(3):
            x, y = GX + j * (CW + G), GY + i * (CH + G)
            is_ref = i == 0 and j == 2
            add(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="8" fill="{shade[j]}" '
                f'stroke="{A}" stroke-width="{3 if is_ref else 1}" stroke-opacity="{1 if is_ref else 0.45}"/>')
            dots(x + CW / 2, y + CH / 2)
    sx, sy = GX + 2 * (CW + G) + CW - 16, GY + 3
    add(f'<path d="M{sx},{sy} l4.3,8.8 9.7,1.4 -7,6.8 1.7,9.6 -8.7,-4.6 -8.7,4.6 1.7,-9.6 -7,-6.8 9.7,-1.4 z" '
        f'fill="{A}" stroke="#ffffff" stroke-width="1"/>')

    # ---------------------------------------------------------------- (b) rescaling
    BX, BW = 738, 340
    panel(BX, PY, BW, PH, INK, "b", "Rescaling", DP + " set per realization")
    ylo, yhi = 5.0, 30.0
    Y0, Y1 = GY + GHT - 6, GY + 4


    def ty(v):
        return Y0 + (np.log(v) - np.log(ylo)) / (np.log(yhi) - np.log(ylo)) * (Y1 - Y0)


    AXX = BX + 82
    XL, XR = BX + 145, BX + 270
    add(f'<line x1="{AXX}" y1="{ty(ylo)}" x2="{AXX}" y2="{ty(yhi)}" stroke="{RULE}" stroke-width="1.4"/>')
    for v in (5, 10, 20, 30):
        add(f'<line x1="{AXX - 7}" y1="{ty(v):.1f}" x2="{AXX}" y2="{ty(v):.1f}" stroke="{MUTED}" stroke-width="1.4"/>')
        text(AXX - 12, ty(v) + 7, f"{v}", FS, anchor="end", fill=MUTED)
    ymid = (Y0 + Y1) / 2
    text(BX + 28, ymid, theta(FS) + "at 48 h (PV)", FS, anchor="middle",
         extra=f'transform="rotate(-90 {BX + 28} {ymid})"')
    add(f'<line x1="{AXX + 4}" y1="{ty(theta_tgt):.1f}" x2="{BX + BW - 14}" y2="{ty(theta_tgt):.1f}" '
        f'stroke="{INK}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    order = np.argsort(theta_ref)
    for i in order:
        add(f'<line x1="{XL}" y1="{ty(theta_ref[i]):.1f}" x2="{XR}" y2="{ty(theta_tgt):.1f}" stroke="{col[i]}" '
            f'stroke-width="1.5" stroke-opacity="0.75"/>')
    for i in order:
        add(f'<circle cx="{XL}" cy="{ty(theta_ref[i]):.1f}" r="6.2" fill="{col[i]}" stroke="#ffffff" stroke-width="1"/>')
    add(f'<circle cx="{XR}" cy="{ty(theta_tgt):.1f}" r="8.5" fill="{B}" stroke="#ffffff" stroke-width="1.4"/>')
    text(XL, PY + 118, "Set A", FL, anchor="middle", weight="bold", fill=A)
    text(XL, PY + 148, DP + " = 1 Pa", FS, anchor="middle", fill=MUTED)
    text(XR, PY + 118, "Set B", FL, anchor="middle", weight="bold", fill=B)
    text(XR, PY + 148, "rescaled", FS, anchor="middle", fill=MUTED)
    text(XR, ty(theta_tgt) - 16, f"{theta_tgt:.1f}", FS, anchor="middle", weight="bold")
    EQY = GY + GHT + 34
    text(BX + BW / 2, EQY, dpx(FS, "new") + "= " + dpx(FS, "ref") + "× " + theta(FS, "target")
         + "/ " + theta(FS, "ref"), FS, anchor="middle")
    text(BX + BW / 2, EQY + 30, dpx(FS, "new") + f"= {dp_lo:.2f}–{dp_hi:.2f} Pa", FS, anchor="middle", fill=MUTED)
    ry = GY + CH / 2
    arrow(GX + GWD + 4, ry, BX - 4, ry, A, 2.4)

    # ---------------------------------------------------------------- (c) Set B
    CX, CWB = 1096, 484
    arrow(BX + BW + 4, ry, CX - 4, ry, B, 2.4)
    panel(CX, PY, CWB, PH, B, "c", "Set B: matched flushing", "RQ2 · n = 75 = 3 " + kr(FS) + "× 25")
    text(CX + 22, PY + 118, "Same " + dpx(FL, "new") + "in all branches", FL, weight="bold", fill=B)
    text(CX + 22, PY + 148, "only " + kr(FS) + "changes", FS, fill=MUTED)
    for i, lab in enumerate(rlab):
        y = GY + i * (CH + G)
        add(f'<rect x="{CX + 18}" y="{y}" width="{CWB - 36}" height="{CH}" rx="8" fill="{B_T}" stroke="{B}" '
            f'stroke-width="1" stroke-opacity="0.5"/>')
        text(CX + 36, y + 56, kr(FL) + "= " + lab, FL, weight="bold", fill=B)
        text(CX + 36, y + 86, "mol cm⁻² s⁻¹", FS, fill=MUTED)
        dots(CX + 296, y + CH / 2)
        text(CX + CWB - 34, y + CH / 2 + 7, "n = 25", FS, anchor="end", fill=MUTED)
    text(CX + CWB / 2, EQY, theta(FS) + f"= {b.theta_48h.min():.3f}–{b.theta_48h.max():.3f} PV", FS,
         anchor="middle", fill=MUTED)
    text(CX + CWB / 2, EQY + 30, "in all 75 runs", FS, anchor="middle", fill=MUTED)

    # ---------------------------------------------------------------- (d) protocol
    DY = PY + PH + 22
    add(f'<rect x="20" y="{DY}" width="1560" height="{H - DY - 12}" rx="10" fill="#ffffff" stroke="{RULE}" stroke-width="1.6"/>')
    text(40, DY + 40, "(d) Common protocol, all 300 runs", FT, weight="bold")
    for k, s in enumerate(["Inlet: " + sub(it("p"), "in", FS) + "= " + sub(it("p"), "out", FS) + "+ " + DP,
                           "Outlet: " + sub(it("p"), "out", FS) + "= 101 325 Pa",
                           "Top, bottom: no flow"]):
        text(40, DY + 90 + k * 34, s, FS, fill=MUTED)
    T0, T1 = 450, 1530


    def tx(d):
        return T0 + d / 10 * (T1 - T0)


    TY = DY + 84
    add(f'<rect x="{tx(0)}" y="{TY}" width="{tx(2) - tx(0)}" height="44" rx="5" fill="url(#hatch)" stroke="{ACID}" stroke-width="1.6"/>')
    add(f'<rect x="{tx(2)}" y="{TY}" width="{tx(10) - tx(2)}" height="44" rx="5" fill="{BG}"/>')
    text((tx(0) + tx(2)) / 2, TY + 30, "Acid, pH 1", FL, anchor="middle", weight="bold", fill="#7a3f0a")
    text((tx(2) + tx(10)) / 2, TY + 30, "Background solution, pH 8", FL, anchor="middle")
    text(T0 - 14, TY + 30, "Inlet", FL, anchor="end", weight="bold")
    AX = TY + 72
    add(f'<line x1="{tx(0)}" y1="{AX}" x2="{tx(10)}" y2="{AX}" stroke="{INK}" stroke-width="1.5"/>')
    for d in range(0, 11, 2):
        add(f'<line x1="{tx(d)}" y1="{AX}" x2="{tx(d)}" y2="{AX + 7}" stroke="{INK}" stroke-width="1.5"/>')
        text(tx(d), AX + 30, f"{d}", FS, anchor="middle", fill=MUTED)
    text(tx(10) + 16, AX + 30, "d", FS, fill=MUTED)
    for d, lab, c in ((2, "48 h: " + theta(FS) + "evaluated", A),
                      (10, "10 d: " + sub(it("R"), "s", FS).replace("> </tspan>", "></tspan>") + ", " + sub(it("R"), "out", FS) + "evaluated", B)):
        add(f'<line x1="{tx(d)}" y1="{TY - 16}" x2="{tx(d)}" y2="{AX}" stroke="{c}" stroke-width="2.2" stroke-dasharray="5 3"/>')
        add(f'<path d="M{tx(d) - 8},{TY - 26} h16 l-8,10 z" fill="{c}"/>')
        text(tx(d) + (14 if d < 10 else -14), TY - 16, lab, FS, anchor="start" if d < 10 else "end", fill=c, weight="bold")

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="{FONT}" role="img" aria-label="Simulation scenario schematic: 25 realizations, '
           f'Set A factorial in pressure drop and rate constant, per-realization pressure rescaling, Set B matched '
           f'flushing at three rate constants, common inlet protocol.">\n' + "\n".join(S) + "\n</svg>\n")
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-render", action="store_true", help="write SVG only")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "fig02_scenario_schematic_v2.svg"
    path.write_text(build_svg())
    print("Saved", path)
    if not args.no_render:
        render_svg(path, width=1600, height=1070)


if __name__ == "__main__":
    main()
