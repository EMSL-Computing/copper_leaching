"""Graphical abstract: two real pore networks, l_r illustrated, and Set B recovery vs l_r.

Left: inlet-side 4.8 x 2 mm of geometry seeds 484 (short l_r) and 423 (long l_r), with
chrysocolla next to inlet-connected flow highlighted and zoom insets whose arrows run
from chrysocolla voxels at the median distance to the nearest connected pore (arrow
length = l_r). Right: Set B solid-phase recovery versus l_r at matched flushing
(k_r = 1e-9) with the paper's exponential fit. Inlet-connected pore is taken from the
x = 0 face, which reproduces the manifest l_r for all 25 realizations.

Example:
    python src/plot_graphical_abstract.py
    python src/plot_graphical_abstract.py --out-dir manuscript/draft/figs
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import _repo  # noqa: F401 - path bootstrap
from _repo import REPO_ROOT
from svg_render import render_svg
import plot_accessibility_ensemble as P
from compute_reactive_accessibility_length import connected_pore_from_inlet
from scipy import ndimage as ndi

DEFAULT_OUT = REPO_ROOT / 'accessibility_ensemble/analysis/combined'


def build_svg(ROOT: Path = REPO_ROOT) -> str:
    """Return the figure as an SVG string."""
    W, H = 1600, 750
    NXC = 240  # inlet-side crop (240 of 400 voxels = 4.8 mm) so single-voxel rims stay visible
    FONT = "Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
    INK, MUTED, RULE = "#1e2430", "#4f5866", "#c3c9d2"
    COLORS = {"solid": "#cdd1d6", "iso": "#ffffff", "pore": "#9cc9e8", "chrys": "#4f9a52", "acc": "#e8892b"}
    ACID, CU = "#c0392b", "#1f6fb2"
    SEEDS = (484, 423)  # short and long l_r, highest and lowest Set B recovery

    tab = ROOT / "accessibility_ensemble" / "analysis" / "tables"
    b = pd.read_csv(tab / "matched_theta_metrics.csv")
    b = b[np.isclose(b.rate_mol_cm2_s, 1e-9, rtol=1e-6, atol=0)].set_index("geom_seed")

    S = []
    add = S.append


    def text(x, y, s, size=24, anchor="start", weight="normal", fill=INK, extra=""):
        add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}" {extra}>{s}</text>')


    def sym(base, lo, size):
        k = 0.7 * size
        return (f'<tspan font-style="italic">{base}</tspan><tspan font-size="{k:.1f}" dy="{0.28 * size:.1f}" '
                f'font-style="italic">{lo}</tspan><tspan dy="{-0.28 * size:.1f}"></tspan>')


    def network(seed):
        """Display-frame category RGB (rows = y top-down, cols = x), distance map, and nearest-pore indices.

        Stored masks are (x, y) = (400, 100); the inlet is x = 0 ("top" in array terms), which
        reproduces the manifest l_r for all 25 realizations.
        """
        pore, chrys, _ = P.load_realization_masks(seed)
        pore = pore > 0.5
        conn = connected_pore_from_inlet(pore, inlet="top")
        chrys = chrys & ~pore
        dist, (ix, iy) = ndi.distance_transform_edt(~conn, return_indices=True)
        acc = chrys & (dist <= 1.0)
        rgb = np.zeros(pore.shape + (3,), np.uint8)
        for mask, key in ((~pore, "solid"), (pore & ~conn, "iso"), (conn, "pore"), (chrys, "chrys"), (acc, "acc")):
            rgb[mask] = [int(COLORS[key][i:i + 2], 16) for i in (1, 3, 5)]
        flip = lambda a: np.transpose(a, (1, 0) + tuple(range(2, a.ndim)))[::-1]  # noqa: E731
        ny = pore.shape[1]
        return dict(rgb=flip(rgb), dist=flip(dist), chrys=flip(chrys), tx=flip(ix), ty=ny - 1 - flip(iy),
                    lr_vox=float(np.median(dist[chrys])))


    def png_b64(rgb, scale=1):
        img = Image.fromarray(rgb)
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


    def pick_window(net, wc, wr, n=4):
        """Window (r0, c0) and up to n chrysocolla voxels at about the median distance whose nearest
        connected-pore voxel lies inside the window."""
        d, ch = net["dist"], net["chrys"]
        med = net["lr_vox"]
        cand = np.argwhere(ch & (np.abs(d - med) <= 0.35))
        best = None
        for r, c in cand[:: max(1, len(cand) // 400)]:
            r0 = int(np.clip(r - wr // 2, 0, d.shape[0] - wr))
            c0 = int(np.clip(c - wc // 2, 20, NXC - wc))
            sel = []
            for rr, cc in cand:
                if not (r0 + 2 <= rr < r0 + wr - 2 and c0 + 2 <= cc < c0 + wc - 2):
                    continue
                tr, tc = net["ty"][rr, cc], net["tx"][rr, cc]
                if not (r0 <= tr < r0 + wr and c0 <= tc < c0 + wc):
                    continue
                if all(np.hypot(rr - a, cc - b2) >= 7 for a, b2, _, _ in sel):
                    sel.append((rr, cc, tr, tc))
                if len(sel) == n:
                    break
            if best is None or len(sel) > len(best[2]):
                best = (r0, c0, sel)
            if len(sel) == n:
                break
        return best


    def arrow(x1, y1, x2, y2, color, w=4):
        add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" '
            f'marker-end="url(#ah-{color[1:]})"/>')


    add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    add("<defs>")
    for c in (INK, ACID, CU):
        add(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="3.5" markerHeight="3.5" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    add("</defs>")

    # ---------------------------------------------------------------- left: real networks
    text(40, 56, "Same chrysocolla grade, same acid throughput", 30, weight="bold")
    IX, IW, IH = 96, 480, 200
    SX, SY = IW / NXC, IH / 100
    WC, WR, ZS = 32, 28, 7  # inset window (voxels) and zoom (px per voxel)
    JX = 712
    for k, seed in enumerate(SEEDS):
        y0 = 126 + k * 282
        net = network(seed)
        lr = b.loc[seed, "l_r_median_um"]
        rec = b.loc[seed, "solid_recovery_pct"]
        text(IX, y0 - 14, ("Mineral near connected flow" if k == 0 else "Mineral far from connected flow")
             + f",  {sym('l', 'r', 24)} = {lr:.0f} µm", 24, weight="bold")
        add(f'<image x="{IX}" y="{y0}" width="{IW}" height="{IH}" preserveAspectRatio="none" '
            f'style="image-rendering:pixelated" href="data:image/png;base64,{png_b64(net["rgb"][:, :NXC])}"/>')
        add(f'<rect x="{IX}" y="{y0}" width="{IW}" height="{IH}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
        cy = y0 + IH / 2
        arrow(IX - 62, cy, IX - 8, cy, ACID)
        text(IX - 64, cy - 14, "H⁺", 24, fill=ACID, weight="bold")
        arrow(IX + IW + 6, cy, IX + IW + 56, cy, CU)
        text(IX + IW + 8, cy - 14, "Cu²⁺", 24, fill=CU, weight="bold")
        text(IX + IW + 8, cy + 46, f"{rec:.0f}%", 36, weight="bold")
        # zoom inset with distance arrows d (median distance = l_r)
        r0, c0, sel = pick_window(net, WC, WR)
        wx, wy, ww, wh = IX + c0 * SX, y0 + r0 * SY, WC * SX, WR * SY
        jy = y0 + (IH - WR * ZS) / 2
        add(f'<rect x="{wx:.1f}" y="{wy:.1f}" width="{ww:.1f}" height="{wh:.1f}" fill="none" stroke="{INK}" stroke-width="2"/>')
        for ya, yb in ((wy, jy), (wy + wh, jy + WR * ZS)):
            add(f'<line x1="{wx + ww:.1f}" y1="{ya:.1f}" x2="{JX}" y2="{yb:.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 3"/>')
        crop = net["rgb"][r0:r0 + WR, c0:c0 + WC]
        add(f'<image x="{JX}" y="{jy}" width="{WC * ZS}" height="{WR * ZS}" style="image-rendering:pixelated" '
            f'href="data:image/png;base64,{png_b64(crop, ZS)}"/>')
        for rr, cc, tr, tc in sel:
            x1, y1 = JX + (cc - c0 + 0.5) * ZS, jy + (rr - r0 + 0.5) * ZS
            x2, y2 = JX + (tc - c0 + 0.5) * ZS, jy + (tr - r0 + 0.5) * ZS
            add(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="4" fill="{INK}"/>')
            add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="3" '
                f'marker-end="url(#ah-{INK[1:]})"/>')
        add(f'<rect x="{JX}" y="{jy}" width="{WC * ZS}" height="{WR * ZS}" fill="none" stroke="{INK}" stroke-width="2"/>')
        print(seed, "median d (um):", round(net["lr_vox"] * 20, 1), "arrows:", len(sel))
    text(JX + WC * ZS / 2, 104, "d: distance to flow", 20, anchor="middle", fill=MUTED)
    text(JX + WC * ZS / 2, 640, sym("l", "r", 20) + " = median d", 20, anchor="middle", fill=MUTED)
    # legend
    lx, ly = 40, 724
    for key, lab in (("pore", "connected pore"), ("chrys", "chrysocolla"), ("acc", "chrysocolla next to flow"), ("solid", "inert solid")):
        add(f'<rect x="{lx}" y="{ly - 17}" width="20" height="20" fill="{COLORS[key]}" stroke="{INK}" stroke-width="0.8"/>')
        text(lx + 28, ly, lab, 21, fill=MUTED)
        lx += 28 + 11.2 * len(lab) + 34

    # ---------------------------------------------------------------- right: recovery vs l_r
    add(f'<line x1="975" y1="40" x2="975" y2="{H - 40}" stroke="{RULE}" stroke-width="1.5"/>')
    text(1020, 56, "Farther from flow, less recovery", 30, weight="bold")
    x0p, x1p, y0p, y1p = 1110, 1560, 600, 120
    lx0, lx1, ry0, ry1 = 40, 120, 8, 18


    def px(v):
        return x0p + (v - lx0) / (lx1 - lx0) * (x1p - x0p)


    def py(v):
        return y0p + (v - ry0) / (ry1 - ry0) * (y1p - y0p)


    add(f'<rect x="{x0p}" y="{y1p}" width="{x1p - x0p}" height="{y0p - y1p}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    for v in (40, 60, 80, 100, 120):
        add(f'<line x1="{px(v)}" y1="{y0p}" x2="{px(v)}" y2="{y0p + 7}" stroke="{INK}" stroke-width="1.5"/>')
        text(px(v), y0p + 32, f"{v}", 21, anchor="middle", fill=MUTED)
    for v in (8, 10, 12, 14, 16, 18):
        add(f'<line x1="{x0p - 7}" y1="{py(v)}" x2="{x0p}" y2="{py(v)}" stroke="{INK}" stroke-width="1.5"/>')
        text(x0p - 12, py(v) + 7, f"{v}", 21, anchor="end", fill=MUTED)
    text((x0p + x1p) / 2, y0p + 72, "Reactive accessibility length (" + sym("l", "r", 24) + ", µm)", 24, anchor="middle")
    ym = (y0p + y1p) / 2
    text(1052, ym, "Cu recovery (%)", 24, anchor="middle", extra=f'transform="rotate(-90 1052 {ym})"')
    xx = np.linspace(lx0 + 2, lx1 - 2, 60)
    pts = " ".join(f"{px(v):.1f},{py(21.6 * np.exp(-v / 136)):.1f}" for v in xx)
    add(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2.2" stroke-dasharray="8 6"/>')
    for seed, row in b.iterrows():
        add(f'<circle cx="{px(row.l_r_median_um):.1f}" cy="{py(row.solid_recovery_pct):.1f}" r="8" '
            f'fill="{COLORS["chrys"]}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1.2"/>')
    for seed in SEEDS:
        add(f'<circle cx="{px(b.loc[seed, "l_r_median_um"]):.1f}" cy="{py(b.loc[seed, "solid_recovery_pct"]):.1f}" r="14" '
            f'fill="none" stroke="{INK}" stroke-width="2.5"/>')
    text(x1p - 14, y1p + 34, "25 networks, matched flushing", 21, anchor="end", fill=MUTED)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="{FONT}" role="img" aria-label="Graphical abstract: two simulated pore networks with the same '
           f'chrysocolla grade and acid throughput recover 16% and 10% of copper; recovery falls as mineral distance to '
           f'connected flow increases.">\n' + "\n".join(S) + "\n</svg>\n")
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-render", action="store_true", help="write SVG only")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "graphical_abstract.svg"
    path.write_text(build_svg())
    print("Saved", path)
    if not args.no_render:
        render_svg(path, width=1600, height=750)


if __name__ == "__main__":
    main()
