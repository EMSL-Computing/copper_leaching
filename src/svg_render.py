"""Render a standalone SVG figure to PNG and PDF with headless Chromium.

Chromium is taken from $CHROME_HEADLESS, else from the Playwright cache
(~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell).

Example:
    from svg_render import render_svg
    render_svg(Path("figure.svg"), width=1600, height=1000)  # writes figure.png (2x) and figure.pdf
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from pathlib import Path


def find_chromium() -> str | None:
    """Return a headless Chromium executable, or None if none is found."""
    env = os.environ.get("CHROME_HEADLESS")
    if env and Path(env).is_file():
        return env
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")))
    return hits[-1] if hits else None


def render_svg(svg_path: Path, width: int, height: int, scale: int = 2) -> bool:
    """Write <stem>.png (scale x) and <stem>.pdf next to svg_path. Returns False if Chromium is missing."""
    chrome = find_chromium()
    if chrome is None:
        print("WARN: headless Chromium not found; set CHROME_HEADLESS. SVG only:", svg_path)
        return False
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>@page{{size:{width}px {height}px;margin:0}}'
            f'html,body{{margin:0;background:#fff}}svg{{display:block}}</style></head><body>'
            f'{Path(svg_path).read_text()}</body></html>')
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html)
        url = page.as_uri()
        stem = Path(svg_path).with_suffix("")
        base = [chrome, "--no-sandbox", "--hide-scrollbars"]
        subprocess.run(base + [f"--window-size={width},{height}", f"--force-device-scale-factor={scale}",
                               f"--screenshot={stem}.png", url], check=True, capture_output=True)
        subprocess.run(base + ["--no-pdf-header-footer", f"--print-to-pdf={stem}.pdf", url],
                       check=True, capture_output=True)
    print("Saved", f"{stem}.png", f"{stem}.pdf")
    return True
