"""Cumulative explained variance of PCA, as an inline SVG line chart (shared by the PCA/ICA reports)."""
import numpy as np

LEVELS = (0.5, 0.8, 0.9, 0.95)


def needed(cum, level):
    """Number of principal components needed to reach a share of the variance."""
    return int(np.searchsorted(cum, level) + 1)


def cumvar_svg(cum, color="var(--b0)"):
    """cum = np.cumsum(explained_variance_ratio_) over all components."""
    w, h, pl, pr, pb, pt = 640, 230, 40, 12, 30, 12
    nmax = min(len(cum), needed(cum, 0.99))
    X = lambda k: pl + (k - 1) / max(nmax - 1, 1) * (w - pl - pr)      # component k (1-based)
    Y = lambda v: h - pb - v * (h - pb - pt)
    out = []
    for t in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{pl}" x2="{w - pr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                   f'<text x="{pl - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="ax">{t:.0%}</text>')
    step = 10 ** max(int(np.log10(nmax)) - 1, 0) * (5 if nmax / 10 ** int(np.log10(nmax)) < 3 else 10)
    for k in range(step, nmax + 1, step):
        out.append(f'<text x="{X(k):.1f}" y="{h - pb + 16}" text-anchor="middle" class="ax">{k}</text>')
    out.append(f'<text x="{(pl + w - pr) / 2:.1f}" y="{h - 2}" text-anchor="middle" class="ax">number of principal components</text>')
    pts = " ".join(f"{X(k):.1f},{Y(cum[k - 1]):.1f}" for k in range(1, nmax + 1))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
    marks = [(2, cum[1], f"2 PCs (a PCA map): {cum[1]:.0%}")] + [(needed(cum, L), L, f"{L:.0%}: {needed(cum, L)} PCs") for L in LEVELS]
    for i, (k, v, txt) in enumerate(marks):
        x, y = X(k), Y(v)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y:.1f}" y2="{h - pb}" class="grid" stroke-dasharray="3 3"/>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"><title>{txt}</title></circle>')
        anchor = "start" if x < w * 0.7 else "end"
        dx = 6 if anchor == "start" else -6
        dy = -6 if i else -8
        out.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" class="lab">{txt}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Cumulative variance explained by principal components" '
            f'style="width:100%;max-width:{w}px;height:auto">{"".join(out)}'
            f'<line x1="{pl}" x2="{w - pr}" y1="{h - pb}" y2="{h - pb}" class="base"/></svg>')
