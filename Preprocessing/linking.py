"""Match article posts on stoky.urza.cz to YouTube videos by title and date."""
from datetime import date as Date
from difflib import SequenceMatcher

import config as C
from textutils import norm_title


def _d(s):
    return Date.fromisoformat(s[:10]) if s else None


def _score(a: str, b: str, floor: float = 0.0) -> float:
    """Title similarity 0–1. Returns 0 early when it cannot reach `floor`."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    # Containment: one title is the other plus/minus a few words.
    contain = 0.9 * len(ta & tb) / min(len(ta), len(tb)) if min(len(ta), len(tb)) >= 3 else 0.0
    sm = SequenceMatcher(None, a, b)
    if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
        return contain
    return max(sm.ratio(), contain)


def _threshold(days):
    for max_days, score in C.LINK_TIERS:
        if max_days is None or (days is not None and days <= max_days):
            return score
    return 1.01


class VideoIndex:
    def __init__(self, videos: list[dict]):
        self.videos = [
            (v["doc_id"], norm_title(v["title"], C.TITLE_PREFIXES, C.TITLE_NOISE), _d(v["date"]))
            for v in videos
        ]

    def match(self, title: str, date: str):
        """Return (video_doc_id, score, method, days_apart) or None."""
        nt = norm_title(title, C.TITLE_PREFIXES, C.TITLE_NOISE)
        d = _d(date)
        best = None
        for vid, vt, vd in self.videos:
            days = abs((vd - d).days) if d and vd else None
            need = _threshold(days)
            if need > 1:
                continue
            s = _score(nt, vt, floor=need)
            if s >= need and (best is None or (s, -(days or 0)) > (best[1], -(best[3] or 0))):
                best = (vid, round(s, 3), "exact" if s == 1.0 else "fuzzy", days)
        return best

    def best_candidate(self, title: str, date: str):
        """Best match ignoring thresholds (for reporting near misses)."""
        nt = norm_title(title, C.TITLE_PREFIXES, C.TITLE_NOISE)
        d = _d(date)
        best = None
        for vid, vt, vd in self.videos:
            s = _score(nt, vt)
            days = abs((vd - d).days) if d and vd else None
            if best is None or s > best[1]:
                best = (vid, round(s, 3), days)
        return best
