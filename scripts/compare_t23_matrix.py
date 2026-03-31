#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics


def gt_map(labels_path: str = "data/eval/labels.json"):
    d = json.load(open(labels_path, "r", encoding="utf-8"))
    m = {}
    for it in d["items"]:
        t = it["task"]
        if t == "crosswalk_signal":
            m[it["id"]] = it["labels"]["walk_signal"]
        elif t == "stairs":
            m[it["id"]] = it["labels"]["stairs_present"]
        else:
            m[it["id"]] = it["labels"]["obstacle_present"]
    return m


def summarize(path: str, m):
    j = json.load(open(path, "r", encoding="utf-8"))
    rows = j["results"]
    n = len(rows)
    pred = [r["classification_label"] for r in rows]
    gt = [m[r["sample_id"]] for r in rows]

    corr = sum(1 for a, b in zip(pred, gt) if a == b)
    unk = sum(1 for a in pred if a == "unknown")
    bang = sum(1 for r in rows if (r.get("response_text") or "").strip() == "!!!!")
    e2e = [r["e2e_total_ms"] for r in rows if r.get("error") is None]

    return {
        "n": n,
        "acc_all": corr / n if n else 0.0,
        "unknown_rate": unk / n if n else 0.0,
        "all_bang_rate": bang / n if n else 0.0,
        "e2e_mean_ms": (sum(e2e) / len(e2e)) if e2e else 0.0,
        "e2e_p50_ms": statistics.median(e2e) if e2e else 0.0,
    }


if __name__ == "__main__":
    files = {
        "c9_t2": "reports/sweeps/int_1p5b_c9_t2_notts_28_phase2_fix2/report_20260331_125311.json",
        "c9_t3": "reports/sweeps/int_1p5b_c9_t3_notts_28_phase2_fix2/report_20260331_125431.json",
        "c9_t4": "reports/sweeps/int_1p5b_c9_t4_notts_28_phase2_fix2/report_20260331_124734.json",
        "c192_t2": "reports/sweeps/int_1p5b_c192_t2_notts_28_phase2_fix2/report_20260331_125628.json",
        "c192_t3": "reports/sweeps/int_1p5b_c192_t3_notts_28_phase2_fix2/report_20260331_125916.json",
        "c192_t4": "reports/sweeps/int_1p5b_c192_t4_notts_28_phase2_fix2/report_20260331_124936.json",
    }

    m = gt_map()
    for k, p in files.items():
        s = summarize(p, m)
        print(
            k,
            f"acc={s['acc_all']:.4f}",
            f"unk={s['unknown_rate']:.4f}",
            f"bang={s['all_bang_rate']:.4f}",
            f"e2e_mean_ms={s['e2e_mean_ms']:.2f}",
            f"e2e_p50_ms={s['e2e_p50_ms']:.2f}",
        )
