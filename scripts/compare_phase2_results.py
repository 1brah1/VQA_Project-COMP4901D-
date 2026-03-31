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
    vlm = [r["vlm_total_ms"] for r in rows if r.get("error") is None]

    integ = j.get("pipeline_integrity", {})
    deg = j.get("degeneration", {})

    print("FILE", path)
    print("n", n)
    print("acc_all", round(corr / n, 4) if n else 0)
    print("unknown_rate", round(unk / n, 4) if n else 0)
    print("all_bang_rate", round(bang / n, 4) if n else 0)
    print("e2e_mean_ms", round(sum(e2e) / len(e2e), 2) if e2e else 0)
    print("e2e_p50_ms", round(statistics.median(e2e), 2) if e2e else 0)
    print("vlm_mean_ms", round(sum(vlm) / len(vlm), 2) if vlm else 0)
    print(
        "integrity",
        integ.get("model_hidden_size_valid"),
        integ.get("compression_selected_valid"),
    )
    if deg:
        print(
            "deg_fields",
            {
                k: deg[k]
                for k in [
                    "unknown_rate",
                    "all_bang_rate",
                    "prompt_echo_rate",
                    "stop_reason_max_new_tokens_rate",
                ]
                if k in deg
            },
        )

    for task in ["crosswalk_signal", "stairs", "obstacles"]:
        tr = [r for r in rows if r["task"] == task]
        nn = len(tr)
        if nn == 0:
            continue
        cc = sum(1 for r in tr if r["classification_label"] == m[r["sample_id"]])
        uu = sum(1 for r in tr if r["classification_label"] == "unknown")
        print(" task", task, "acc", round(cc / nn, 4), "unk", round(uu / nn, 4))

    print("---")


if __name__ == "__main__":
    m = gt_map()
    paths = [
        "reports/sweeps/int_1p5b_c9_t4_notts_28_phase1/report_20260331_123715.json",
        "reports/sweeps/int_1p5b_c9_t4_notts_28_phase2_fix2/report_20260331_124734.json",
        "reports/sweeps/int_1p5b_c192_t4_notts_28_phase1/report_20260331_123804.json",
        "reports/sweeps/int_1p5b_c192_t4_notts_28_phase2_fix2/report_20260331_124936.json",
    ]
    for p in paths:
        summarize(p, m)
