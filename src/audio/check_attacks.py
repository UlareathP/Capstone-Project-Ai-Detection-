"""Per-attack breakdown on the test split: which kinds of fake voice does the model miss?

For each attack type (A07, A08, ...) it prints how many fake clips of that type the model
flagged as fake (the "caught" rate), the average fake-probability it assigned, and the ROC AUC
of that attack against all real clips (threshold-free). Real clips are shown as one group.

Run from the repo root, after train.py:  python src/audio/check_attacks.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from common import (MODEL_PATH, SegmentDataset, build_model, device,
                    eval_transform, predict, read_splits)

for path in (MODEL_PATH, "data/audio_segments/splits.csv"):
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Run from the repo root, after training.")

test = [c for c in read_splits() if c["split"] == "test"]
model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
loader = DataLoader(SegmentDataset(test, eval_transform), batch_size=32, shuffle=False)
_, (labels, probs) = predict(model, loader, len(test))   # probs = probability the clip is FAKE

groups = defaultdict(list)
for c, p in zip(test, probs):
    groups[c["attack_id"]].append(p)

real_probs = groups.get("bonafide", [])
print(f"{'group':10} {'clips':>6} {'flagged fake':>13} {'avg fake-prob':>14} {'AUC vs real':>12}")
print("-" * 60)
for attack in sorted(a for a in groups if a != "bonafide"):
    ps = groups[attack]
    caught = sum(p >= 0.5 for p in ps) / len(ps)
    auc = ""
    if real_probs:
        auc = f"{roc_auc_score([1] * len(ps) + [0] * len(real_probs), ps + real_probs):.3f}"
    print(f"{attack:10} {len(ps):6d} {caught:12.1%} {sum(ps) / len(ps):14.3f} {auc:>12}")
if real_probs:
    kept = sum(p < 0.5 for p in real_probs) / len(real_probs)
    print("-" * 60)
    print(f"{'real':10} {len(real_probs):6d} {kept:12.1%}  (share correctly kept as real)")
