"""Train NonAutoregressiveDecisionHead on frozen embeddings (train split only).

Objective mirrors von's published recipe: listwise softmax cross-entropy +
Brier, AdamW, early stop on DEV accuracy. Test embeddings are never touched.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anydecision.models.decision_head import NonAutoregressiveDecisionHead


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(mask < 0.5, float("-inf"))
    return torch.softmax(logits, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--dev-npz", required=True)
    ap.add_argument("--out-head", required=True)
    ap.add_argument("--out-log", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--brier-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center", action="store_true", default=True)
    ap.add_argument("--no-center", dest="center", action="store_false")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = np.load(args.train_npz)
    dv = np.load(args.dev_npz)
    hidden = int(tr["context"].shape[1])
    head = NonAutoregressiveDecisionHead(hidden_dim=hidden, projection_dim=256)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    # Anisotropy control: center by TRAIN means only (no dev/test leakage).
    ctx_mean = np.zeros(tr["context"].shape[1], dtype=np.float32)
    opt_mean = np.zeros(tr["options"].shape[2], dtype=np.float32)
    if args.center:
        ctx_mean = tr["context"].mean(axis=0).astype(np.float32)
        opt_mean = tr["options"][tr["mask"] > 0.5].mean(axis=0).astype(np.float32)

    def _prep(npz: dict) -> tuple:
        ctx = torch.from_numpy(npz["context"].astype(np.float32) - ctx_mean)
        opts = torch.from_numpy(npz["options"].astype(np.float32) - opt_mean)
        return (ctx, opts, torch.from_numpy(npz["mask"]),
                torch.from_numpy(npz["label_index"]))

    ctx_tr, opt_tr, mask_tr, y_tr = _prep(tr)
    ctx_dv, opt_dv, mask_dv, y_dv = _prep(dv)

    def split_metrics(ctx, opts, mask, y):
        head.eval()
        with torch.no_grad():
            n = ctx.shape[0]
            correct, brier, nll, confs, hits = 0, 0.0, 0.0, [], []
            for i in range(n):
                logits = head(ctx[i:i + 1], opts[i])  # [1, C]
                valid = mask[i].bool()
                probs = masked_softmax(logits, mask[i:i + 1])[0, valid]
                pv = probs.cpu().numpy()
                pred = int(np.argmax(pv))
                true = int(y[i])
                correct += int(pred == true)
                oh = np.zeros_like(pv)
                oh[true] = 1.0
                brier += float(np.sum((pv - oh) ** 2))
                nll += float(-np.log(max(1e-12, pv[true])))
                confs.append(float(np.max(pv)))
                hits.append(int(pred == true))
        return {
            "accuracy": correct / n,
            "brier": brier / n,
            "nll": nll / n,
            "mean_confidence": float(np.mean(confs)),
            "n": n,
        }

    best_dev, best_state, bad, history = -1.0, None, 0, []
    head.train()
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        n = ctx_tr.shape[0]
        ce_sum, br_sum = 0.0, 0.0
        for i in range(n):
            logits = head(ctx_tr[i:i + 1], opt_tr[i])
            probs = masked_softmax(logits, mask_tr[i:i + 1])
            valid = mask_tr[i].bool()
            pv = probs[0, valid]
            true = int(y_tr[i])
            ce_sum = ce_sum + F.nll_loss(torch.log(pv + 1e-12).unsqueeze(0),
                                         torch.tensor([true]))
            oh = torch.zeros_like(pv)
            oh[true] = 1.0
            br_sum = br_sum + torch.sum((pv - oh) ** 2)
        loss = ce_sum / n + args.brier_weight * br_sum / n
        loss.backward()
        opt.step()

        tr_m = split_metrics(ctx_tr, opt_tr, mask_tr, y_tr)
        dv_m = split_metrics(ctx_dv, opt_dv, mask_dv, y_dv)
        history.append({"epoch": epoch, "loss": float(loss.detach()),
                        "train_acc": tr_m["accuracy"], "dev_acc": dv_m["accuracy"]})
        if dv_m["accuracy"] > best_dev + 1e-12:
            best_dev = dv_m["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 25 == 0 or bad == 0:
            print(f"ep {epoch:3d} loss={loss.item():.4f} "
                  f"train_acc={tr_m['accuracy']:.3f} dev_acc={dv_m['accuracy']:.3f}")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (best dev_acc={best_dev:.3f})")
            break

    assert best_state is not None
    head.load_state_dict(best_state)
    out_h = Path(args.out_head)
    out_h.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "hidden_dim": hidden,
                "projection_dim": 256, "backbone": str(tr["backbone"][0]),
                "center": bool(args.center),
                "ctx_mean": ctx_mean, "opt_mean": opt_mean},
               out_h)
    final_dev = split_metrics(ctx_dv, opt_dv, mask_dv, y_dv)
    out_l = Path(args.out_log)
    out_l.write_text(json.dumps({
        "seed": args.seed, "epochs_run": len(history), "best_dev_acc": best_dev,
        "final_dev": final_dev, "lr": args.lr, "brier_weight": args.brier_weight,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"saved {out_h}; best dev_acc={best_dev:.3f}")


if __name__ == "__main__":
    main()
