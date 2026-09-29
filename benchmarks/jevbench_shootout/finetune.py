"""Full fine-tune: ModernBERT encoder (unfrozen) + bilinear decision head.

Per item: premise + joint premise+option pairs in ONE batched forward;
bilinear head scores context vs pair vectors; loss = CE + 0.5*Brier.
Discriminative LRs (encoder 2e-5, head 2e-4), linear schedule, grad clip.
Early stop on AUX-dev accuracy; JB-dev reported as gate (never selected on).
Checkpoints per epoch; --resume continues.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anydecision.models.decision_head import NonAutoregressiveDecisionHead
from adapters import describe_item, noul_option_texts, score_levels


def load_arc() -> dict[str, dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows: dict[str, dict] = {}
    for subset, path in (("easy", "ARC-Easy/train-00000-of-00001.parquet"),
                         ("challenge", "ARC-Challenge/train-00000-of-00001.parquet")):
        local = hf_hub_download("allenai/ai2_arc", path, repo_type="dataset")
        for r in pq.read_table(local).to_pylist():
            rows[f"arc-{subset}-{r['id']}"] = r
    return rows


def load_jb(data_dir: Path) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for f in ("easy", "original", "hard"):
        for line in open(data_dir / f"{f}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            items[d["id"]] = d
    return items


def build_texts(_id: str, arc: dict, jb: dict) -> tuple[str, list[str], int] | None:
    """Return (premise, option_texts, label_index) or None if unusable."""
    if _id.startswith("arc-"):
        r = arc.get(_id)
        if r is None:
            return None
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        if r["answerKey"] not in labels:
            return None
        return r["question"], [f"{l}: {t}" for l, t in zip(labels, texts)], labels.index(r["answerKey"])
    item = jb.get(_id)
    if item is None:
        return None
    prompt, labels, qtype, exp = describe_item(item)
    q = item["question"]
    premise = f"{q.get('instructions', '')}\n{item.get('state', '')}"
    if qtype == "choice":
        crit = q.get("criteria", {}) or {}
        otexts = [f"{lab}: {crit.get(lab, lab)}" for lab in labels]
    elif qtype == "noul":
        labels, otexts = noul_option_texts(item)
    else:
        levels = score_levels(item)
        otexts = [f"{lab}: {desc}" for lab, desc in zip(labels, levels)]
    if exp not in labels:
        return None
    return premise, otexts, labels.index(exp)


def encode_batch(texts: list[str], tok, model) -> torch.Tensor:
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    out = model(**enc)
    h = out.last_hidden_state.float()
    mask = enc["attention_mask"].unsqueeze(-1).float()
    v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)


def evaluate(ids: list[str], arc, jb, tok, model, head) -> dict:
    model.eval()
    head.eval()
    correct, brier, n = 0, 0.0, 0
    with torch.no_grad():
        for _id in ids:
            built = build_texts(_id, arc, jb)
            if built is None:
                continue
            premise, otexts, true = built
            vecs = encode_batch([premise] + [f"{premise}\nCandidate: {o}" for o in otexts], tok, model)
            logits = head(vecs[0:1].detach(), vecs[1:].detach())[0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            pred = int(np.argmax(probs))
            correct += int(pred == true)
            oh = np.zeros_like(probs)
            oh[true] = 1.0
            brier += float(np.sum((probs - oh) ** 2))
            n += 1
    model.train()
    head.train()
    return {"accuracy": correct / max(1, n), "brier": brier / max(1, n), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--train-ids", required=True, help="json list of training ids (ARC + JB-train)")
    ap.add_argument("--aux-dev-ids", required=True, help="json list of aux-dev ids")
    ap.add_argument("--jb-dev-npz", default=None, help="frozen JB-dev npz for gate eval (unused: rebuilt from text)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.backbone)
    enc_model = AutoModel.from_pretrained(args.backbone, dtype="float32")
    head = NonAutoregressiveDecisionHead(hidden_dim=enc_model.config.hidden_size
                                        if hasattr(enc_model.config, "hidden_size")
                                        else 768,
                                        projection_dim=256)
    hidden_dim = head.hidden_dim
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        enc_model.load_state_dict(ckpt["encoder"])
        head.load_state_dict(ckpt["head"])
        print(f"resumed from {args.resume}")

    arc = load_arc()
    jb = load_jb(Path(args.data_dir))
    train_ids = json.load(open(args.train_ids))
    aux_dev_ids = json.load(open(args.aux_dev_ids))
    splits = json.load(open(args.splits))["splits"]
    jb_dev_ids = splits["dev"]
    print(f"train={len(train_ids)} aux-dev={len(aux_dev_ids)} jb-dev={len(jb_dev_ids)} hidden={hidden_dim}")

    opt = torch.optim.AdamW([{"params": enc_model.parameters(), "lr": args.lr_enc},
                             {"params": head.parameters(), "lr": args.lr_head}],
                            weight_decay=0.01)
    total_steps = args.epochs * len(train_ids)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=min(200, total_steps // 10))
    # Linear decay after warmup via second scheduler chained manually below.
    sched2 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0,
                                               total_iters=max(1, total_steps))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    best_aux, bad = -1.0, 0
    step = 0
    enc_model.train()
    head.train()
    for epoch in range(1, args.epochs + 1):
        order = np.random.RandomState(args.seed + epoch).permutation(len(train_ids))
        ce_sum, br_sum, n_seen = 0.0, 0.0, 0
        for k, oi in enumerate(order):
            built = build_texts(train_ids[int(oi)], arc, jb)
            if built is None:
                continue
            premise, otexts, true = built
            vecs = encode_batch([premise] + [f"{premise}\nCandidate: {o}" for o in otexts], tok, enc_model)
            logits = head(vecs[0:1], vecs[1:])[0]
            probs = F.softmax(logits, dim=-1)
            ce = F.nll_loss(torch.log(probs + 1e-12).unsqueeze(0), torch.tensor([true]))
            oh = torch.zeros_like(probs)
            oh[true] = 1.0
            br = torch.sum((probs - oh) ** 2)
            (ce + 0.5 * br).backward()
            torch.nn.utils.clip_grad_norm_(list(enc_model.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            if step < 200:
                sched.step()
            else:
                sched2.step()
            opt.zero_grad()
            step += 1
            ce_sum += float(ce.detach())
            br_sum += float(br.detach())
            n_seen += 1
            if (k + 1) % 500 == 0:
                print(f"  ep{epoch} item {k + 1}/{len(order)} loss={ce_sum / n_seen:.3f}", flush=True)
        aux_m = evaluate(aux_dev_ids, arc, jb, tok, enc_model, head)
        jb_m = evaluate(jb_dev_ids, arc, jb, tok, enc_model, head)
        rec = {"epoch": epoch, "train_loss": ce_sum / max(1, n_seen),
               "aux_dev_acc": aux_m["accuracy"], "aux_dev_brier": aux_m["brier"],
               "jb_dev_acc": jb_m["accuracy"], "jb_dev_brier": jb_m["brier"]}
        log.append(rec)
        print(f"EPOCH {epoch}: train_loss={rec['train_loss']:.3f} "
              f"aux-dev={aux_m['accuracy']:.3f} jb-dev={jb_m['accuracy']:.3f}", flush=True)
        torch.save({"encoder": enc_model.state_dict(), "head": head.state_dict(),
                    "epoch": epoch, "hidden_dim": hidden_dim},
                   out_dir / f"ckpt_ep{epoch}.pt")
        if aux_m["accuracy"] > best_aux + 1e-12:
            best_aux = aux_m["accuracy"]
            bad = 0
            torch.save({"encoder": enc_model.state_dict(), "head": head.state_dict(),
                        "epoch": epoch, "hidden_dim": hidden_dim},
                       out_dir / "best.pt")
        else:
            bad += 1
        (out_dir / "finetune_log.json").write_text(json.dumps({
            "args": vars(args), "best_aux_dev_acc": best_aux, "log": log,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
    print(f"done. best aux-dev={best_aux:.3f}")


if __name__ == "__main__":
    main()
