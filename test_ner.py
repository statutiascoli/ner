import argparse
import json


ENTITY_TYPES = ("persona", "luogo", "evento")

LABEL_MAP = {
    "PER": "persona",
    "LOC": "luogo",
    "FES": "evento",
}


def entity_key_exact(e):
    if e["text"].lower().startswith("asc"):
        return None
    return (e["type"], e["start"], e["end"])


def iou(e1, e2):
    start_i = max(e1["start"], e2["start"])
    end_i   = min(e1["end"],   e2["end"])
    inter   = max(0, end_i - start_i)
    if inter == 0:
        return 0.0
    union = (e1["end"] - e1["start"]) + (e2["end"] - e2["start"]) - inter
    return inter / union if union > 0 else 0.0


def match_overlap(pred_ents, gold_ents, iou_threshold = 0.5):
    matched_gold = set()
    matched_pred = set()
    tp, fp, fn = [], [], []

    for pi, pred in enumerate(pred_ents):
        best_iou  = 0.0
        best_gi   = None
        for gi, gold in enumerate(gold_ents):
            if gi in matched_gold:
                continue
            if pred["type"] != gold["type"]:
                continue
            score = iou(pred, gold)
            if score > best_iou:
                best_iou = score
                best_gi  = gi

        if best_gi is not None and best_iou >= iou_threshold:
            tp.append(pred)
            matched_gold.add(best_gi)
            matched_pred.add(pi)
        else:
            fp.append(pred)

    for gi, gold in enumerate(gold_ents):
        if gi not in matched_gold:
            fn.append(gold)

    return tp, fp, fn


def compute_prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def evaluate_exact(paired_data):
    global_tp = {t: 0 for t in ENTITY_TYPES}
    global_fp = {t: 0 for t in ENTITY_TYPES}
    global_fn = {t: 0 for t in ENTITY_TYPES}
    fp_examples, fn_examples = [], []

    for doc in paired_data:
        doc_id   = doc["doc_id"]
        gold_map = {entity_key_exact(e): e for e in doc["gold"] if entity_key_exact(e)}
        pred_map = {entity_key_exact(e): e for e in doc["pred"] if entity_key_exact(e)}

        for key, pred_e in pred_map.items():
            t = pred_e["type"]
            if t not in ENTITY_TYPES:
                continue
            if key in gold_map and gold_map[key]["type"] == t:
                global_tp[t] += 1
            else:
                global_fp[t] += 1
                fp_examples.append({"doc_id": doc_id, "entity": {"type": t, "text": pred_e["text"]}})

        for key, gold_e in gold_map.items():
            t = gold_e["type"]
            if t not in ENTITY_TYPES:
                continue
            if key not in pred_map or pred_map[key]["type"] != t:
                global_fn[t] += 1
                fn_examples.append({"doc_id": doc_id, "entity": {"type": t, "text": gold_e["text"]}})

    return _build_scores(global_tp, global_fp, global_fn, fp_examples, fn_examples)


def evaluate_overlap(paired_data, iou_threshold = 0.5):
    global_tp = {t: 0 for t in ENTITY_TYPES}
    global_fp = {t: 0 for t in ENTITY_TYPES}
    global_fn = {t: 0 for t in ENTITY_TYPES}
    fp_examples, fn_examples = [], []

    for doc in paired_data:
        doc_id = doc["doc_id"]

        # filter out Ascoli self-references
        gold = [e for e in doc["gold"] if not e["text"].lower().startswith("asc")]
        pred = [e for e in doc["pred"] if not e["text"].lower().startswith("asc")]

        for t in ENTITY_TYPES:
            gold_t = [e for e in gold if e["type"] == t]
            pred_t = [e for e in pred if e["type"] == t]

            tp_list, fp_list, fn_list = match_overlap(pred_t, gold_t, iou_threshold)

            global_tp[t] += len(tp_list)
            global_fp[t] += len(fp_list)
            global_fn[t] += len(fn_list)

            for e in fp_list:
                fp_examples.append({"doc_id": doc_id, "entity": {"type": t, "text": e["text"]}})
            for e in fn_list:
                fn_examples.append({"doc_id": doc_id, "entity": {"type": t, "text": e["text"]}})

    return _build_scores(global_tp, global_fp, global_fn, fp_examples, fn_examples)


def _build_scores(global_tp, global_fp, global_fn, fp_examples, fn_examples):
    micro = {}
    for t in ENTITY_TYPES:
        p, r, f = compute_prf(global_tp[t], global_fp[t], global_fn[t])
        micro[t] = {"precision": p, "recall": r, "f1": f,
                    "tp": global_tp[t], "fp": global_fp[t], "fn": global_fn[t]}

    total_tp = sum(global_tp.values())
    total_fp = sum(global_fp.values())
    total_fn = sum(global_fn.values())
    p, r, f  = compute_prf(total_tp, total_fp, total_fn)
    micro["overall"] = {"precision": p, "recall": r, "f1": f,
                        "tp": total_tp, "fp": total_fp, "fn": total_fn}

    return {
        "micro": micro,
        "errors": {"false_positives": fp_examples, "false_negatives": fn_examples}
    }


def print_scores(scores, label):
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    for entity_type, vals in scores["micro"].items():
        tp_info = (f"  tp={vals['tp']} fp={vals['fp']} fn={vals['fn']}"
                   if "tp" in vals else "")
        print(f"  {entity_type:10s}  "
              f"P={vals['precision']:.4f}  "
              f"R={vals['recall']:.4f}  "
              f"F1={vals['f1']:.4f}{tp_info}")
    print("=" * 62)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("records", "data", "items", "statutes"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError(f"Cannot locate a list of records in '{path}'.")


def main(gold_path, pred_path, output_path, iou_threshold):
    gold_records = load_json(gold_path)
    pred_records = load_json(pred_path)

    if len(gold_records) != len(pred_records):
        raise ValueError("Gold/pred length mismatch")

    paired_data = []
    skipped = 0

    for gold_rec, pred_rec in zip(gold_records, pred_records):
        gold_ents = [e for e in gold_rec.get("entities", []) if e.get("type") in ENTITY_TYPES]
        pred_ents = [
            {**e, "type": LABEL_MAP.get(e.get("type"), e.get("type"))}
            for e in pred_rec.get("entities", [])
            if e.get("type") in LABEL_MAP
        ]
        if not gold_ents:
            skipped += 1
            continue
        paired_data.append({"doc_id": gold_rec["id"], "gold": gold_ents, "pred": pred_ents})

    print(f"Evaluating {len(paired_data)} docs ({skipped} skipped)")

    scores_exact   = evaluate_exact(paired_data)
    scores_overlap = evaluate_overlap(paired_data, iou_threshold)

    print_scores(scores_exact, f"EXACT MATCH")
    print_scores(scores_overlap, f"OVERLAP MATCH  (IoU ≥ {iou_threshold})")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"exact": scores_exact, "overlap": scores_overlap},
                      f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gold",      required=True)
    parser.add_argument("-p", "--pred",      required=True)
    parser.add_argument("-o", "--output",    default=None)
    parser.add_argument("--iou",             type=float, default=0.5)
    args = parser.parse_args()
    main(args.gold, args.pred, args.output, args.iou)