#!/usr/bin/env python3
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm import tqdm
from transformers import BertTokenizer

from Preprocess.dataCollect import get_annotated_data
from Preprocess.spanMatcher import returnMask


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError("Boolean flag expected.")


def contiguous_ranges(indices):
    if not indices:
        return []
    ranges = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev + 1))
        start = idx
        prev = idx
    ranges.append((start, prev + 1))
    return ranges


def get_evidence(post_id, anno_text, explanations):
    output = []
    active_indices = [i for i, each in enumerate(explanations) if each == 1]
    for start, end in contiguous_ranges(active_indices):
        output.append(
            {
                "docid": post_id,
                "end_sentence": -1,
                "end_token": end,
                "start_sentence": -1,
                "start_token": start,
                "text": " ".join([str(x) for x in anno_text[start:end]]),
            }
        )
    return output


def convert_to_eraser_format(dataframe, method, save_path, id_division, params_data, tokenizer):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "docs"), exist_ok=True)

    train_fp = open(os.path.join(save_path, "train.jsonl"), "w")
    val_fp = open(os.path.join(save_path, "val.jsonl"), "w")
    test_fp = open(os.path.join(save_path, "test.jsonl"), "w")

    try:
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            post_id = row["post_id"]
            majority_label = row["final_label"]

            if majority_label in ["undecided", "normal"]:
                continue

            anno_text, attention_masks = returnMask(row, params_data, tokenizer)
            if len(attention_masks) == 0:
                continue

            explanations = [list(each_explain) for each_explain in attention_masks]
            if method != "union":
                raise ValueError("Only union mode is currently supported.")
            final_explanation = [int(any(each)) for each in zip(*explanations)]

            temp = {
                "annotation_id": post_id,
                "classification": majority_label,
                "evidences": [get_evidence(post_id, list(anno_text), final_explanation)],
                "query": "What is the class?",
                "query_type": None,
            }

            with open(os.path.join(save_path, "docs", post_id), "w") as doc_fp:
                doc_fp.write(" ".join([str(x) for x in list(anno_text)]))

            if post_id in id_division["train"]:
                train_fp.write(json.dumps(temp) + "\n")
            elif post_id in id_division["val"]:
                val_fp.write(json.dumps(temp) + "\n")
            elif post_id in id_division["test"]:
                test_fp.write(json.dumps(temp) + "\n")
    finally:
        train_fp.close()
        val_fp.close()
        test_fp.close()


def main():
    parser = argparse.ArgumentParser(description="Build ERASER-format eval files from HateXplain data")
    parser.add_argument("--data-file", default="Data/dataset.json")
    parser.add_argument("--split-file", default="Data/post_id_divisions.json")
    parser.add_argument("--class-file", default="Data/classes.npy")
    parser.add_argument("--save-path", default="Data/Evaluation/Model_Eval")
    parser.add_argument("--bert-tokens", type=str2bool, default=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--method", default="union")
    args = parser.parse_args()

    params = {
        "num_classes": 3,
        "data_file": args.data_file,
        "class_names": args.class_file,
    }
    data_all_labeled = get_annotated_data(params)

    params_data = {
        "include_special": False,
        "bert_tokens": args.bert_tokens,
        "type_attention": "softmax",
        "set_decay": 0.1,
        "majority": 2,
        "max_length": args.max_length,
        "variance": 10,
        "window": 4,
        "alpha": 0.5,
        "p_value": 0.8,
        "method": "additive",
        "decay": False,
        "normalized": False,
        "not_recollect": True,
    }

    tokenizer = None
    if params_data["bert_tokens"]:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)

    with open(args.split_file, "r") as fp:
        id_division = json.load(fp)

    convert_to_eraser_format(
        dataframe=data_all_labeled,
        method=args.method,
        save_path=args.save_path,
        id_division=id_division,
        params_data=params_data,
        tokenizer=tokenizer,
    )

    print("Wrote ERASER files to:", args.save_path)


if __name__ == "__main__":
    main()
