#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter


def majority_label(example):
    votes = [ann.get("label") for ann in example.get("annotators", []) if ann.get("label")]
    if not votes:
        return "unknown"
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "undecided"
    return counts[0][0]


def sample_ids(ids, ratio, min_count, rng):
    if not ids:
        return []
    sample_size = int(round(len(ids) * ratio))
    sample_size = max(min_count, sample_size)
    sample_size = min(len(ids), sample_size)
    return rng.sample(ids, sample_size)


def summarize_split(dataset, split_dict):
    summary = {}
    for split_name in ["train", "val", "test"]:
        labels = Counter()
        for post_id in split_dict.get(split_name, []):
            if post_id in dataset:
                labels[majority_label(dataset[post_id])] += 1
        summary[split_name] = {
            "count": len(split_dict.get(split_name, [])),
            "labels": dict(labels),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create a sampled HateXplain subset with matching split ids")
    parser.add_argument("--input-data", default="Data/dataset.json")
    parser.add_argument("--input-split", default="Data/post_id_divisions.json")
    parser.add_argument("--output-data", default="Data/dataset_3pct.json")
    parser.add_argument("--output-split", default="Data/post_id_divisions_3pct.json")
    parser.add_argument("--ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-split", type=int, default=10)
    args = parser.parse_args()

    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError("--ratio must be in (0, 1].")

    with open(args.input_data, "r") as fp:
        dataset = json.load(fp)

    with open(args.input_split, "r") as fp:
        split_dict = json.load(fp)

    rng = random.Random(args.seed)
    sampled_split = {
        "train": sample_ids(list(split_dict.get("train", [])), args.ratio, args.min_per_split, rng),
        "val": sample_ids(list(split_dict.get("val", [])), args.ratio, args.min_per_split, rng),
        "test": sample_ids(list(split_dict.get("test", [])), args.ratio, args.min_per_split, rng),
    }

    sampled_ids = set(sampled_split["train"] + sampled_split["val"] + sampled_split["test"])
    sampled_dataset = {post_id: dataset[post_id] for post_id in sampled_ids if post_id in dataset}

    with open(args.output_split, "w") as fp:
        json.dump(sampled_split, fp, indent=2)

    with open(args.output_data, "w") as fp:
        json.dump(sampled_dataset, fp)

    original_summary = summarize_split(dataset, split_dict)
    sampled_summary = summarize_split(sampled_dataset, sampled_split)

    print("Original split summary:")
    print(json.dumps(original_summary, indent=2))
    print("\nSampled split summary:")
    print(json.dumps(sampled_summary, indent=2))
    print("\nWrote:")
    print(args.output_data)
    print(args.output_split)


if __name__ == "__main__":
    main()
