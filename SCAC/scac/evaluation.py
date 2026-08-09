from __future__ import annotations

import logging
import pickle
from pathlib import Path

from .core.localization import calibrate_and_rerank, evaluate_oracle_localization, generate_intervals


def load_retrieval_result(path):
    with open(path, "rb") as f:
        result = pickle.load(f)
    if not isinstance(result, list) or len(result) < 7:
        raise ValueError("Retrieval pickle must follow the legacy 7-item list format.")
    return {
        "r1": result[0]["r1"],
        "r5": result[1]["r5"],
        "r10": result[2]["r10"],
        "r20": result[3]["r20"],
        "r50": result[4]["r50"],
        "r100": result[5]["r100"],
        "similarity_scores_all": result[6]["similarity_scores_all"],
    }


def _cross_video_metric(retrieval_dict, location_result, threshold):
    correct = 0
    total = len(retrieval_dict)
    for query_vid, top_videos in retrieval_dict.items():
        target_vid = query_vid.rsplit("_", 1)[0]
        localized = location_result.get(query_vid)
        if localized and target_vid in [vid for vid, _ in top_videos] and localized[0] >= threshold:
            correct += 1
    return correct / total if total else 0.0


def run_evaluation(cfg):
    dataset = cfg["dataset"]["name"]
    loc_cfg = cfg["localization"]
    paths = cfg["paths"]
    runtime = cfg["runtime"]
    device = runtime.get("device", "cuda:0")
    top_k = int(runtime.get("top_k", 200))

    query_chain = Path(paths["query_chain_jsonl"])
    retrieval_pkl = Path(cfg.get("retrieval", {}).get("output_pkl", paths.get("retrieval_pkl", "")))
    interval_file = Path(paths["interval_file"])
    video_feature_root = Path(paths["video_feature_root"])
    text_feature_root = Path(paths["text_feature_root"])
    interval_file.parent.mkdir(parents=True, exist_ok=True)

    retrieval = load_retrieval_result(retrieval_pkl)
    similarities = {k: v[:top_k] for k, v in retrieval["similarity_scores_all"].items()}

    stride = int(loc_cfg["stride"])
    max_stride = float(loc_cfg["max_stride_factor"])
    logging.info("dataset=%s stride=%s max_stride_factor=%s", dataset, stride, max_stride)

    if cfg.get("evaluation", {}).get("generate_intervals", True):
        generate_intervals(
            query_chain,
            dataset,
            similarities,
            stride,
            max_stride,
            logging,
            interval_file,
            video_feature_root,
            text_feature_root,
            device=device,
        )

    if cfg.get("evaluation", {}).get("oracle_localization", True):
        evaluate_oracle_localization(
            query_chain,
            dataset,
            stride,
            max_stride,
            logging,
            video_feature_root,
            text_feature_root,
            device=device,
        )

    location_result = calibrate_and_rerank(
        query_chain,
        dataset,
        similarities,
        stride,
        max_stride,
        logging,
        interval_file,
        video_feature_root,
        text_feature_root,
        step=int(loc_cfg.get("calibration_step", 1)),
        overlap_threshold=float(loc_cfg.get("overlap_threshold", 0.3)),
        cc=int(loc_cfg.get("cc", 1)),
        device=device,
    )

    for n, retrieval_dict in ((10, retrieval["r10"]), (100, retrieval["r100"])):
        for th in (0.3, 0.5, 0.7):
            value = _cross_video_metric(retrieval_dict, location_result, th)
            print(f"R@{n} IoU={th}: {value * 100:.6f}")
            logging.info("R@%s IoU=%s: %s", n, th, value * 100)
    return location_result
