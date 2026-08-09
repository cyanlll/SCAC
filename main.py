#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging

from scac.config import load_config


def _parser():
    p = argparse.ArgumentParser(description="SCAC reproducible pipeline")
    p.add_argument("--config", required=True, help="YAML config file")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("query-chain", help="Generate query event chains with Ollama")
    sub.add_parser("format-query-chain", help="Convert query-chain JSON to JSONL")
    sub.add_parser("caption", help="Generate frame captions")
    sub.add_parser("extract-text", help="Extract query/query-chain text features")
    sub.add_parser("extract-video", help="Extract original video features")
    sub.add_parser("extract-video-chain", help="Extract video event-chain features")
    sub.add_parser("retrieve", help="Run event-chain cross-modal retrieval")
    sub.add_parser("evaluate", help="Run localization and final evaluation")
    sub.add_parser("pipeline", help="Run configured preprocessing stages then evaluate")
    return p


def run(command, cfg):
    if command == "query-chain":
        from scac.preprocess.query_chain import generate_query_chains
        q = cfg["query_chain"]
        return generate_query_chains(
            q["input_json"], q["raw_output_json"], q["prompt_file"],
            model=q.get("model", "llama3.1:8b-instruct-fp16"),
            temperature=q.get("temperature", 0.7), max_tokens=q.get("max_tokens", 256),
            retries=q.get("retries", 50),
        )
    if command == "format-query-chain":
        from scac.preprocess.query_chain import convert_json_to_jsonl
        q = cfg["query_chain"]
        return convert_json_to_jsonl(q["raw_output_json"], cfg["paths"]["query_chain_jsonl"])
    if command == "caption":
        from scac.preprocess.caption import generate_video_captions
        c = cfg["caption"]
        return generate_video_captions(c["image_root"], c["output_jsonl"], c["model_path"], c.get("device", "cpu"))
    if command == "extract-text":
        from scac.preprocess.text_features import extract_text_features
        f = cfg["features"]["text"]
        return extract_text_features(
            cfg["paths"]["query_chain_jsonl"], f["save_root"],
            device=cfg["runtime"].get("device", "cuda:0"),
            encoder_mode=f.get("encoder_mode", "legacy_qformer"),
            model_name=f.get("model_name", "blip_image_text_matching"), model_type=f.get("model_type", "large"),
        )
    if command == "extract-video":
        from scac.preprocess.video_features import extract_video_features
        f = cfg["features"]["video"]
        return extract_video_features(
            f["input_root"], f["save_root"], fps=f.get("fps", 3), batch_size=f.get("batch_size", 128),
            device=cfg["runtime"].get("device", "cuda:0"), model_name=f.get("model_name", "blip_image_text_matching"),
            model_type=f.get("model_type", "large"),
        )
    if command == "extract-video-chain":
        from scac.preprocess.video_event_features import extract_video_event_features
        f = cfg["features"]["video_chain"]
        return extract_video_event_features(
            f["input_root"], f["save_root"], fps=f.get("fps", 3), batch_size=f.get("batch_size", 256),
            device=cfg["runtime"].get("device", "cuda:0"), model_name=f.get("model_name", "blip_image_text_matching"),
            model_type=f.get("model_type", "large"),
        )
    if command == "retrieve":
        from scac.core.retrieval import run_retrieval
        return run_retrieval(cfg)
    if command == "evaluate":
        from scac.evaluation import run_evaluation
        return run_evaluation(cfg)
    if command == "pipeline":
        stages = cfg.get("pipeline", {}).get("stages", ["query-chain", "format-query-chain", "caption", "extract-text", "extract-video", "extract-video-chain", "retrieve", "evaluate"])
        for stage in stages:
            logging.info("=== %s ===", stage)
            run(stage, cfg)
        return None
    raise ValueError(command)


def main():
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config(args.config)
    run(args.command, cfg)


if __name__ == "__main__":
    main()
