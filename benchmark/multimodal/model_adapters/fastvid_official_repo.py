from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FASTVID_ROOT = PROJECT_ROOT / "baselines/FastVID/fastvid_llavaonevision"
DEFAULT_FASTVID_COMMIT = "a40a109"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FastVIDOfficialRepoConfig:
    repo_root: Path
    pretrained: str
    conv_template: str = "qwen_1_5"
    model_name: str = "llava_qwen"
    max_frames_num: int = 32
    attn_implementation: str = "flash_attention_2"
    retention_ratio: float = 0.10
    dyseg_c: int = 8
    dyseg_tau: float = 0.9
    stprune_d: float = 0.4
    dtm_p: int = 4
    dtm_beta: float = 0.6


def _prepend_fastvid_paths(repo_root: Path) -> None:
    llava_next = repo_root / "LLaVA-NeXT"
    if not llava_next.exists():
        raise FileNotFoundError(f"FastVID LLaVA-OneVision source tree not found: {llava_next}")
    for path in (llava_next,):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


class FastVIDOfficialRepoRuntime:
    supports_batch_generation = False

    def __init__(self, cfg: FastVIDOfficialRepoConfig, device: torch.device):
        _prepend_fastvid_paths(cfg.repo_root)
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        self.cfg = cfg
        self.device = device
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token

        overwrite_config = {
            "mm_spatial_pool_stride": 2,
            "mm_spatial_pool_mode": "bilinear",
            "fastvid_retention_ratio": cfg.retention_ratio,
            "fastvid_DySeg_c": cfg.dyseg_c,
            "fastvid_DySeg_tau": cfg.dyseg_tau,
            "fastvid_STPrune_d": cfg.stprune_d,
            "fastvid_DTM_p": cfg.dtm_p,
            "fastvid_DTM_beta": cfg.dtm_beta,
        }
        AutoConfig.from_pretrained(cfg.pretrained)
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            cfg.pretrained,
            None,
            cfg.model_name,
            device_map=str(device),
            multimodal=True,
            attn_implementation=cfg.attn_implementation,
            overwrite_config=overwrite_config,
        )
        self.config = self.model.config
        self.model.eval()
        self.model.to(device)

    def _probe_video(self, path: Path) -> tuple[int, int, float]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError(f"ffprobe found no video stream for {path}.")
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
        duration = float((payload.get("format") or {}).get("duration") or 0.0)
        if width <= 0 or height <= 0 or duration <= 0:
            raise RuntimeError(f"Invalid ffprobe metadata for {path}: width={width} height={height} duration={duration}")
        return width, height, duration

    def _load_video_ffmpeg(self, path: Path) -> np.ndarray:
        width, height, duration = self._probe_video(path)
        frames = int(self.cfg.max_frames_num)
        fps = max(frames / duration, 1e-6)
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps:.12f}",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg frame extraction failed for {path}: {completed.stderr.decode(errors='replace').strip()}")
        frame_size = height * width * 3
        if frame_size <= 0 or len(completed.stdout) < frame_size:
            raise RuntimeError(f"ffmpeg decoded no complete RGB frames from {path}.")
        frame_count = len(completed.stdout) // frame_size
        usable = completed.stdout[: frame_count * frame_size]
        array = np.frombuffer(usable, dtype=np.uint8).reshape(frame_count, height, width, 3)
        if frame_count < frames:
            pad = np.repeat(array[-1:, ...], frames - frame_count, axis=0)
            array = np.concatenate([array, pad], axis=0)
        return array[:frames].copy()

    def _sample_frame_directory(self, path: Path) -> list[Path]:
        frame_paths = sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS and not child.name.startswith("._")
        )
        if not frame_paths:
            raise FileNotFoundError(f"FastVID official repo frame directory contains no image frames: {path}")
        target = max(1, int(self.cfg.max_frames_num))
        if target == 1:
            return [frame_paths[len(frame_paths) // 2]]
        if len(frame_paths) == 1:
            return [frame_paths[0] for _ in range(target)]
        return [frame_paths[round(idx * (len(frame_paths) - 1) / (target - 1))] for idx in range(target)]

    def _load_frame_directory(self, path: Path) -> np.ndarray:
        arrays: list[np.ndarray] = []
        for frame_path in self._sample_frame_directory(path):
            with Image.open(frame_path) as image:
                arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        first_shape = arrays[0].shape
        mismatched = [array.shape for array in arrays if array.shape != first_shape]
        if mismatched:
            raise RuntimeError(
                f"FastVID official repo frame directory has inconsistent frame shapes: "
                f"{path}; first={first_shape} mismatched_example={mismatched[0]}"
            )
        return np.stack(arrays, axis=0)

    def load_video(self, video_path: str) -> tuple[np.ndarray, str]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"FastVID official repo video file not found: {path}")
        if path.is_dir():
            return self._load_frame_directory(path), "frame_directory"
        try:
            vr = VideoReader(str(path), ctx=cpu(0))
            total_frame_num = len(vr)
            if total_frame_num <= 0:
                raise RuntimeError(f"FastVID official repo decoded zero frames from video: {path}")
            frame_idx = np.linspace(0, total_frame_num - 1, int(self.cfg.max_frames_num), dtype=int).tolist()
            return vr.get_batch(frame_idx).asnumpy(), "decord"
        except Exception as exc:
            if path.suffix.lower() != ".webm":
                raise
            try:
                return self._load_video_ffmpeg(path), "ffmpeg_fallback"
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"decord failed for {path} with {type(exc).__name__}: {exc}; "
                    f"ffmpeg fallback also failed with {type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc

    @torch.inference_mode()
    def generate_video_qa(self, *, text: str, video_path: str, max_new_tokens: int) -> dict[str, Any]:
        from llava.conversation import SeparatorStyle

        frame_start = time.perf_counter()
        frames, frame_decoder = self.load_video(video_path)
        frame_seconds = time.perf_counter() - frame_start
        preprocess_start = time.perf_counter()
        pixel_values = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        image_tensor = [pixel_values.to(device=self.device, dtype=torch.float16)]
        preprocess_seconds = time.perf_counter() - preprocess_start

        question = f"{self.DEFAULT_IMAGE_TOKEN}\n{text}"
        conv = copy.deepcopy(self.conv_templates[self.cfg.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_id).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = [self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]
        self.config.mm_spatial_pool_stride = 2
        self.config.mm_spatial_pool_mode = "bilinear"

        generation_start = time.perf_counter()
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            images=image_tensor,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            temperature=0,
            do_sample=False,
            top_p=None,
            num_beams=1,
            modalities=["video"],
            stopping_criteria=stopping_criteria,
        )
        generation_seconds = time.perf_counter() - generation_start
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_len = int(output_ids.shape[-1])
        input_len = int(input_ids.shape[-1])
        return {
            "text": decoded,
            "input_tokens": input_len,
            "video_tokens": None,
            "new_tokens": max(0, output_len - input_len) if output_len >= input_len else output_len,
            "frames": int(frames.shape[0]),
            "frame_decoder": frame_decoder,
            "frame_load_seconds": frame_seconds,
            "processor_seconds": preprocess_seconds,
            "generation_seconds": generation_seconds,
        }


def load_fastvid_official_repo_model(args: Any, device: torch.device) -> tuple[FastVIDOfficialRepoRuntime, dict[str, Any]]:
    cfg = FastVIDOfficialRepoConfig(
        repo_root=Path(args.fastvid_official_repo_dir),
        pretrained=args.fastvid_official_pretrained,
        conv_template=args.fastvid_official_conv_template,
        model_name=args.fastvid_official_model_name,
        max_frames_num=int(args.num_video_frames),
        attn_implementation=args.attn_implementation,
        retention_ratio=float(args.visual_keep_ratio),
        dyseg_c=int(args.fastvid_official_dyseg_c),
        dyseg_tau=float(args.fastvid_official_dyseg_tau),
        stprune_d=float(args.fastvid_official_stprune_d),
        dtm_p=int(args.fastvid_official_dtm_p),
        dtm_beta=float(args.fastvid_official_dtm_beta),
    )
    runtime = FastVIDOfficialRepoRuntime(cfg, device)
    policy = {
        "method": "fastvid_official_repo",
        "source_repo": f"LunarShen/FastVID@{DEFAULT_FASTVID_COMMIT}",
        "source_tree": str(cfg.repo_root),
        "model_loader": "FastVID fastvid_llavaonevision/LLaVA-NeXT llava.model.builder.load_pretrained_model",
        "pretrained": cfg.pretrained,
        "conv_template": cfg.conv_template,
        "model_name": cfg.model_name,
        "retention_ratio": cfg.retention_ratio,
        "dyseg_c": cfg.dyseg_c,
        "dyseg_tau": cfg.dyseg_tau,
        "stprune_d": cfg.stprune_d,
        "dtm_p": cfg.dtm_p,
        "dtm_beta": cfg.dtm_beta,
        "supports_batch_generation": False,
        "frame_sampling": "FastVID official repo uniform decord sampling from video path",
    }
    return runtime, policy
