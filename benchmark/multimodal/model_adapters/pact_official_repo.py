from __future__ import annotations

import copy
import json
import os
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACT_ROOT = PROJECT_ROOT / "baselines/PACT"
DEFAULT_PACT_COMMIT = "18669a5"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PACTOfficialRepoConfig:
    repo_root: Path
    pretrained: str
    config_path: Path
    conv_template: str = "qwen_1_5"
    model_name: str = "llava_qwen"
    max_frames_num: int = 32
    attn_implementation: str = "flash_attention_2"
    cutoff: float = 0.21
    pruning_tokeep_percentage_value: float = 0.55
    mm_spatial_pool_stride: int = 2
    mm_spatial_pool_mode: str = "bilinear"


def _path_is_relative_to(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _prepend_pact_paths(repo_root: Path) -> None:
    llava_next = repo_root / "LLaVA-NeXT"
    transformers_root = repo_root / "transformers"
    if not llava_next.exists():
        raise FileNotFoundError(f"PACT LLaVA-NeXT source tree not found: {llava_next}")
    if not transformers_root.exists():
        raise FileNotFoundError(f"PACT vendored transformers source tree not found: {transformers_root}")

    loaded_transformers = sys.modules.get("transformers")
    if loaded_transformers is not None and not _path_is_relative_to(
        getattr(loaded_transformers, "__file__", None),
        transformers_root,
    ):
        raise RuntimeError(
            "PACT official repo requires its vendored transformers to be imported first. "
            f"Already loaded transformers from {getattr(loaded_transformers, '__file__', None)!r}. "
            "Run PACT in a standalone evaluator process and do not mix it with HF methods."
        )
    loaded_llava = sys.modules.get("llava")
    if loaded_llava is not None and not _path_is_relative_to(getattr(loaded_llava, "__file__", None), llava_next):
        raise RuntimeError(
            "PACT official repo requires its LLaVA-NeXT package to be imported first. "
            f"Already loaded llava from {getattr(loaded_llava, '__file__', None)!r}. "
            "Run PACT in a standalone evaluator process and do not mix it with other official repos."
        )

    for path in (repo_root, llava_next):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _configure_pact_env(cfg: PACTOfficialRepoConfig) -> None:
    if not cfg.config_path.exists():
        raise FileNotFoundError(f"PACT config file not found: {cfg.config_path}")
    os.environ["pact_config_path"] = str(cfg.config_path)
    os.environ["cutoff"] = str(cfg.cutoff)
    os.environ["pruning_tokeep_percentage_value"] = str(cfg.pruning_tokeep_percentage_value)


class PACTOfficialRepoRuntime:
    supports_batch_generation = False

    def __init__(self, cfg: PACTOfficialRepoConfig, device: torch.device):
        _configure_pact_env(cfg)
        _prepend_pact_paths(cfg.repo_root)

        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from transformers import AutoConfig
        import transformers
        from transformers.PACT.utils import load_config

        transformers_path = getattr(transformers, "__file__", "")
        expected_transformers_root = cfg.repo_root / "transformers"
        if not _path_is_relative_to(transformers_path, expected_transformers_root):
            raise RuntimeError(
                f"PACT vendored transformers was not loaded. transformers.__file__={transformers_path!r}; "
                f"expected under {expected_transformers_root}."
            )

        self.cfg = cfg
        self.device = device
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.conv_templates = conv_templates
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.pact_config = vars(load_config()).copy()

        overwrite_config = {
            "mm_spatial_pool_stride": cfg.mm_spatial_pool_stride,
            "mm_spatial_pool_mode": cfg.mm_spatial_pool_mode,
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

    def _build_prompt(self, text: str, placeholder_count: int = 1) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_tokens = " ".join([self.DEFAULT_IMAGE_TOKEN] * max(1, int(placeholder_count)))
        question = f"{image_tokens}\n{text}"
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
        return input_ids, attention_mask, conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2

    @torch.inference_mode()
    def generate_image_qa(self, *, text: str, images: list[Image.Image], max_new_tokens: int) -> dict[str, Any]:
        if not images:
            raise ValueError("PACT official repo image generation requires at least one image.")
        preprocess_start = time.perf_counter()
        image_tensor = self.process_images(images, self.image_processor, self.config)
        if isinstance(image_tensor, list):
            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        else:
            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
        preprocess_seconds = time.perf_counter() - preprocess_start

        input_ids, attention_mask, _ = self._build_prompt(text, placeholder_count=len(images))
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        image_sizes = [image.size for image in images]
        generation_start = time.perf_counter()
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            images=image_tensor,
            image_sizes=image_sizes,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            temperature=0,
            do_sample=False,
            top_p=None,
            num_beams=1,
        )
        generation_seconds = time.perf_counter() - generation_start
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_len = int(output_ids.shape[-1])
        input_len = int(input_ids.shape[-1])
        visual_tokens = None
        if torch.is_tensor(image_tensor):
            visual_tokens = int(image_tensor.shape[0])
        return {
            "text": decoded,
            "input_tokens": input_len,
            "visual_tokens": visual_tokens,
            "new_tokens": max(0, output_len - input_len) if output_len >= input_len else output_len,
            "processor_seconds": preprocess_seconds,
            "generation_seconds": generation_seconds,
        }

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
            raise FileNotFoundError(f"PACT official repo frame directory contains no image frames: {path}")
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
                f"PACT official repo frame directory has inconsistent frame shapes: "
                f"{path}; first={first_shape} mismatched_example={mismatched[0]}"
            )
        return np.stack(arrays, axis=0)

    def load_video(self, video_path: str) -> tuple[np.ndarray, str]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"PACT official repo video file not found: {path}")
        if path.is_dir():
            return self._load_frame_directory(path), "frame_directory"
        try:
            vr = VideoReader(str(path), ctx=cpu(0))
            total_frame_num = len(vr)
            if total_frame_num <= 0:
                raise RuntimeError(f"PACT official repo decoded zero frames from video: {path}")
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
        frame_start = time.perf_counter()
        frames, frame_decoder = self.load_video(video_path)
        frame_seconds = time.perf_counter() - frame_start
        preprocess_start = time.perf_counter()
        pixel_values = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        image_tensor = [pixel_values.to(device=self.device, dtype=torch.float16)]
        preprocess_seconds = time.perf_counter() - preprocess_start

        input_ids, attention_mask, stop_str = self._build_prompt(text, placeholder_count=1)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        stopping_criteria = [self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]
        self.config.mm_spatial_pool_stride = self.cfg.mm_spatial_pool_stride
        self.config.mm_spatial_pool_mode = self.cfg.mm_spatial_pool_mode

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


def load_pact_official_repo_model(args: Any, device: torch.device) -> tuple[PACTOfficialRepoRuntime, dict[str, Any]]:
    cfg = PACTOfficialRepoConfig(
        repo_root=Path(args.pact_official_repo_dir),
        pretrained=args.pact_official_pretrained,
        config_path=Path(args.pact_official_config_path),
        conv_template=args.pact_official_conv_template,
        model_name=args.pact_official_model_name,
        max_frames_num=int(getattr(args, "num_video_frames", 32)),
        attn_implementation=args.pact_official_attn_implementation,
        cutoff=float(args.pact_official_cutoff),
        pruning_tokeep_percentage_value=float(args.pact_official_pruning_tokeep_percentage_value),
    )
    runtime = PACTOfficialRepoRuntime(cfg, device)
    policy = {
        "method": "pact_official_repo",
        "source_repo": f"orailix/PACT@{DEFAULT_PACT_COMMIT}",
        "source_tree": str(cfg.repo_root),
        "model_loader": "PACT LLaVA-NeXT llava.model.builder.load_pretrained_model",
        "pretrained": cfg.pretrained,
        "config_path": str(cfg.config_path),
        "resolved_pact_config": runtime.pact_config,
        "conv_template": cfg.conv_template,
        "model_name": cfg.model_name,
        "attn_implementation": cfg.attn_implementation,
        "cutoff": cfg.cutoff,
        "pruning_tokeep_percentage_value": cfg.pruning_tokeep_percentage_value,
        "layer_for_reduction": runtime.pact_config.get("layer_for_reduction"),
        "supports_batch_generation": False,
        "frame_sampling": "PACT official repo uniform decord sampling from video path",
    }
    return runtime, policy
