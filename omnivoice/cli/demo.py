#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Output folder for generated audio (openable from the WebUI)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.abspath("outputs")


def _open_output_folder() -> str:
    """Open the generated-audio output folder in the OS file manager."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(OUTPUT_DIR)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", OUTPUT_DIR])
        else:
            subprocess.Popen(["xdg-open", OUTPUT_DIR])
    except Exception as e:
        return f"無法開啟資料夾: {e}"
    return f"已開啟: {OUTPUT_DIR}"


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    # -- named voice library: cached & disk-persisted voice-clone prompts --
    # Creating a clone prompt costs the "reference processing" time every call.
    # Saving it under a name lets a few fixed voices be processed once and
    # reused, and keeps the reference length constant (steadier shapes).
    VOICES_DIR = os.path.abspath("voices")
    NEW_REF = "(新參考音 / New reference)"
    _voice_cache: Dict[str, VoiceClonePrompt] = {}

    # Serialize all GPU generation (user requests + background warmup) so they
    # never run model.generate concurrently on the same device.
    _gpu_lock = threading.Lock()

    def _voice_file(name: str) -> str:
        return os.path.join(VOICES_DIR, name + ".pt")

    def _voice_names():
        return [NEW_REF] + sorted(_voice_cache.keys())

    def _load_saved_voices():
        _voice_cache.clear()
        if not os.path.isdir(VOICES_DIR):
            return
        dev = model.audio_tokenizer.device
        for fn in os.listdir(VOICES_DIR):
            if not fn.endswith(".pt"):
                continue
            name = fn[:-3]
            try:
                d = torch.load(_voice_file(name), map_location=dev, weights_only=True)
                _voice_cache[name] = VoiceClonePrompt(
                    ref_audio_tokens=d["ref_audio_tokens"].to(dev),
                    ref_text=d["ref_text"],
                    ref_rms=float(d["ref_rms"]),
                )
            except Exception as e:
                logging.warning("Failed to load saved voice %s: %s", fn, e)

    def _save_voice(name, ref_audio, ref_text):
        name = (name or "").strip()
        if not name:
            return "請先輸入聲音名稱。", gr.update()
        if not ref_audio:
            return "請先上傳參考音頻再儲存。", gr.update()
        try:
            p = model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=(ref_text or None)
            )
        except Exception as e:
            return f"建立失敗: {type(e).__name__}: {e}", gr.update()
        _voice_cache[name] = p
        os.makedirs(VOICES_DIR, exist_ok=True)
        torch.save(
            {
                "ref_audio_tokens": p.ref_audio_tokens.cpu(),
                "ref_text": p.ref_text,
                "ref_rms": p.ref_rms,
            },
            _voice_file(name),
        )
        return (
            f"已儲存聲音「{name}」，之後從上方下拉選單直接選用即可（免再處理參考音）。",
            gr.update(choices=_voice_names(), value=name),
        )

    def _delete_voice(name):
        if not name or name == NEW_REF or name not in _voice_cache:
            return "請先在下拉選單選一個已存的聲音。", gr.update()
        _voice_cache.pop(name, None)
        try:
            os.remove(_voice_file(name))
        except OSError:
            pass
        return f"已刪除聲音「{name}」。", gr.update(choices=_voice_names(), value=NEW_REF)

    _load_saved_voices()

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        voice_name=None,
        seed=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        t0 = time.perf_counter()
        if mode == "clone":
            if voice_name and voice_name in _voice_cache:
                # Reuse a saved voice -> skip reference processing entirely.
                kw["voice_clone_prompt"] = _voice_cache[voice_name]
            elif ref_audio:
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            else:
                return None, "請上傳參考音頻，或從下拉選單選一個已存的聲音。"
        prompt_time = time.perf_counter() - t0

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        with _gpu_lock:
            # Fix the sampling RNG so a given seed reproduces the same take.
            if seed is not None and int(seed) >= 0:
                torch.manual_seed(int(seed))
            t1 = time.perf_counter()
            try:
                audio = model.generate(**kw)
            except Exception as e:
                return None, f"Error: {type(e).__name__}: {e}"
            gen_time = time.perf_counter() - t1

        waveform = (audio[0] * 32767).astype(np.int16)
        dur = len(waveform) / sampling_rate
        total = prompt_time + gen_time
        gen_rtf = gen_time / dur if dur > 0 else 0.0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = datetime.now().strftime("omnivoice_%Y%m%d_%H%M%S.wav")
        sf.write(os.path.join(OUTPUT_DIR, fname), waveform, sampling_rate)

        status = (
            f"Done. 總耗時 {total:.2f}s = 參考音處理 {prompt_time:.2f}s"
            f" + 生成 {gen_time:.2f}s · 音頻 {dur:.2f}s · 生成RTF {gen_rtf:.2f}"
            f" · 已存 {fname}"
        )
        return (sampling_rate, waveform), status

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # Manual warmup: pre-compile the JIT kernels for the audio-length buckets
    # that real generations use, plus the one-time per-process GEMM tuning. Uses
    # the voice currently selected in the Voice Clone tab (a saved voice, or the
    # uploaded reference) so the warmed shapes match what the user will generate.
    #
    # Crucial: real text picks a target = integer multiple of the bucket frame
    # count (see _bucket_tokens). frame_rate is ~16.67 Hz (non-integer), so
    # warming by clean second values (int(sec*fr)) lands on DIFFERENT token
    # counts and misses. Here we warm exactly those bucket multiples instead,
    # mapping each back to a duration that floors to the same token count.
    # The GPU lock keeps it from clashing with a concurrent request; yields let
    # the status box show progress across the (multi-bucket, ~minutes) run.
    def _run_warmup(voice_name=None, ref_audio=None, ref_text=None):
        voice_kw = {}
        via = "auto voice"
        if voice_name and voice_name in _voice_cache:
            voice_kw["voice_clone_prompt"] = _voice_cache[voice_name]
            via = f"「{voice_name}」"
        elif ref_audio:
            try:
                voice_kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=(ref_text or None)
                )
                via = "上傳參考音"
            except Exception as e:
                yield f"暖機失敗：建立參考音 {type(e).__name__}: {e}"
                return

        fr = model.audio_tokenizer.config.frame_rate
        bucket_s = float(os.environ.get("OMNIVOICE_DURATION_BUCKET") or 0)
        if bucket_s <= 0:
            yield (
                "未啟用長度分桶（OMNIVOICE_DURATION_BUCKET），暖機無法對齊形狀。"
                "請先在 start_official.bat 設定後重啟再暖機。"
            )
            return
        bucket_frames = max(1, int(bucket_s * fr))
        # Warm every bucket up to the chunk threshold; longer text auto-chunks
        # into ~15s pieces whose shapes fall inside this range.
        max_s = float(os.environ.get("OMNIVOICE_WARMUP_MAX_SECONDS", "30"))
        n_max = max(1, round(max_s / bucket_s))
        cfg = OmniVoiceGenerationConfig(num_step=32)
        results = []
        for n in range(1, n_max + 1):
            target = n * bucket_frames
            d = (target + 0.5) / fr  # floors to exactly `target` tokens
            label = f"{n * bucket_s:.0f}s"
            try:
                with _gpu_lock:
                    s = time.perf_counter()
                    model.generate(
                        text="預熱",
                        language="Chinese",
                        duration=d,
                        generation_config=cfg,
                        **voice_kw,
                    )
                results.append(f"{label}={time.perf_counter() - s:.1f}s")
            except Exception as e:  # warmup must never crash the server
                logging.warning("Warmup %s failed: %s", label, e)
                results.append(f"{label}=失敗")
            yield f"暖機中（{via}）{n}/{n_max}: " + "  ".join(results)
        final = f"暖機完成（{via}）: " + "  ".join(results)
        logging.info(final)
        yield final

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Chinese"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set."
                "｜白話：告訴播音員「請你念快一點（或慢一點）」。設了 Duration 就以 Duration 為準。",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                    "｜白話：告訴播音員「這段稿子你只能用 5 秒念完」。"
                    "留空＝用 Speed 自動估時長。"
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=16,
                step=1,
                label="Inference Steps",
                info="Default: 16 (fast). Raise (e.g. 32) for better quality."
                "｜白話：愈少，生成愈快，但音質可能粗糙；愈多，生成愈慢，音質更細緻。",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0."
                "｜白話：數值越高→模型越緊扣你給的條件（參考音色、文字內容），"
                "結果更「準確」，但可能聽起來比較生硬；"
                "數值越低→模型更自由發揮，聲音可能更自然，但有時會飄離參考音色。",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
            sd = gr.Number(
                value=-1,
                precision=0,
                label="Seed / 隨機種子",
                info="-1＝每次隨機（預設）。\n"
                "• 想固定：填一個整數（例如 42），同文字、同步數，每次生成都一樣。\n"
                "• 想試不同版本：把種子改成 1、2、3… 逐個生成，聽到滿意的就記下那個數字，"
                "之後填回去就能重現。\n"
                "• -1：每次隨機。注意這模式下你無法回放剛剛那一版（不知道用了哪個種子）；"
                "要「挑一版留著」時，請用固定整數、別用 -1。",
            )
        return ns, gs, dn, sp, du, pp, po, sd

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_voice = gr.Dropdown(
                            label="常用聲音 / Saved Voice",
                            choices=_voice_names(),
                            value=NEW_REF,
                            info="選一個已存的聲音可直接重用（免再處理參考音）；"
                            "選「新參考音」則用下方上傳的音頻。",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional)" " / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        with gr.Row():
                            vc_voice_name = gr.Textbox(
                                label="聲音名稱 / Name",
                                placeholder="例如：旁白男聲",
                                scale=3,
                            )
                            vc_save = gr.Button("💾 儲存為常用聲音", scale=1)
                            vc_delete = gr.Button("🗑 刪除選取聲音", scale=1)
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                            vc_sd,
                        ) = _gen_settings()
                        with gr.Row():
                            vc_btn = gr.Button(
                                "Generate / 生成", variant="primary", scale=3
                            )
                            vc_gen_stop = gr.Button("⏹ 中止生成", scale=1)
                        with gr.Row():
                            vc_warmup = gr.Button("🔥 暖機 / Warm up")
                            vc_stop = gr.Button("⏹ 中止暖機")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)
                        vc_open = gr.Button("📂 開啟輸出資料夾")
                        vc_open.click(_open_output_folder, outputs=vc_status)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, voice, instruct,
                    ns, gs, dn, sp, du, pp, po, sd
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                        voice_name=(voice if voice and voice != NEW_REF else None),
                        seed=sd,
                    )

                vc_gen_evt = vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_voice,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                        vc_sd,
                    ],
                    outputs=[vc_audio, vc_status],
                )
                # Stop waiting on a generation (cancels the queued/running event).
                # An in-flight GPU op still finishes in the background — we never
                # force-kill it (that would trigger LLVM ERROR); the result is
                # just discarded so the UI is free again.
                vc_gen_stop.click(
                    lambda: "已中止：若正在生成，GPU 會在背景算完才釋放。",
                    outputs=vc_status,
                    cancels=[vc_gen_evt],
                )

                vc_save.click(
                    _save_voice,
                    inputs=[vc_voice_name, vc_ref_audio, vc_ref_text],
                    outputs=[vc_status, vc_voice],
                )
                vc_delete.click(
                    _delete_voice,
                    inputs=[vc_voice],
                    outputs=[vc_status, vc_voice],
                )
                vc_warmup_evt = vc_warmup.click(
                    _run_warmup,
                    inputs=[vc_voice, vc_ref_audio, vc_ref_text],
                    outputs=vc_status,
                )
                # Graceful stop: cancels the warmup generator. It can only stop
                # at a yield boundary, so the bucket currently compiling finishes
                # first (this AVOIDS force-killing a live GPU op -> LLVM ERROR).
                vc_stop.click(
                    lambda: "已要求中止：會在目前這格暖完後停止。",
                    outputs=vc_status,
                    cancels=[vc_warmup_evt],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                            vd_sd,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)
                        vd_open = gr.Button("📂 開啟輸出資料夾")
                        vd_open.click(_open_output_folder, outputs=vd_status)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, sd, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                        seed=sd,
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                        vd_sd,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
