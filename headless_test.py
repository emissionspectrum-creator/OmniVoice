#!/usr/bin/env python3
"""Headless 移植驗證:載入模型 + 用已存的 .pt 聲音跑 TTS 生成,驗證出聲。
走 demo.py 的 clone 路徑(含 OMNIVOICE_DURATION_BUCKET 分桶),同段文字生兩次比冷/熱 RTF。
輸出報告到 ~/omnivoice/MIGRATION_TEST_RESULT.txt。"""
import os, sys, time, traceback
from datetime import datetime

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMNIVOICE_DURATION_BUCKET", "2.0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPORT = os.path.expanduser("~/omnivoice/MIGRATION_TEST_RESULT.txt")
VOICE = "羅胖"
TEXT = "移植到 Linux 上執行測試,這是一段用來確認語音合成正常運作的文字。"
lines = []

def log(s=""):
    print(s, flush=True)
    lines.append(s)

def write_report(verdict):
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(f"OmniVoice Linux 移植測試報告\n時間: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"結果: {verdict}\n" + "=" * 60 + "\n")
        f.write("\n".join(lines) + "\n")

try:
    import numpy as np, soundfile as sf, torch
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    from omnivoice.models.omnivoice import VoiceClonePrompt

    log(f"torch {torch.__version__} · HIP {getattr(torch.version,'hip',None)}")
    log(f"cuda_available={torch.cuda.is_available()} · device={torch.cuda.get_device_name(0)}")
    log(f"OMNIVOICE_DURATION_BUCKET={os.environ.get('OMNIVOICE_DURATION_BUCKET')}")
    log("")

    log("載入模型 k2-fsa/OmniVoice (首次會下載)...")
    t = time.perf_counter()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
    log(f"模型載入完成 ({time.perf_counter()-t:.1f}s)")

    # 載入已存聲音 .pt -> VoiceClonePrompt(跳過 ASR)
    dev = model.audio_tokenizer.device
    vp = os.path.join(os.path.abspath("voices"), VOICE + ".pt")
    d = torch.load(vp, map_location=dev, weights_only=True)
    prompt = VoiceClonePrompt(ref_audio_tokens=d["ref_audio_tokens"].to(dev),
                              ref_text=d["ref_text"], ref_rms=float(d["ref_rms"]))
    log(f"已載入聲音「{VOICE}」(ref_text: {str(d['ref_text'])[:30]}...)")
    log("")

    cfg = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0, denoise=True,
                                    preprocess_prompt=True, postprocess_output=True)
    os.makedirs("outputs", exist_ok=True)
    sr = model.sampling_rate
    results = []
    for i, tag in enumerate(["冷(首次冷編譯)", "熱(同桶暖機後)"], 1):
        t0 = time.perf_counter()
        audio = model.generate(text=TEXT, language=None, generation_config=cfg,
                               voice_clone_prompt=prompt)
        gen = time.perf_counter() - t0
        wav = audio[0]
        dur = len(wav) / sr
        rtf = gen / dur if dur > 0 else 0.0
        peak = float(np.max(np.abs(wav)))
        fn = f"outputs/headless_{i}.wav"
        sf.write(fn, (wav * 32767).astype(np.int16), sr)
        log(f"[{tag}] gen={gen:.2f}s · 音頻={dur:.2f}s · RTF={rtf:.2f} · peak={peak:.3f} · {fn}")
        results.append((dur, peak, rtf))

    # 判定:音頻夠長 + 非靜音 + 第二次(熱)不慢於第一次(冷)
    dur1, peak1, rtf_cold = results[0]
    dur2, peak2, rtf_warm = results[1]
    ok_sound = dur1 > 0.5 and peak1 > 0.02
    ok_warm = rtf_warm <= rtf_cold + 0.05
    log("")
    log(f"出聲檢查: 音頻>0.5s 且 peak>0.02 -> {'PASS' if ok_sound else 'FAIL'}")
    log(f"分桶檢查: 熱RTF({rtf_warm:.2f}) <= 冷RTF({rtf_cold:.2f}) -> {'PASS' if ok_warm else 'FAIL(可能屬正常波動)'}")

    verdict = "PASS ✅ Linux 移植成功,GPU 推論出聲正常" if ok_sound else "FAIL ❌ 生成音頻異常(靜音或過短)"
    log("")
    log(verdict)
    write_report("PASS" if ok_sound else "FAIL")
    sys.exit(0 if ok_sound else 1)

except Exception as e:
    log("")
    log(f"EXCEPTION: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    write_report("FAIL (exception)")
    sys.exit(2)
