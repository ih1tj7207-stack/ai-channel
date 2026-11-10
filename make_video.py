#!/usr/bin/env python3
"""AI雑談チャンネル 段階1: 台本テキスト -> 縦型MP4 (1080x1920)

- 音声: VOICEVOX ENGINE (http://127.0.0.1:50021) のずんだもん
- 映像: ホログラム/ニューラルネット風の背景 + 音声に反応する光の球 + 字幕
- 出力: output/video.mp4
使い方: python make_video.py script.txt
script.txt は1行1文。最初の行がフック(冒頭1〜3秒)になります。
"""
import json
import math
import subprocess
import sys
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ENGINE = "http://127.0.0.1:50021"
SPEAKER = 3  # ずんだもん(ノーマル)
W, H, FPS = 1080, 1920, 30
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CREDIT = "VOICEVOX:ずんだもん"
OUT = Path("output")
OUT.mkdir(exist_ok=True)


def synth(text: str, path: Path) -> None:
    q = urllib.request.Request(
        f"{ENGINE}/audio_query?speaker={SPEAKER}&text={urllib.parse.quote(text)}",
        method="POST",
    )
    query = json.loads(urllib.request.urlopen(q).read())
    query["speedScale"] = 1.05
    query["outputSamplingRate"] = 24000
    query["outputStereo"] = False
    s = urllib.request.Request(
        f"{ENGINE}/synthesis?speaker={SPEAKER}",
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    path.write_bytes(urllib.request.urlopen(s).read())


def build_audio(lines):
    """各文を合成して連結。字幕用の (開始, 終了, 文) と全体のサンプル列を返す。"""
    rate = 24000
    gap = np.zeros(int(rate * 0.35), dtype=np.int16)
    head = np.zeros(int(rate * 0.4), dtype=np.int16)
    chunks, timings, t = [head], [], len(head) / rate
    for i, line in enumerate(lines):
        p = OUT / f"seg{i}.wav"
        synth(line, p)
        with wave.open(str(p), "rb") as w:
            assert w.getframerate() == rate
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        timings.append((t, t + len(data) / rate, line))
        chunks += [data, gap]
        t += (len(data) + len(gap)) / rate
    chunks.append(np.zeros(int(rate * 0.6), dtype=np.int16))
    audio = np.concatenate(chunks)
    with wave.open(str(OUT / "voice.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.tobytes())
    return audio, rate, timings


def wrap(draw, text, font, max_w):
    rows, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > max_w and cur:
            rows.append(cur)
            cur = ch
        else:
            cur += ch
    rows.append(cur)
    return rows


def make_nodes(seed=7, n=46):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([60, 120], [W - 60, H - 120], size=(n, 2))
    vel = rng.normal(0, 0.35, size=(n, 2))
    return pts, vel


def render(audio, rate, timings, dur):
    n_frames = int(dur * FPS)
    pts, vel = make_nodes()
    f_sub = ImageFont.truetype(FONT_PATH, 66)
    f_small = ImageFont.truetype(FONT_PATH, 34)
    # 音量(0..1) をフレームごとに
    spf = rate // FPS
    amp = np.array([
        np.sqrt(np.mean(audio[i * spf:(i + 1) * spf].astype(np.float64) ** 2)) if i * spf < len(audio) else 0
        for i in range(n_frames)
    ])
    amp = np.clip(amp / (np.percentile(amp, 95) + 1e-9), 0, 1)
    amp_s = np.convolve(amp, np.ones(4) / 4, mode="same")

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(OUT / "voice.wav"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
         str(OUT / "video.mp4")],
        stdin=subprocess.PIPE,
    )
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, 760
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    base = np.zeros((H, W, 3), dtype=np.float32)
    base[..., 0] = 6 + 10 * (yy / H)
    base[..., 1] = 8 + 22 * (yy / H)
    base[..., 2] = 22 + 40 * (yy / H)

    for fi in range(n_frames):
        t = fi / FPS
        a = float(amp_s[fi])
        # 光の球(音声に反応)
        r = 170 + 70 * a
        glow = np.exp(-((dist - r) / (60 + 40 * a)) ** 2) * (0.55 + 0.45 * a)
        core = np.exp(-(dist / (r * 0.9)) ** 2) * 0.35
        frame = base.copy()
        frame[..., 0] += 40 * glow + 30 * core
        frame[..., 1] += 235 * glow + 150 * core
        frame[..., 2] += 255 * glow + 200 * core
        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))

        # ニューラルネット風の線と点
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        p = pts + vel * fi
        p[:, 0] = np.abs((p[:, 0] % (2 * (W - 120))) - (W - 120)) + 60
        p[:, 1] = np.abs((p[:, 1] % (2 * (H - 240))) - (H - 240)) + 120
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                dd = math.dist(p[i], p[j])
                if dd < 300:
                    al = int(110 * (1 - dd / 300) * (0.6 + 0.4 * a))
                    d.line([tuple(p[i]), tuple(p[j])], fill=(90, 230, 255, al), width=2)
        for x, y in p:
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(160, 250, 255, 200))
        # 回転する幾何学リング
        for k, rad in enumerate((r + 120, r + 190)):
            ang = t * (0.8 if k == 0 else -0.5)
            pts_ring = [(cx + rad * math.cos(ang + i * math.pi / 3), cy + rad * math.sin(ang + i * math.pi / 3))
                        for i in range(7)]
            d.line(pts_ring, fill=(120, 255, 240, 150), width=3)
        img = Image.alpha_composite(img.convert("RGBA"), layer)

        # 字幕
        d2 = ImageDraw.Draw(img)
        cur = next((s for s in timings if s[0] <= t <= s[1] + 0.25), None)
        if cur:
            rows = wrap(d2, cur[2], f_sub, W - 160)
            y0 = 1330
            for row in rows:
                tw = d2.textlength(row, font=f_sub)
                x0 = (W - tw) / 2
                d2.text((x0, y0), row, font=f_sub, fill=(255, 255, 255, 255),
                        stroke_width=7, stroke_fill=(0, 20, 40, 255))
                y0 += 92
        tw = d2.textlength(CREDIT, font=f_small)
        d2.text(((W - tw) / 2, H - 150), CREDIT, font=f_small, fill=(200, 235, 255, 230),
                stroke_width=3, stroke_fill=(0, 20, 40, 255))
        ff.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    script = Path(sys.argv[1] if len(sys.argv) > 1 else "script.txt")
    lines = [l.strip() for l in script.read_text(encoding="utf-8").splitlines() if l.strip()]
    audio, rate, timings = build_audio(lines)
    dur = len(audio) / rate
    render(audio, rate, timings, dur)
    print(f"done: {OUT / 'video.mp4'} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
