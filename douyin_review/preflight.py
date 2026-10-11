"""Small paid capability checks for the three Model Studio API paths."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .models import Bailian


def main() -> None:
    parser = argparse.ArgumentParser(description="百炼按量计费工作空间的小样本接口检查（会产生少量费用）")
    parser.add_argument("--skip-video", action="store_true", help="跳过视频输入检查")
    args = parser.parse_args()
    model = Bailian.from_env()
    with tempfile.TemporaryDirectory(prefix="video-review-preflight-") as directory:
        root = Path(directory)
        picture = root / "image.jpg"
        image = Image.new("RGB", (320, 240), color="#e75b43")
        ImageDraw.Draw(image).text((30, 100), "TEST 2026", fill="white")
        image.save(picture)
        observed = model.describe_frames([{"id": "preflight_0", "scene_id": "test",
                                          "timestamp_ms": 0, "path": str(picture)}])
        if not observed or not observed[0]["description"]:
            raise RuntimeError("图片视觉调用未返回描述")
        print("图片视觉接口：成功")
        evidence = [{"id": "visual_test", "start_ms": 0, "end_ms": 1000,
                     "text": observed[0]["description"], "source": "visual"}]
        summary = model.summarize(evidence, 1000, {"title": "接口检查"})
        if not summary.get("summary"):
            raise RuntimeError("文本总结未返回摘要")
        print("文本总结接口：成功")
        audio = root / "tone.mp3"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1", "-ac", "1", "-ar", "16000",
                        "-c:a", "libmp3lame", str(audio)], check=True)
        model.transcribe([{"path": str(audio), "offset_ms": 0, "duration_ms": 1000}])
        print("语音接口：成功（输入为无语音测试音，不检验识别文字）")
        if not args.skip_video:
            video = root / "color.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=blue:s=320x240:d=4:r=2", "-c:v", "libx264", str(video)], check=True)
            result = model.describe_clip(str(video), 0, 4000)
            if not result["description"]:
                raise RuntimeError("视频视觉调用未返回描述")
            print("视频视觉接口：成功")


if __name__ == "__main__":
    main()
