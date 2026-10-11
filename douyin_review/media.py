"""Deterministic downloading, probing, sampling, and frame ranking."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import httpx


class MediaError(RuntimeError):
    pass


def command(args: list[str], *, timeout: int = 600) -> str:
    try:
        completed = subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout)
        return completed.stdout.strip()
    except FileNotFoundError as exc:
        raise MediaError(f"缺少可执行程序：{args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"命令执行超时：{args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()[-1200:]
        raise MediaError(f"{args[0]} 执行失败：{details}") from exc


_SHARE_URL = re.compile(r"https?://[^\s<>\"'，,]+")
_VIDEO_PATH = re.compile(r"^/(?:share/)?video/(\d+)(?:/|$)")
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"


def _douyin_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"douyin.com", "iesdouyin.com"} or host.endswith((".douyin.com", ".iesdouyin.com"))


def _canonical_video_url(url: str) -> str | None:
    if not _douyin_host(url):
        return None
    parsed = urlparse(url)
    matched = _VIDEO_PATH.match(parsed.path)
    if matched:
        return f"https://www.douyin.com/video/{matched.group(1)}"
    modal_id = parse_qs(parsed.query).get("modal_id", [""])[0]
    if parsed.path.startswith("/jingxuan/search/") and modal_id.isdigit():
        return f"https://www.douyin.com/video/{modal_id}"
    return None


def _resolve_short_http(url: str) -> str:
    current = url
    with httpx.Client(timeout=25, headers={"User-Agent": _BROWSER_UA}, follow_redirects=False) as client:
        for _ in range(8):
            if not _douyin_host(current) or urlparse(current).scheme != "https":
                raise MediaError("短链跳转到了非抖音 HTTPS 地址")
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise MediaError("抖音短链没有返回跳转地址")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
            canonical = _canonical_video_url(current)
            if canonical:
                return canonical
            raise MediaError("短链未跳转到可识别的单条抖音视频")
    raise MediaError("抖音短链跳转次数过多")


def _resolve_short_chrome(url: str, profile: str, interactive: bool) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise MediaError("短链浏览器回退需要安装 playwright") from exc
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(profile, channel="chrome", headless=not interactive)
        try:
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            canonical = _canonical_video_url(page.url)
            if not canonical:
                raise MediaError("Chrome 未能将短链解析为单条抖音视频")
            return canonical
        finally:
            context.close()


def douyin_url(text: str, chrome_profile: str | None = None, interactive_browser: bool = False) -> str:
    """Find one Douyin link in copied share text and resolve it to a video URL."""
    candidates = []
    for match in _SHARE_URL.finditer(text):
        candidate = match.group(0).rstrip("。；;）)】]}>！？!?")
        if _douyin_host(candidate):
            candidates.append(candidate)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise MediaError("分享文字中没有找到抖音链接")
    if len(candidates) != 1:
        raise MediaError("分享文字中包含多个抖音链接，请只提供一个视频")
    url = candidates[0]
    canonical = _canonical_video_url(url)
    if canonical:
        return canonical
    if (urlparse(url).hostname or "").lower() != "v.douyin.com":
        raise MediaError("抖音链接不是可识别的单条视频地址")
    secure_url = "https://" + url.split("://", 1)[1]
    try:
        return _resolve_short_http(secure_url)
    except (httpx.HTTPError, MediaError) as exc:
        if chrome_profile:
            try:
                return _resolve_short_chrome(secure_url, chrome_profile, interactive_browser)
            except Exception as browser_exc:
                raise MediaError(f"短链解析失败：HTTP {type(exc).__name__}；Chrome {type(browser_exc).__name__}") from browser_exc
        raise MediaError(f"短链解析失败：{type(exc).__name__}: {str(exc)[:200]}") from exc


def _yt_dlp(url: str, directory: Path, cookie_profile: str | None = None, *, referer: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    args = [
        "yt-dlp", "--no-playlist", "--no-progress", "--no-overwrites", "--no-simulate",
        "--write-info-json", "--merge-output-format", "mp4",
        "--paths", str(directory), "-o", "source.%(ext)s" if referer else "%(id)s.%(ext)s",
        "--print", "after_move:filepath", "--socket-timeout", "25",
        "--retries", "2", "--extractor-retries", "2",
    ]
    if cookie_profile:
        args += ["--cookies-from-browser", f"chrome:{cookie_profile}"]
    if referer:
        args += ["--referer", referer]
    args += [url]
    output = command(args, timeout=480)
    paths = [Path(line) for line in output.splitlines() if Path(line).is_file()]
    if not paths:
        raise MediaError("yt-dlp 未返回可读取的下载文件")
    return paths[-1].resolve()


def _capture_media_url(url: str, profile: str, interactive: bool) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise MediaError("Chrome 页面回退需要安装 playwright 可选依赖") from exc
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            profile, channel="chrome", headless=not interactive,
            args=["--disable-blink-features=AutomationControlled"],
            user_agent=_BROWSER_UA, locale="zh-CN",
        )
        try:
            page = context.new_page()
            candidates: list[str] = []

            def record(response: object) -> None:
                response_url = getattr(response, "url", "")
                content_type = getattr(response, "headers", {}).get("content-type", "")
                if "video/" in content_type or ".mp4" in response_url.split("?")[0]:
                    candidates.append(response_url)

            page.on("response", record)
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            if interactive:
                input("请在 Chrome 中完成必要的登录或验证，再按回车继续：")
            try:
                page.wait_for_function(
                    "() => [...document.querySelectorAll('video')].some(v => (v.currentSrc || v.src).startsWith('http'))",
                    timeout=20000,
                )
            except Exception:
                pass
            current = page.locator("video").evaluate_all(
                "els => els.map(el => el.currentSrc || el.src).filter(src => src.startsWith('http'))"
            )
            video_id = urlparse(url).path.rstrip("/").split("/")[-1]
            options = current + candidates
            matching = [item for item in options if video_id and video_id in item]
            vod = [item for item in matching if "douyinvod.com" in (urlparse(item).hostname or "")]
            media_url = (vod or matching or current or candidates or [""])[0]
            description = page.locator('meta[name="description"]').first.get_attribute("content") if page.locator('meta[name="description"]').count() else ""
            return {"media_url": media_url, "page_title": page.title().removesuffix(" - 抖音"),
                    "page_description": description or ""}
        finally:
            context.close()


def download_video(url: str, run_dir: Path, chrome_profile: str | None,
                   interactive_browser: bool, *, platform: str = "douyin",
                   source_url: str | None = None) -> dict:
    """Download one supported video, keeping signed access URLs out of reports."""
    if platform == "douyin":
        url = douyin_url(url)
    public_url = source_url or url
    media_dir = run_dir / "media"
    failures: list[str] = []
    attempts: list[tuple[str, str | None, str | None]] = [(url, None, None)]
    if chrome_profile and platform != "kuaishou":
        attempts.append((url, chrome_profile, None))
    for target, profile, referer in attempts:
        try:
            path = _yt_dlp(target, media_dir, profile, referer=referer)
            return {"source_url": public_url, "media_path": str(path),
                    "platform": platform,
                    "download_method": "yt-dlp+chrome-cookies" if profile else "yt-dlp"}
        except MediaError as exc:
            failures.append(str(exc))
    if chrome_profile:
        for _ in range(3 if platform == "kuaishou" else 1):
            try:
                page_data = _capture_media_url(url, chrome_profile, interactive_browser)
                media_url = page_data["media_url"]
                if media_url.startswith("http"):
                    path = _yt_dlp(media_url, media_dir, None, referer=url)
                    return {"source_url": public_url, "media_path": str(path), "platform": platform,
                            "download_method": "chrome-media-url+yt-dlp",
                            "page_title": page_data["page_title"],
                            "page_description": page_data["page_description"]}
                failures.append("Chrome 页面中未发现可下载的媒体地址")
            except Exception as exc:
                failures.append(f"Chrome 页面回退失败：{type(exc).__name__}: {str(exc)[:300]}")
    raise MediaError("下载失败；请确认视频可访问。" + " | ".join(failures[-3:]))


def probe(path: str | Path) -> dict:
    raw = command(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], timeout=60)
    data = json.loads(raw)
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise MediaError("ffprobe 未读到有效视频时长")
    streams = data.get("streams", [])
    videos = [item for item in streams if item.get("codec_type") == "video"]
    if not videos:
        raise MediaError("下载文件中没有视频流")
    return {
        "duration_ms": round(duration * 1000),
        "width": videos[0].get("width"), "height": videos[0].get("height"),
        "has_audio": any(item.get("codec_type") == "audio" for item in streams),
        "subtitle_streams": [item for item in streams if item.get("codec_type") == "subtitle"],
        "video_codec": videos[0].get("codec_name"),
    }


def extract_subtitles(media_path: str, run_dir: Path, streams: list[dict]) -> list[str]:
    output: list[str] = []
    for index, stream in enumerate(streams):
        destination = run_dir / "subtitles" / f"stream_{index}.srt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            command(["ffmpeg", "-y", "-v", "error", "-i", media_path, "-map", f"0:{stream['index']}", str(destination)], timeout=120)
            if destination.is_file() and destination.stat().st_size:
                output.append(str(destination))
        except MediaError:
            continue
    return output


def make_audio_chunks(media_path: str, run_dir: Path, duration_ms: int, has_audio: bool) -> list[dict]:
    if not has_audio:
        return []
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    duration_s = duration_ms / 1000
    chunks: list[dict] = []
    start = 0.0
    number = 0
    while start < duration_s:
        length = min(180.0, duration_s - start)
        destination = audio_dir / f"chunk_{number:03d}.mp3"
        command(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", media_path,
                 "-t", f"{length:.3f}", "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "libmp3lame", "-b:a", "64k", str(destination)], timeout=180)
        if destination.stat().st_size > 7_000_000:
            raise MediaError("音频分段超过安全大小限制")
        chunks.append({"path": str(destination), "offset_ms": round(start * 1000), "duration_ms": round(length * 1000)})
        if start + length >= duration_s:
            break
        start += length - 1.0
        number += 1
    return chunks


def detect_scenes(media_path: str, duration_ms: int) -> list[dict]:
    from scenedetect import detect
    from scenedetect.detectors import AdaptiveDetector

    pairs = detect(media_path, AdaptiveDetector(min_scene_len="0.4s"), show_progress=False)
    scenes = [{"id": f"scene_{i:03d}", "start_ms": round(start.seconds * 1000),
               "end_ms": round(end.seconds * 1000)} for i, (start, end) in enumerate(pairs)]
    return scenes or [{"id": "scene_000", "start_ms": 0, "end_ms": duration_ms}]


def _frame_at(capture: object, timestamp_ms: int):
    import cv2

    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    success, frame = capture.read()
    return frame if success else None


def _frame_quality(frame: object) -> float:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    exposure = max(0.0, 1.0 - abs(brightness - 128) / 128)
    return math.log1p(sharpness) + 2.0 * exposure


def _shrink_frame(frame: object, max_width: int = 1080):
    """Limit analysis-image size while preserving the original downloaded video."""
    import cv2

    width = frame.shape[1]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def sample_video(media_path: str, run_dir: Path, scenes: list[dict], duration_ms: int) -> dict:
    import cv2

    capture = cv2.VideoCapture(media_path)
    if not capture.isOpened():
        raise MediaError("OpenCV 无法解码视频")
    candidate_dir = run_dir / "candidates"
    ocr_dir = run_dir / "ocr_samples"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    ocr_samples: list[dict] = []
    try:
        for scene_index, scene in enumerate(scenes):
            start, end = scene["start_ms"], scene["end_ms"]
            length = end - start
            fractions = [0.25, 0.5, 0.75] if length > 1000 else [0.5]
            positions = {min(duration_ms - 1, round(start + length * fraction)) for fraction in fractions}
            if scene_index == 0:
                positions.add(min(duration_ms - 1, start + 150))
            if scene_index == len(scenes) - 1:
                positions.add(max(start, min(duration_ms - 1, end - 150)))
            if length > 8000:
                positions.update(range(start + 1250, end, 2500))
            for j, ms in enumerate(sorted(positions)):
                frame = _frame_at(capture, ms)
                if frame is None:
                    continue
                frame = _shrink_frame(frame)
                path = candidate_dir / f"{scene['id']}_{j}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                candidates.append({"id": f"{scene['id']}_{j}", "scene_id": scene["id"],
                                   "timestamp_ms": ms, "path": str(path), "quality": _frame_quality(frame)})
        for ms in range(0, duration_ms, 500):
            frame = _frame_at(capture, ms)
            if frame is None:
                continue
            frame = _shrink_frame(frame)
            path = ocr_dir / f"{ms:09d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
            ocr_samples.append({"timestamp_ms": ms, "path": str(path)})
    finally:
        capture.release()
    if not candidates:
        raise MediaError("未能提取任何视频帧")
    return {"scenes": scenes, "candidates": candidates, "ocr_samples": ocr_samples}


def select_keyframes(candidates: list[dict], ocr_spans: list[dict], asr_segments: list[dict], max_frames: int = 24) -> list[dict]:
    import imagehash
    from PIL import Image

    by_scene: dict[str, list[dict]] = {}
    for item in candidates:
        by_scene.setdefault(item["scene_id"], []).append(item)
    priorities: list[tuple[float, dict, str]] = []
    for group in by_scene.values():
        top = max(group, key=lambda item: item["quality"])
        priorities.append((100 + top["quality"], top, "镜头代表画面"))
        for item in group:
            novelty = sum(1 for span in ocr_spans if abs(span["start_ms"] - item["timestamp_ms"]) <= 700)
            speech = sum(1 for segment in asr_segments if segment["start_ms"] <= item["timestamp_ms"] <= segment["end_ms"] and len(segment["text"]) >= 12)
            priorities.append((item["quality"] + 3 * novelty + min(speech, 1), item, "文字或语音对应画面"))
    selected: list[dict] = []
    hashes: list[object] = []
    seen_ids: set[str] = set()
    for _, item, reason in sorted(priorities, key=lambda entry: entry[0], reverse=True):
        if item["id"] in seen_ids:
            continue
        digest = imagehash.phash(Image.open(item["path"]))
        if any(digest - previous <= 6 for previous in hashes):
            continue
        seen_ids.add(item["id"])
        hashes.append(digest)
        selected.append({**item, "reason": reason})
        if len(selected) >= max_frames:
            break
    return sorted(selected, key=lambda item: item["timestamp_ms"])
