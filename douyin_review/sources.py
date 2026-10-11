"""Resolve one supported video URL from a direct link or copied share text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from .media import MediaError, douyin_url


_SHARE_URL = re.compile(r"https?://[^\s<>\"'，,]+")
_BILIBILI_VIDEO = re.compile(r"^/video/(BV[A-Za-z0-9]{10}|av\d+)(?:/|$)", re.IGNORECASE)
_KUAISHOU_VIDEO = re.compile(r"^/short-video/([A-Za-z0-9]+)(?:/|$)")
_XIAOHONGSHU_VIDEO = re.compile(r"^/(?:explore|discovery/item)/([0-9a-fA-F]{24})(?:/|$)")


@dataclass(frozen=True)
class VideoSource:
    platform: str
    source_url: str
    download_url: str
    input_kind: str


def _platform(host: str) -> str | None:
    domains = {
        "douyin": ("douyin.com", "iesdouyin.com"),
        "bilibili": ("bilibili.com",),
        "kuaishou": ("kuaishou.com",),
        "xiaohongshu": ("xiaohongshu.com",),
    }
    for platform, names in domains.items():
        if any(host == name or host.endswith("." + name) for name in names):
            return platform
    return None


def resolve_video_source(text: str, chrome_profile: str | None = None,
                         interactive_browser: bool = False) -> VideoSource:
    """Validate the platform and retain access tokens only in the internal download URL."""
    candidates: list[tuple[str, str]] = []
    for match in _SHARE_URL.finditer(text):
        url = match.group(0).rstrip("。；;）)】]}>！？!?")
        host = (urlparse(url).hostname or "").lower()
        platform = _platform(host)
        if platform:
            candidates.append((url, platform))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise MediaError("输入中没有找到受支持的视频链接（B站、抖音、快手、小红书）")
    if len(candidates) != 1:
        raise MediaError("输入中包含多个受支持的视频链接，请只提供一个视频")

    url, platform = candidates[0]
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise MediaError("视频链接必须使用 HTTP 或 HTTPS")
    kind = "url" if text.strip() == url else "share-text"
    if platform == "douyin":
        canonical = douyin_url(url, chrome_profile, interactive_browser)
        return VideoSource(platform, canonical, canonical, kind)
    if platform == "bilibili":
        matched = _BILIBILI_VIDEO.match(parsed.path)
        if not matched:
            raise MediaError("B站链接不是可识别的单条视频地址")
        canonical = f"https://www.bilibili.com/video/{matched.group(1)}/"
        return VideoSource(platform, canonical, canonical, kind)
    if platform == "kuaishou":
        matched = _KUAISHOU_VIDEO.match(parsed.path)
        if not matched:
            raise MediaError("快手链接不是可识别的单条短视频地址")
        canonical = f"https://www.kuaishou.com/short-video/{matched.group(1)}"
        return VideoSource(platform, canonical, url, kind)
    matched = _XIAOHONGSHU_VIDEO.match(parsed.path)
    if not matched:
        raise MediaError("小红书链接不是可识别的单条视频地址")
    canonical = f"https://www.xiaohongshu.com/explore/{matched.group(1)}"
    return VideoSource(platform, canonical, url, kind)
