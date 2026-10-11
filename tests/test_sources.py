from __future__ import annotations

from pathlib import Path

import pytest

from douyin_review.media import MediaError, download_video
from douyin_review.sources import resolve_video_source


@pytest.mark.parametrize("url,platform,canonical", [
    ("https://www.bilibili.com/video/BV1yJaH6DEV1/?spm_id_from=search", "bilibili",
     "https://www.bilibili.com/video/BV1yJaH6DEV1/"),
    ("https://www.douyin.com/jingxuan/search/example?modal_id=7684591599804937466", "douyin",
     "https://www.douyin.com/video/7684591599804937466"),
    ("https://www.kuaishou.com/short-video/3x56qx9f9beqtmm?authorId=abc", "kuaishou",
     "https://www.kuaishou.com/short-video/3x56qx9f9beqtmm"),
    ("https://www.xiaohongshu.com/explore/69e09cb0000000001f0035b3?xsec_token=secret", "xiaohongshu",
     "https://www.xiaohongshu.com/explore/69e09cb0000000001f0035b3"),
])
def test_resolves_four_platforms(url: str, platform: str, canonical: str) -> None:
    result = resolve_video_source(url)
    assert result.platform == platform
    assert result.source_url == canonical
    assert result.input_kind == "url"
    assert result.download_url == (canonical if platform in {"bilibili", "douyin"} else url)


def test_share_text_has_exactly_one_supported_url() -> None:
    url = "https://www.bilibili.com/video/BV1yJaH6DEV1/"
    assert resolve_video_source(f"看看这个视频 {url} 复制打开").input_kind == "share-text"
    with pytest.raises(MediaError, match="多个"):
        resolve_video_source(url + " https://www.kuaishou.com/short-video/3x56qx9f9beqtmm")
    with pytest.raises(MediaError, match="没有找到"):
        resolve_video_source("https://example.com/video/123")


def test_kuaishou_uses_headless_media_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from douyin_review import media

    attempts = []
    captures = []

    def fake_yt_dlp(url, directory, cookie_profile=None, *, referer=None):
        attempts.append((url, cookie_profile, referer))
        if referer is None:
            raise MediaError("yt-dlp 不支持该页面")
        path = directory / "source.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return path

    monkeypatch.setattr(media, "_yt_dlp", fake_yt_dlp)
    def fake_capture(url, profile, interactive):
        captures.append(url)
        return {"media_url": "" if len(captures) == 1 else "https://cdn.example.com/video.mp4",
                "page_title": "快手样例", "page_description": "描述"}

    monkeypatch.setattr(media, "_capture_media_url", fake_capture)
    source = resolve_video_source("https://www.kuaishou.com/short-video/3x56qx9f9beqtmm?authorId=abc")
    result = download_video(source.download_url, tmp_path, "chrome-profile", False,
                            platform=source.platform, source_url=source.source_url)
    assert result["platform"] == "kuaishou"
    assert result["source_url"] == source.source_url
    assert result["download_method"] == "chrome-media-url+yt-dlp"
    assert len(attempts) == 2
    assert len(captures) == 2
    assert attempts[1][2] == source.download_url
