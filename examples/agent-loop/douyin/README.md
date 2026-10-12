# Agent Loop 示例：抖音｜中国人能飞

[返回项目主页](../../../README.md) · [完整模型报告](report.md)

这份结果来自 2026-10-12 对原始链接的一次**全新下载和完整分析**。报告由模型辅助生成，时间戳、证据 ID 和截图供人工核对。原视频不在仓库中。

- 原始页面：https://www.douyin.com/video/7684591599804937466
- 时长：16.85 秒；下载方式：`chrome-media-url+yt-dlp`。
- 镜头 1 个；候选帧 12 张；OCR 采样帧 34 张。
- ASR 2 条；画面文字 26 条；图片观察 2 条；时间轴证据 35 条。
- 本地最终结果有 12 张关键帧；本页只公开 3 张。

## Agent 实际做了什么

`qwen3.8-flash` 每轮依据检查点中的阶段状态选择一个受控动作；本次路径为：

`resolve_source → download → probe → sample_frames → transcribe → read_screen_text → describe_visuals → fuse_evidence → write_analysis → write_report → finish`

| 环节 | 本次作用 |
| --- | --- |
| 获取视频 | `chrome-media-url+yt-dlp` 下载，`ffprobe` 确认音视频和时长。 |
| 声音 | FFmpeg 制作音频块；百炼 `qwen-audio-3.0-asr-flash` 返回带时间戳的转写。 |
| 画面 | PySceneDetect/OpenCV 切镜头和抽帧；PaddleOCR 在本地读取文字。 |
| 视觉 | 百炼 `qwen3.8-flash` 描述镜头代表帧和最多三段无声短片。 |
| 结论 | 合并来源证据后，用 LangChain/Pydantic 调 `qwen3.8-flash` 生成结构、摘要和不确定项。 |
| 交付 | LangGraph 检查报告和证据引用；公开报告保留时间戳，截图只取三张。 |

**本次质量复核**：初稿有未完成的时间轴段落；repair_structure 依据所引证据补全并重写报告。复核路径：`import_workflow_run → repair_structure → write_report → finish`。下面展示的是最终版本。

## 这条视频讲了什么

视频拍摄于热带白沙滩，画面中多名年轻男子依次表演侧手翻（空翻）动作，背景为碧蓝海水、停泊的游艇帆船与绿色山丘。画外音从约1.2秒起反复高喊“中国人能飞”。

## 分段与证据

| 原视频时间 | 情节或观点 | 证据 ID |
| --- | --- | --- |
| 00:00–00:01 | 画面以广角镜头展示热带海滩全景：白色沙滩、碧蓝海水、远处游艇与帆船、绿色山丘与蓝天白云。一名穿白色T恤黑色短裤的男子站在沙滩上面朝镜头，画面底部持续显示白色“DJI OSMO POCKET 4”字样。 | visual_scene_000_0, ocr_0000 |
| 00:01–00:06 | 白衣男子背对镜头完成侧手翻，随后镜头转向沙滩另一侧，另一名穿浅绿色长袖上衣的男子也完成侧手翻。画外音开始反复喊“中国人能飞”。 | clip_1731_6231, asr_000_0000 |
| 00:06–00:12 | 一名穿浅灰色长袖上衣、深色短裤的男子在镜头前完成侧手翻：起势、腾空倒置、落地站稳。背景仍为碧蓝海水与停泊船只，画面下方持续显示设备水印。 | clip_7750_12250, visual_scene_000_3 |
| 00:12–00:16 | 灰衣男子面带笑容朝镜头走来后出画，镜头转向开阔海面，可见游泳者和涉水者。画外音继续喊“中国人能飞”。 | clip_12350_16850, asr_000_0001 |

## 选出的三张截图

**00:00**

![douyin opening.jpg](frames/opening.jpg)

**00:08**

![douyin middle.jpg](frames/middle.jpg)

**00:16**

![douyin ending.jpg](frames/ending.jpg)

## 不确定与人工复核

- 无法从画面证据确认该视频为单次连续拍摄还是多段拼接。
- 模型返回了一条未完成的不确定项；已剔除该片段，请结合原始证据人工复核。

[查看完整模型报告，包括视觉描述、语音和画面文字](report.md)。证据 ID 可定位模型所用记录，但不能独立证明其解释必然正确。
