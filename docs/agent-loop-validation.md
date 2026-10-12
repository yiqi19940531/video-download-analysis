# Agent Loop 四平台验证记录

[返回项目主页](../README.md)

2026-10-12 使用用户给出的四条原始链接，分别从空运行目录开始执行 `video-review-agent`。程序实际重新下载原视频，完成镜头、语音、OCR、视觉和结构化分析。公开目录只保留文字报告与每条 3 张截图；原视频、全部候选帧、运行日志、SQLite 检查点及可能含临时访问参数的内部 JSON 留在本地。

## 从链接到报告

| 视频 | 下载方式 | 时长 | 镜头 / OCR 采样 | ASR / 视觉 / 总证据 | 报告 |
| --- | --- | ---: | ---: | ---: | --- |
| B站《学以乱用》 | `yt-dlp` | 78.62 秒 | 21 / 158 | 12 / 21 / 203 | [详情](../examples/agent-loop/bilibili/README.md) |
| 抖音《中国人会飞》 | 无头 Chrome + `yt-dlp` | 16.85 秒 | 1 / 34 | 2 / 2 / 35 | [详情](../examples/agent-loop/douyin/README.md) |
| 快手《哪个状元更有含金量》 | 无头 Chrome + `yt-dlp` | 57.67 秒 | 19 / 116 | 24 / 20 / 178 | [详情](../examples/agent-loop/kuaishou/README.md) |
| 小红书《酱板鸭反转短剧》 | `yt-dlp` | 39.40 秒 | 22 / 79 | 15 / 23 / 67 | [详情](../examples/agent-loop/xiaohongshu/README.md) |

这四条全新运行均由 `qwen3.8-flash` 决策模型选择工具。ASR 使用百炼 `qwen-audio-3.0-asr-flash`；图片、无声片段和结构化总结使用百炼 `qwen3.8-flash`；PaddleOCR 在本地处理画面文字。表中数字是保存的片段、观察或证据条数，不等于计费请求数。每条最终分析所引用的证据 ID 都存在，报告里的关键帧文件也都能读取。

| B站 33 秒 | 抖音 8 秒 | 快手 24 秒 | 小红书 16 秒 |
| --- | --- | --- | --- |
| ![B站代表帧](../examples/agent-loop/bilibili/frames/middle.jpg) | ![抖音侧手翻](../examples/agent-loop/douyin/frames/middle.jpg) | ![快手跑道场景](../examples/agent-loop/kuaishou/frames/middle.jpg) | ![小红书男子看书](../examples/agent-loop/xiaohongshu/frames/middle.jpg) |

## 循环决策和人工补充

正常运行会由模型逐轮选择 `resolve_source`、`download`、`probe`、音频/画面工具、`fuse_evidence`、`write_analysis`、`write_report` 和 `finish`。小红书首次完整运行在总结后还选了 `inspect_evidence`，然后才写报告；这一额外检查来自模型决策，不是固定 Workflow 图中的节点。

缺失信息的测试从输入“请分析这个视频”开始。模型要求具体的视频链接或本地文件路径；随后连续三次只回复“视频在网上，请继续分析”。每次来源解析都找不到可用链接，第三次补充后图返回 `human_input_exhausted`、命令退出码 2，没有第四次提问。成功推进任务的工具会重置连续反问计数；总体决策仍受 `--max-steps` 限制。实现见 [graph.py](../agent_loop/graph.py) 和 [cli.py](../agent_loop/cli.py)。

## 实测发现并修复的问题

1. **残缺文本**：小红书初稿的摘要和多个时间轴段落停在半句话；抖音有 3 个残缺分段，快手有 1 个。Agent Loop 增加了摘要与分段完整性检查。抖音、快手通过 `repair_structure` 使用对应的证据完成分段；小红书的定向分段修复接口返回空内容，Agent 选择强制重新分析，产生了完整分段。公开示例均使用最终报告，原始草稿只在本地。
2. **导入后的关键帧路径**：在小红书的复核运行中，重新总结时暴露出 `evidence.json` 内关键帧仍引用旧运行目录。`import_workflow_run` 现会一并重写这些路径，并有离线测试覆盖。最终交付的关键帧文件全部存在。
3. **无效补充的退出**：原来只有总步数上限；现在图记录连续人工提问次数。三次仍缺少有效来源时直接结束，避免无休止反问。

完整性检查能识别明显的半句和缺失文件，不能判定每个剧情解释一定正确。例如 B站节奏拟声词的 ASR 与画面字幕有差异，小红书角色关系也存在叙事歧义；示例报告保留这些不确定点，供人工复核。

## 可重复的本地检查

```bash
.venv/bin/python -m pytest -q
```

本次本地结果为 **34 passed**。生成公开示例时，使用 [`scripts/build_agent_loop_examples.py`](../scripts/build_agent_loop_examples.py) 从本地验证记录提取每条 3 张截图和文字报告；脚本会拒绝包含访问令牌或密钥的公开文本，并去掉报告中指向未上传关键帧的链接。Git 忽略原视频、运行缓存、检查点、浏览器配置和密钥文件。
