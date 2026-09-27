# 验证记录

验证日期：2026-09-18。

| 检查 | 结果 |
|---|---|
| pytest | 48 passed，10 subtests passed |
| SDK 打包 | wheel 构建及无依赖安装成功 |
| 数据结构 | 原数据和重建数据均为 28 episode、300 QA，通过校验 |
| QA 重建 | 从原始 reference 数据和 rollout 证据重建，300 题逐字节一致 |
| 统一入口 | `rollout.py` 复用四房间 accepted run，reference → v2.5 导出通过，300 题逐字节一致 |
| 视频抽帧 | 真实数据前两题通过 `--dry-run --limit 2`，未调用 API |
| 模拟器 | 发布包 SDK 成功启动 FloorPlan1；RGB 为 640×480，metadata 含 77 个物体 |
| API | 本地 HTTP mock 验证多模态请求、503 重试、导出和断点恢复 |
| 错误处理 | 无效响应保留原题并返回退出码 2；`--resume --retry-failed` 可重试 |
| 前缀采样 | 用不同颜色标识合成视频的前缀和未来帧，确认请求只含前缀帧 |
| 字段一致性 | 除 question/video_evidence 外，原字段保持不变；拒绝带额外标签字段的响应 |
| 数据搬迁 | 润色目录改名后，视频路径和数据校验通过 |

原数据与两次重建结果的 `questions.jsonl` SHA-256 相同：

```text
f9b1a668e411cf391b861a286b37fc18d945b8fe0fc80e1e074aaf8c94f2345d
```

## 环境

离线测试使用 macOS arm64、Python 3.11.16，依赖版本为 numpy 2.4.6、PyYAML 6.0.3、Pillow 12.3.0、imageio 2.37.4、imageio-ffmpeg 0.6.0、pytest 9.1.1。

模拟器测试使用原 Linux x86_64 服务器、Python 3.11.15、Unity commit `4d2e1f1d04051fafcd9794b810f227551121253a`。默认 GPU 启动在 90 秒内未完成，选择 GPU 3 后成功。超时原因尚未确定。

本次修复了两处问题：SDK 在指定本地程序时忽略 platform 参数；历史 QA 测试会改写共享 builder，影响后续测试。前者修改见 [NOTICE](../NOTICE.md)，后者由 pytest fixture 恢复模块状态。生产入口仍在独立进程中调用各配方。

## 覆盖范围

仿真测试覆盖启动和单帧渲染；完整 QA 导出使用已有 accepted rollout，未从零渲染全部 28 段视频。SDK 在原服务器依赖环境中加载，尚未验证全新 Linux 主机的安装过程。

VLM 接口使用本地 mock 测试，真实视频只做了请求预演。外部模型的改写质量、语义等价性及改写后的基准表现仍需实测。
