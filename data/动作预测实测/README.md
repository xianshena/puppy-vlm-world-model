# 动作预测实测数据 · 2026-11-19

先看图例和结果，再用原始回答与轨迹核对分数。[实验报告](../../docs/动作预测实测.md)

| 要查什么 | 在哪里 |
| --- | --- |
| 同一批图的单路／双路表现 | [吞吐与逐图结果](verified-throughput-progress.json) · [吞吐摘要](verified-throughput.json) |
| 限制动作输出前后，模型到底答了什么 | [72 个成对问题](interface.jsonl) |
| 各划分用过哪些布局、看过哪些画面 | [冻结清单](plan.json) · [划分审计](split-audit.json) · [训练](train.jsonl.gz) · [验证](validation.jsonl.gz) · [封存测试](test.jsonl.gz) |
| VLM 实际读到了什么、预测模型怎样训练 | [逐图回答](perception.jsonl.gz) · [视觉调用](visual-requests.jsonl.gz) · [六组逐轮曲线](weights/) |
| 指定动作或隐藏变化下，模型预测了什么 | [实际的分叉结果](forks.jsonl.gz) · [六组模型的逐步预测](evaluation/) |
| 每一步是怎么选的，最后球在哪里 | [1,920 次闭环的动作与结果](evaluation/closed-all.jsonl.gz) · [执行前记录的预测](closed-forecast-ledger.jsonl.gz) |
| 最终数值与数据出处 | [完整指标](metrics.json) · [实验身份](实验身份.json) · [来源与 SHA‑256](来源.json) |

从仓库根目录运行下面命令。用 Python 3.10 或更新版本的标准库即可复算，约几秒钟完成。

```console
python -B "data/动作预测实测/verify.py"
```

公开包保留正式预测、训练曲线、精选合成图和复算代码。逐帧图像、完整模型检查点及设备运行档案保存在原始实验归档。JSONL 文件的帧名由原图 SHA‑256 标识，图例的对应关系见来源表。闭环 JSONL 由原归档的 384 个小文件按顺序拼成，逐条内容可以追溯。预测值中的地面与持球字段对应同一画面的两个区域。
