# 下一步会看见蓝球吗？

小狗从大厅出发，去桌子 A、沙发 B 或柜子 C 检查。给定走过的 12 帧和接下来准备执行的动作，先预测下一眼是否能看见蓝球，再揭开结果。这个数据集适合练习部分可观测环境中的预测。

| 文件 | 怎么用 |
| --- | --- |
| [inputs.jsonl.gz](inputs.jsonl.gz) | 每行一个实际执行过的分支，含历史地点、12 张图像的索引及拟检查地点。 |
| [targets.jsonl.gz](targets.jsonl.gz) | 按同一个 `id` 取出下一张真实渲染图、模拟器的蓝球可见标签和冻结 VLM 的读图回答。先保存预测再读标签。 |
| [正式图像.tar.gz](../图像/正式图像.tar.gz) | 原实验与练习共用的一份图像包。练习引用其中 1,404 张。解压到本目录后，相对路径与清单一致。文件名取 JPEG 字节的 SHA‑256。 |
| [provenance.jsonl.gz](provenance.jsonl.gz) | 每条分支来自哪个布局、哪个情境、哪些策略。用于分组、追溯和防止数据泄漏。 |
| [summary.json](summary.json) | 样本数、分支覆盖和三个解压清单的 SHA‑256。 |

给模型输入像素、观察顺序和允许的动作。配对 ID、文件名、策略名称与情境名称留在加载器中。JPEG 本身底部保留旧模拟器的地点标识。正式新实验会移除该标识，以检验模型能否从场景外观辨认位置。

844 个正式回合一共记录了 1,668 个不同的“回合＋合法首动作”，同一回合中多个策略作出同一首选只保留一条。235 局只实际检查了一种首步，394 局有两种，215 局有三种。在 1,210 种不同的“完整可见历史＋候选动作”里，有 333 种既出现了有球，也出现了无球。隐藏干预、策略选择和重复样本都会影响这里的频率。这批已读数据用于教学开发，正式测评采用下一轮新布局。

**解压清单即可用 Python 或其他工具读取。** 在此目录运行 `python 复核练习.py`，或从项目根目录把脚本路径传给 Python。脚本使用标准库，对照图像哈希、输入标签配对和歧义样本计数逐项核验。下面示例在读完第一条输入后按索引读第一帧。

```python
import gzip, json, tarfile
with gzip.open("inputs.jsonl.gz", "rt", encoding="utf-8") as rows:
    sample = json.loads(next(rows))
with tarfile.open("../图像/正式图像.tar.gz", "r:gz") as images:
    jpeg = images.extractfile(sample["history"][0]["image"]).read()
print(sample["inspect"], len(jpeg))
```
