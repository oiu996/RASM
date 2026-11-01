"""结构化指标记录工具。

采用 JSON Lines (jsonl) 格式：文件里每行是一个独立完整的 JSON 对象。

为什么选 jsonl 而不是 csv / 单个 json：
  * 追加写入即可，训练中途被 kill 也不会破坏已产生的记录
    （csv 需要回写表头，单个 json 需要整体重写，中断即丢全部）
  * 每行自包含，字段可以随实验演进增删，不会破坏历史文件
  * 取值方便：pandas 里 pd.read_json(path, lines=True) 直接变 DataFrame
"""
import json
import os
import datetime

# 只导出这三个工具函数，避免 `from .metrics_utils import *` 把 os/json/datetime
# 也一并灌进 utils 命名空间造成污染
__all__ = ['now_iso', 'append_jsonl', 'read_jsonl', 'write_json']


def now_iso():
    """当前本地时间的 ISO8601 字符串（含微秒），用于给每条记录打时间戳。"""
    return datetime.datetime.now().isoformat()


def append_jsonl(path, record):
    """向 jsonl 文件追加一行记录，父目录不存在时自动创建。

    用 'a' 模式逐行追加，保证训练/测试被中断时已产生的记录不丢失。
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    return path


def read_jsonl(path):
    """读回 jsonl，返回 dict 列表；文件不存在时返回空列表。"""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path, obj):
    """整体写入一个 JSON 文件，适合 config / summary 这类一次性产物。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path
