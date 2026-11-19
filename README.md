# MemoryVLA server tools

用于现有 RoboMME / MemoryVLA Linux 训练环境的独立测速工具。工具只使用 Python 标准库；训练依赖由现有 `memvla` 环境提供。

## 对照方式

- 默认依次运行 GPU 0、1，每卡 batch 32；GPU 0～3，每卡 batch 16。加 `--gpus 4` 只测四卡，加 `--gpus 2` 只测两卡，加 `--gpus 1` 只测 GPU 0。
- 总 batch 均为 64，`group_size=16`，shuffle buffer 为 1024。单卡每次前后向 batch 16，梯度累积 4 次；两卡、四卡无需梯度累积。
- 单卡测速子进程使用 FSDP `NO_SHARD`，完整模型、梯度和优化器驻留一张卡；实际显存是否足够需要服务器验证，显存不足会记录为失败。
- 从训练仓库的 `script/train/robomme/train.sh` 生成临时启动脚本，保留学习率、数据和初始权重等配置。
- 各运行 100 步，排除前 20 步，统计后 80 步的平均/中位耗时、吞吐量，并检查 loss 为有限数值。
- 仅在测速子进程中禁用 checkpoint 写入，避免终止步骤的权重保存干扰测速。正式训练代码、启动脚本、数据和已有产物不改动。
- 不会在测试结束后自动启动正式训练。100 步仅用于性能对照，不代表模型效果评测。

## 服务器首次安装

```bash
git clone https://github.com/1saachen/memoryvla-server-tools.git /data/embodied-ai/yaolin/memoryvla-server-tools
```

## 更新代码

```bash
git -C /data/embodied-ai/yaolin/memoryvla-server-tools pull --ff-only
```

## 后台启动测速

当前示例使用已有环境 `/data/embodied-ai/yaolin/envs/memvla` 和训练仓库 `/data/embodied-ai/yaolin/MemoryVLA`。可按实际目录替换。

`--stop-existing` 会向工作目录恰好为目标训练仓库的 Python/torchrun `train.py` 进程发送 SIGTERM，放弃其未保存进度。省略此参数时，发现已有训练就退出。不要对共享给其他任务的训练仓库使用该参数。旧进程 45 秒内未退出则停止测试，不自动强制杀进程。

下面这条只测四卡，适用于已停止两卡测试的情况。复制整行执行一次：

```bash
mkdir -p /data/embodied-ai/yaolin/MemoryVLA/log/robomme && { perf_log="/data/embodied-ai/yaolin/MemoryVLA/log/robomme/perf_driver_$(date +%Y%m%d_%H%M%S).log"; nohup /data/embodied-ai/yaolin/envs/memvla/bin/python -u /data/embodied-ai/yaolin/memoryvla-server-tools/perf_compare.py --root /data/embodied-ai/yaolin/MemoryVLA --gpus 4 > "$perf_log" 2>&1 < /dev/null & perf_pid=$!; printf '%s\n' "$perf_pid" > "${perf_log%.log}.pid"; printf '%s\n' "$perf_log" > /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_perf_log.txt; echo "测速 PID：$perf_pid；日志：$perf_log"; }
```

后台进程显式使用所选 Python 环境，其中的 `torchrun` 也通过 PATH 优先使用。缓存默认放在训练仓库上一级的 `hf_cache`，默认端点是 `https://huggingface.co`；已有 `HF_HOME`、`HF_ENDPOINT` 环境变量优先生效。训练启动脚本中的 export 也会作用于训练进程。

只测单卡时，把上述命令中的 `--gpus 4` 改成 `--gpus 1`。需要先结束上一轮测速，可执行下面的命令；它匹配此工具的绝对路径，向测速主进程和 workers 发送 SIGTERM。再次执行 `pgrep` 无输出后再启动下一轮，以免与旧测速重叠。

```bash
pkill -TERM -f '[/]data/embodied-ai/yaolin/memoryvla-server-tools/perf_compare[.]py'
pgrep -af '[/]data/embodied-ai/yaolin/memoryvla-server-tools/perf_compare[.]py'
```

## 日志和结果

总体进度：

```bash
tail -n 50 -f "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_perf_log.txt)"
```

各组详细日志和 NCCL 信息位于结果目录的 `perf1.log`、`perf2.log`、`perf4.log`。每组包括初始化在内最多运行一小时。非零退出的配置会标记为失败，再尝试下一组；超时会清理本组进程并结束测试。

单卡以 `Global Step` 和 JSONL 为准：100 个参数更新步对应 400 个微批次，原训练进度条按微批次计数，可能超过 `100/100`。结果中的 `mean_step_seconds` 仍为处理总 batch 64 的一次参数更新耗时，可与多卡比较；单卡 `gradient_accumulation_steps` 应为 4。

```bash
cat "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_perf_dir.txt)/summary.json"
```

默认两组均成功才输出 `COMPARISON_COMPLETE`；否则输出 `COMPARISON_INCOMPLETE` 或具体异常。指定 `--gpus` 时，单组成功输出 `BENCHMARK_COMPLETE`，失败输出 `BENCHMARK_INCOMPLETE` 或具体异常。结果中的 `mean_step_seconds` 越低越好，`samples_per_second` 越高越好。显存不足、网络失败、数据未就绪等情况需要根据详细日志处理。

## 本地检查

```bash
python -m unittest discover -s tests -p 'test_*.py'
python perf_compare.py --help
```

测试涵盖启动参数生成、已有服务器配置、异常输入、测速进程禁用保存、热身排除、缺失/非有限指标以及停止进程的范围判断。实际吞吐量和多卡运行必须在训练服务器验证。
