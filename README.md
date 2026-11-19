# MemoryVLA server tools

用于现有 RoboMME / MemoryVLA Linux 训练环境的独立测速工具。工具只使用 Python 标准库；训练依赖由现有 `memvla` 环境提供。

## 对照方式

- 默认依次运行 GPU 0、1，每卡 batch 32；GPU 0～3，每卡 batch 16。加 `--gpus 1/2/4/8` 可选择其中一种卡数。
- 默认单卡、两卡、四卡总 batch 为 64，八卡每卡 batch 16、总 batch 128。`group_size=16`，shuffle buffer 为 1024。默认单卡梯度累积 4 次，其余配置无需梯度累积。
- 用 `--batch-per-gpu 16`、`--batch-per-gpu 32` 等调整每卡 batch；必须为 16 的正整数倍，保证完整的记忆组。指定后总 batch 自动等于卡数乘以每卡 batch，无需梯度累积，不再固定为 64。
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

下面这条测八卡 × 每卡 batch 16，总 batch 128，适用于已停止上一轮测速的情况。复制整行执行一次：

```bash
mkdir -p /data/embodied-ai/yaolin/MemoryVLA/log/robomme && { perf_log="/data/embodied-ai/yaolin/MemoryVLA/log/robomme/perf_driver_$(date +%Y%m%d_%H%M%S).log"; nohup /data/embodied-ai/yaolin/envs/memvla/bin/python -u /data/embodied-ai/yaolin/memoryvla-server-tools/perf_compare.py --root /data/embodied-ai/yaolin/MemoryVLA --gpus 8 --batch-per-gpu 16 > "$perf_log" 2>&1 < /dev/null & perf_pid=$!; printf '%s\n' "$perf_pid" > "${perf_log%.log}.pid"; printf '%s\n' "$perf_log" > /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_perf_log.txt; echo "测速 PID：$perf_pid；日志：$perf_log"; }
```

后台进程显式使用所选 Python 环境，其中的 `torchrun` 也通过 PATH 优先使用。缓存默认放在训练仓库上一级的 `hf_cache`，默认端点是 `https://huggingface.co`；已有 `HF_HOME`、`HF_ENDPOINT` 环境变量优先生效。训练启动脚本中的 export 也会作用于训练进程。

下一次试八卡 × 每卡 32 时，将 `--batch-per-gpu 16` 改为 `--batch-per-gpu 32`；总 batch 为 256。只测单卡并保持旧的总 batch 64 对照时，使用 `--gpus 1` 并省略 `--batch-per-gpu`。需要先结束上一轮测速，可执行下面的命令；它匹配此工具的绝对路径，向测速主进程和 workers 发送 SIGTERM。再次执行 `pgrep` 无输出后再启动下一轮，以免与旧测速重叠。

```bash
pkill -TERM -f '[/]data/embodied-ai/yaolin/memoryvla-server-tools/perf_compare[.]py'
pgrep -af '[/]data/embodied-ai/yaolin/memoryvla-server-tools/perf_compare[.]py'
```

## 日志和结果

总体进度：

```bash
tail -n 50 -f "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_perf_log.txt)"
```

各组详细日志和 NCCL 信息位于结果目录的 `perf1.log`、`perf2.log`、`perf4.log`、`perf8.log`。每组包括初始化在内最多运行一小时。非零退出的配置会标记为失败，再尝试下一组；超时会清理本组进程并结束测试。

未指定每卡 batch 的单卡对照以 `Global Step` 和 JSONL 为准：100 个参数更新步对应 400 个微批次，原训练进度条按微批次计数，可能超过 `100/100`；其 `gradient_accumulation_steps` 应为 4。

```bash
cat "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_perf_dir.txt)/summary.json"
```

默认两组均成功才输出 `COMPARISON_COMPLETE`；否则输出 `COMPARISON_INCOMPLETE` 或具体异常。指定 `--gpus` 时，单组成功输出 `BENCHMARK_COMPLETE`，失败输出 `BENCHMARK_INCOMPLETE` 或具体异常。

不同总 batch 的配置按 `samples_per_second`（总 batch / 平均每步秒数）比较，越高越好。自动对照也按吞吐量选择结果，不再按每步耗时选择。每份结果记录 `global_batch_size`、`batch_per_gpu` 和 `gradient_accumulation_steps`。例如 batch 256、8 秒/步的吞吐量为 32 样本/秒，高于 batch 64、7 秒/步的约 9.1 样本/秒。

这些都是 100 步的性能测试，学习率等训练配置沿用原脚本。正式训练需要根据总 batch 重新评估学习率、训练步数和验证表现；batch 翻倍后照跑同样步数，处理的样本数也会翻倍。相同样本量不保证相同收敛或任务成功率。显存不足、网络失败、数据未就绪等情况需要根据详细日志处理。

## 本地检查

```bash
python -m unittest discover -s tests -p 'test_*.py'
python perf_compare.py --help
```

测试涵盖启动参数生成、已有服务器配置、异常输入、测速进程禁用保存、热身排除、缺失/非有限指标以及停止进程的范围判断。实际吞吐量和多卡运行必须在训练服务器验证。
