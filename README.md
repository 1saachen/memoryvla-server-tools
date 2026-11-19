# MemoryVLA server tools

用于现有 RoboMME / MemoryVLA Linux 训练环境的独立测速工具。工具只使用 Python 标准库；训练依赖由现有 `memvla` 环境提供。

## 对照方式

- 默认依次运行 GPU 0、1，每卡 batch 32；GPU 0～3，每卡 batch 16。加 `--gpus 1/2/4/8` 可选择其中一种卡数。
- 默认单卡、两卡、四卡总 batch 为 64，八卡每卡 batch 16、总 batch 128。`group_size=16`，shuffle buffer 为 1024。默认单卡梯度累积 4 次，其余配置无需梯度累积。
- 用 `--batch-per-gpu 16`、`--batch-per-gpu 32` 等调整每卡 batch；必须为 16 的正整数倍，保证完整的记忆组。指定后总 batch 自动等于卡数乘以每卡 batch，无需梯度累积，不再固定为 64。
- `--gpus 2 --devices 3,4` 只使用 `nvidia-smi` 编号 3、4 的物理卡。工具在启动前把编号解析为 UUID，避免 CUDA 枚举顺序影响选卡；卡号必须互不重复且数量与 `--gpus` 一致。省略 `--devices` 时仍使用从 0 开始的连续 CUDA 编号。工具不预留卡或自动判断是否独占，启动前需要自行确认占用。
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

先确认八卡均可使用。如果只有物理 GPU 3、4 空闲，把下面命令的 `--gpus 8 --batch-per-gpu 16` 替换为 `--gpus 2 --devices 3,4 --batch-per-gpu 32`，总 batch 为 64。详细日志为 `perf2.log`。GPU 3、4 在本次服务器拓扑中跨 NUMA，该结果不能代表同一 NUMA 内卡对的带宽。

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

## 正式训练：四卡，总 batch 128，2000 步

`train_run.py` 是当前商定配置的正式训练启动器：物理 GPU 3、4、5、6，每卡 batch 32，总 batch 128，2000 个参数更新步，每 500 步保存。学习率、数据、初始权重、增强等沿用训练仓库脚本；group size 16、shuffle buffer 1024 与上一轮测速一致。该次从初始权重重新训练，resume 关闭。

它生成新的独立目录和启动脚本，直接运行训练仓库的 `train.py`，不经过禁用保存的测速 worker。训练仓库源码和原启动脚本不改动。每次输出位于 `log/robomme/formal_<时间>/runs/memvla_robomme_4gpu_bs32[--image_aug]/checkpoints/`。

`--stop-benchmark` 仅向本工具目录下、目标训练仓库对应的测速进程发送 SIGTERM，并等待最多 45 秒。不会停止视频实验或已有正式训练；发现其他 `train.py` 训练时退出。测速未退出或锁仍被占用时不启动正式训练。运行期间持有锁，避免同工具的测速与训练重叠。

更新并后台启动（复制整行执行一次）：

```bash
git -C /data/embodied-ai/yaolin/memoryvla-server-tools pull --ff-only && mkdir -p /data/embodied-ai/yaolin/MemoryVLA/log/robomme && { train_log="/data/embodied-ai/yaolin/MemoryVLA/log/robomme/train_$(date +%Y%m%d_%H%M%S).log"; nohup /data/embodied-ai/yaolin/envs/memvla/bin/python -u /data/embodied-ai/yaolin/memoryvla-server-tools/train_run.py --root /data/embodied-ai/yaolin/MemoryVLA --stop-benchmark > "$train_log" 2>&1 < /dev/null & train_pid=$!; printf '%s\n' "$train_pid" > /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_train_pid.txt; printf '%s\n' "$train_log" > /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_train_log.txt; echo "正式训练 PID：$train_pid；日志：$train_log"; }
```

跟踪日志：

```bash
tail -n 50 -f "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_train_log.txt)"
```

看到 `FORMAL_TRAINING_START` 表示测速已退出并开始正式训练启动流程，不代表初始化已经完成。约 10 秒/步时，纯训练耗时约 5 小时 33 分，另加初始化与 checkpoint 保存。500、1000、1500、2000 步保存到本次独立目录。启动器等训练退出后检查这四个非空权重文件，成功时写入 `completed.json` 并输出 `FORMAL_TRAINING_COMPLETE`；仅检查文件存在及大小，不替代加载验证或任务评测。权重文件保存沿用训练仓库实现，并不额外添加优化器保存或精确断点续训能力。

正式运行同时在本次目录写入 `gpu_monitor.log`，约每两秒采样一次 GPU 3～6 的显存、利用率和可见进程，并记录目标仓库 `train.py` 的 PID。训练前后也各采样一次，退出时停止采样。驱动查询失败会记录诊断错误，不调整训练配置。`nvidia-smi` 可能因容器/PID 可见性而省略部分进程，因此进程占用之和不一定等于整卡占用。torchrun 通过 `--tee 3 --log-dir` 同时保留各 rank 的 stdout/stderr 到 `rank_logs/`，主日志照常可见。

如果再次 OOM，保留本次目录，查看失败前的采样和各 rank 的异常；不要仅用退出后的空闲显存判断失败时的占用：

```bash
tail -n 220 "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_train_dir.txt)/gpu_monitor.log"
grep -RIn -B 4 -A 8 'OutOfMemoryError' "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_train_dir.txt)/rank_logs"
```

查看本次已生成的 checkpoint：

```bash
find "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_train_dir.txt)/runs" -type f -name '*.pt' -printf '%p %s bytes\n'
```

需要中止正式训练时，对启动器 PID 发送 SIGTERM；启动器会清理自己启动的训练进程组：

```bash
kill -TERM "$(cat /data/embodied-ai/yaolin/MemoryVLA/log/robomme/latest_train_pid.txt)"
```

## 本地检查

```bash
python -m unittest discover -s tests -p 'test_*.py'
python perf_compare.py --help
python train_run.py --help
```

测试涵盖启动参数生成、已有服务器配置、异常输入、测速进程禁用保存、热身排除、缺失/非有限指标以及停止进程的范围判断。实际吞吐量和多卡运行必须在训练服务器验证。
