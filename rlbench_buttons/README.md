# RLBench 三色按键数据与 MemoryVLA 微调

这套命令从 RLBench 的原版三按钮场景生成自定义示范，导出 MemoryVLA 能读取的 RLDS，再从公开的 `memvla-libero-100` 权重继续微调。示范由路径规划自动生成。先验收一条轨迹，再生成小批数据试跑训练。本文以 x86-64 Ubuntu、已安装的 Conda 和可用的 NVIDIA 驱动为例。采集和导出可以在 CPU 机器上做，微调在 GPU 机器上做。

## 1. 在服务器同步代码

以下命令在 Linux 服务器的 Bash 中执行。`LAB` 默认指向用户主目录，也可以设为服务器上有足够空间的可写目录。Git 仓库位于 `server-tools/`，轨迹、RLDS、权重和日志都在其外部。

```bash
export LAB="$HOME"
export WORK="$LAB/rlbench-memoryvla"
mkdir -p "$WORK/logs"
git clone --branch bench-buttons-workflow --single-branch \
  https://github.com/1saachen/memoryvla-server-tools.git "$WORK/server-tools"
export PLAN="$WORK/server-tools/rlbench_buttons"
git -C "$WORK/server-tools" rev-parse HEAD
```

同一 checkout 后续更新用 `git -C "$WORK/server-tools" pull --ff-only`。重连后恢复 `LAB`、`WORK`、`PLAN` 和当前阶段所需的环境变量。长任务在分配到的计算节点上用 `tmux` 或集群作业运行。

## 2. 安装采集环境并验收第一条轨迹

系统软件只需装一次。如果服务器的系统包由管理员管理，请让管理员提供这些库、Xvfb 和 `xauth`。本例用 Xvfb 与 Mesa 软件渲染。如果当前 Bash 找不到 `conda`，先加载已有 Conda 安装目录下的 `etc/profile.d/conda.sh`。

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git xz-utils xvfb xauth \
  libgl1 libglu1-mesa libgl1-mesa-dri libxrender1 libxcb-randr0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
  libxcb-xinerama0 libxkbcommon-x11-0

conda create -y -n buttons-sim --override-channels -c https://conda.anaconda.org/conda-forge python=3.10 pip
conda activate buttons-sim
mkdir -p "$WORK/coppeliasim"
curl -fL https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
  -o "$WORK/CoppeliaSim.tar.xz"
tar -xJf "$WORK/CoppeliaSim.tar.xz" -C "$WORK/coppeliasim" --strip-components=1
```

激活 `buttons-sim` 后设置以下变量。重新打开终端采集时也执行这段。

```bash
export COPPELIASIM_ROOT="$WORK/coppeliasim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1
```

首次搭建时再装 RLBench 和 PyRep。

```bash
git clone https://github.com/stepjam/RLBench.git "$WORK/RLBench"
git -C "$WORK/RLBench" checkout 02720bba4c73fe02eb75df946b8791b806028a9d
python -m pip install 'numpy==1.26.4' 'scipy==1.12.0' 'gymnasium==1.0.0a2' pillow pyquaternion natsort
python -m pip install 'git+https://github.com/stepjam/PyRep.git@8f420be8064b1970aae18a9cfbc978dfb15747ef'
python -m pip install --no-deps -e "$WORK/RLBench"
xvfb-run -a python -c 'import rlbench; from pyrep import PyRep; print("buttons-sim ready")'
```

采集脚本加载原版 `push_buttons.ttm`，通过任务子类设置固定颜色、六种指令顺序、回位点与严格的按键事件记录。脚本没有更改按钮物理属性，所以需要先验收原资产的实际回弹。环境关闭终端后，如果重新连接服务器，需要恢复上面的 `WORK`、`PLAN` 和 CoppeliaSim 环境变量。[RLBench 安装说明](https://github.com/stepjam/RLBench#install) · [PyRep 无界面渲染](https://github.com/stepjam/PyRep#running-headless)

```bash
conda activate buttons-sim
set -o pipefail
xvfb-run -a -s '-screen 0 1024x768x24' \
  python "$PLAN/collect.py" --output "$WORK/data/pilot" \
  --split train --orders 0 --per-order 1 --save-check-frames \
  2>&1 | tee "$WORK/logs/pilot.log"
```

顺序 0 对应蓝→绿→粉。脚本先用原生规划器生成参考轨迹，再从相同布局出发，使用末端增量控制器跟踪参考位姿。每条训练样本把动作与**发动作前**的前视 RGB 对齐。动作向量由世界坐标系的平移增量（米）、相对旋转的 XYZ 欧拉角（弧度）和夹爪开合构成。保存的是实际交给控制器的命令，供后续策略在同一控制器上执行。

成功应输出 `pilot/train/order0_episode00000.npz` 和并排的四帧 `*_home.png`。四帧对应开始前、按完第一键、第二键和第三键的回位时刻。先确认三个真实按键事件顺序正确，按钮都松开后回到近似相同的外观，各回位画面没有明显的姿态差异。脚本会打印回位处的关节偏差，不能把末端位置接近当成关节姿态已经相同。如果原版关节没有稳定回弹，先在 CoppeliaSim 中调整场景的按钮关节设置，再重跑首条示范。若日志里的错误是路径不可达或回放失配，保留错误信息和当前变体进行排查。

通过后先生成 18 条，覆盖六种顺序，每种 3 条，用于验证整个训练入口。

```bash
xvfb-run -a -s '-screen 0 1024x768x24' \
  python "$PLAN/collect.py" --output "$WORK/data/smoke" \
  --split train --per-order 3 --resume
```

## 3. 准备 MemoryVLA 环境并导出短跑数据

导出器读取 `train/*.npz`，输出 `custom_finetuning/1.0.0`。验证和测试留在原始 NPZ 中，因为上游训练代码用 TFDS 的 `all` 来计算动作归一化统计。每个 RLDS 回合包含 RGB、指令、动作和边界标记，也提供加载器要求的末端位姿与夹爪状态字段。按键事件和布局留在原始轨迹里，供评测使用。[MemoryVLA 自定义数据格式](https://github.com/shihao1895/MemoryVLA#training) · [训练数据加载源码](https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/vla/datasets/rlds/dataset.py)

此工作流用独立的上游 MemoryVLA checkout 锁定代码版本。

```bash
export HF_HOME="${HF_HOME:-$LAB/hf_cache}"
export MEMVLA_REPO="$WORK/MemoryVLA"
git clone --branch openvla-codebase https://github.com/shihao1895/MemoryVLA.git "$MEMVLA_REPO"
git -C "$MEMVLA_REPO" checkout d732ea9072bc063399ccc817aed74ab172eb50be
```

如果已有 MemoryVLA 环境，可激活该环境再跑预检。下面假设该环境名为 `memvla`。在已分配的 GPU 节点上，`torch` 应能识别你的设备。

```bash
conda activate memvla
nvidia-smi
df -h "$WORK"
python -c 'import sys, torch, flash_attn, tensorflow, tensorflow_datasets, dlimp; print(sys.version, torch.__version__, torch.cuda.device_count(), tensorflow.__version__, tensorflow_datasets.__version__)'
(cd "$MEMVLA_REPO" && python -c 'import train; print(train.__file__)')
```

如果服务器没有可复用环境，在独立环境安装下面的依赖，然后重跑上面的 import 预检。上游包的 `requires-python` 需允许 Conda 安装的 Python 3.10 补丁版本。这里仅修改新建的上游 checkout 的包元数据。 [锁定版本的依赖定义](https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/pyproject.toml#L3-L12)

```bash
conda create -y -n buttons-train --override-channels -c https://conda.anaconda.org/conda-forge python=3.10 pip
conda activate buttons-train
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
flash=flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
curl -fL "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/$flash" \
  -o "$WORK/$flash"
python -m pip install "$WORK/$flash"
sed -i 's/^requires-python = "==3\.10"$/requires-python = "==3.10.*"/' "$MEMVLA_REPO/pyproject.toml"
python -m pip install -e "$MEMVLA_REPO" 'tensorflow-metadata==1.15.0'
```

训练环境就绪后，导出 18 条短跑轨迹并下载初始化权重。已有 Hugging Face 登录缓存可直接复用，跳过 `huggingface-cli login`。下载 Llama 2 的 `config.json` 用于验证访问权限，初始化训练还会读取或下载 Llama 2 基座权重。TensorFlow 导出用 CPU。若已有同名 RLDS 版本目录，复用验收过的导出结果，或换一个新的输出目录重新导出。 [Llama 2 模型页](https://huggingface.co/meta-llama/Llama-2-7b-hf) · [初始化权重文件](https://huggingface.co/shihao1895/memvla-libero-100/tree/main)

```bash
CUDA_VISIBLE_DEVICES="" python "$PLAN/export_rlds.py" \
  --source "$WORK/data/smoke" --data-root "$WORK/rlds/smoke"

huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf config.json
mkdir -p "$WORK/pretrained"
huggingface-cli download shihao1895/memvla-libero-100 \
  config.json dataset_statistics.json checkpoints/memvla-libero-100.pt \
  --local-dir "$WORK/pretrained/memvla-libero-100"
export INITIAL_PT="$WORK/pretrained/memvla-libero-100/checkpoints/memvla-libero-100.pt"
```

如果采集和训练分开用两台机器，可在采集侧用 Python 3.10、`numpy==1.26.4`、`tensorflow==2.15.0`、`tensorflow-datasets==4.9.3`、`tensorflow-metadata==1.15.0` 完成导出，然后把 `$WORK/rlds/smoke` 或 `$WORK/rlds/full` 复制到 GPU 服务器。

## 4. 用小样本跑两步训练

第一版冻结视觉与语言骨干，训练投影层、记忆模块和动作头。每卡每批是一条回合中采样的 **16 帧**，四卡全局 batch **64 帧**，记忆容量 **16 条**，监督动作块包含当前动作和后续 15 步。学习率 `2e-5`，图像增强关闭。上游在 FSDP 分片前会在每个 rank 的 GPU 上加载完整初始化权重。先用四张大显存卡跑 **2 步**，以实际分配的设备和峰值显存为准。磁盘要同时容纳初始权重、Hugging Face 缓存、数据和多份数十 GB 的 checkpoint。 [初始化权重加载源码](https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/vla/memory_vla.py#L571-L581) · [训练分组与记忆](https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/vla/memory_vla.py#L231-L269)

如果集群已设置 `CUDA_VISIBLE_DEVICES`，沿用分配结果。自行选卡时，先用 `nvidia-smi` 找到分配给你的四张卡，再设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `CUDA_VISIBLE_DEVICES`。例如分配到物理卡 3、4、5、6 时，执行 `export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5,6`。

```bash
set -o pipefail
RUN_ID="buttons-smoke-$(date +%Y%m%d-%H%M%S)"
GPUS=4 SAVE_INTERVAL=2 bash "$PLAN/train.sh" "$MEMVLA_REPO" "$WORK/rlds/smoke" \
  "$INITIAL_PT" "$WORK/runs" "$RUN_ID" 2 2>&1 | tee "$WORK/logs/$RUN_ID.log" && \
  ls -lh "$WORK/runs/$RUN_ID" "$WORK/runs/$RUN_ID/checkpoints"
```

等 `torchrun` 正常退出，确认第 2 步的 loss 为有限数，`config.json`、`dataset_statistics.json` 和 `checkpoints/step-000002-*.pt` 已生成，再继续。

## 5. 生成正式数据并微调

短跑验收后采集正式数据。六种顺序各 50 条训练、10 条验证、10 条测试，总数 **300 / 60 / 60**。中断后按原参数运行 `--resume`。更改布局种子、采样步长或图像尺寸时使用新的数据目录。

```bash
conda activate buttons-sim
# 恢复第 2 节中 CoppeliaSim 和软件渲染变量。
for split in train val test; do
  n=10
  if [[ "$split" == train ]]; then n=50; fi
  xvfb-run -a -s '-screen 0 1024x768x24' \
    python "$PLAN/collect.py" --output "$WORK/data/full" \
    --split "$split" --per-order "$n" --resume || break
done
for split in train val test; do
  echo "$split $(find "$WORK/data/full/$split" -maxdepth 1 -name 'order*_episode*.npz' | wc -l)"
done
```

确认三组计数后，切回第 4 节跑短跑的训练终端，或恢复同一环境、路径和 GPU 分配。先导出 RLDS，验收输出为 300 条训练轨迹。

```bash
CUDA_VISIBLE_DEVICES="" python "$PLAN/export_rlds.py" \
  --source "$WORK/data/full" --data-root "$WORK/rlds/full"
```

正式组从同一公开初始权重开始，计划 **5,000 步**，每 **1,000 步**保存一次。

```bash
set -o pipefail
RUN_ID="buttons-mem16-$(date +%Y%m%d-%H%M%S)"
GPUS=4 SAVE_INTERVAL=1000 bash "$PLAN/train.sh" "$MEMVLA_REPO" "$WORK/rlds/full" \
  "$INITIAL_PT" "$WORK/runs" "$RUN_ID" 5000 2>&1 | tee "$WORK/logs/$RUN_ID.log" && \
  ls -lh "$WORK/runs/$RUN_ID/checkpoints"
```

以训练正常退出、第 5,000 步 checkpoint 落盘和同目录下的 `dataset_statistics.json` 为验收。训练脚本以新运行目录从传入的权重热启动，`--resume` 是采集脚本的选项。推理用 `custom_finetuning` 的统计反归一化动作。

如果后续要比较 16 与 24 条记忆容量，并希望训练时用到第 17–24 条历史，两组均可设 `GROUP_SIZE=32`，仅改变 `MEM_LENGTH`。每卡 batch 同步增至 32 帧，需要单独做显存短跑。

闭环策略评测在留出的布局上进行，分别记录完整顺序、重复按压和首次错按。上面的命令覆盖采集、导出和训练，首条仿真示范以及 GPU 短跑需要在服务器上确认。
