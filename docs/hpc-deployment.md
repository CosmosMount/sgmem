# 容器与 HPC 部署

项目环境与主机 Python 分开管理：镜像使用 Python 3.11，按 [requirements-docker.txt](../requirements-docker.txt) 安装固定的 Linux x86_64 运行依赖，再安装本项目。依赖清单由 `pyproject.toml` 的 `dense` 和 `serving` 依赖解析，包含 PyTorch 2.12、Transformers 和 Sentence Transformers；模型、数据、模型缓存和结果通过挂载提供，不放入镜像。

**本机已安装 Docker Engine 29.9.3 和 NVIDIA Container Toolkit 1.20.1；用户提供的验证输出显示 `sgmem:local` 已构建，容器内 PyTorch 2.12.0+cu130 可访问 NVIDIA GeForce RTX 3090。** 随后已在容器内验证 Qwen 服务、BM25、E5 Dense、判分和失败分析命令：两种方法运行同一道 Finance 开发集题目，验证输出和测试容器均已清理。这是连通性测试，不是正式准确率实验。该清单固定 Python 依赖，但 `python:3.11-slim` 基础镜像和系统包目前没有固定 digest/版本，不能视为完整的字节级环境复现。清单面向 Linux x86_64/Python 3.11；其他架构需要重新解析并测试。PyTorch CUDA 包与主机 NVIDIA 驱动是否兼容，需要在目标集群确认。

## Ubuntu 本机安装 Docker 和 NVIDIA 支持

当前工作站为 Ubuntu 22.04 amd64，RTX 3090 已有 NVIDIA 驱动。Docker 和 NVIDIA Container Toolkit 已安装；在一台未安装 Docker 的新机器上，可在本机终端、仓库根目录执行以下脚本。系统安装需要交互式管理员密码，不要将密码发送到聊天中：

```bash
bash scripts/install-docker-ubuntu.sh
```

脚本使用 Docker 和 NVIDIA 官方 apt 仓库，安装 Docker Engine、Compose/Buildx 和 NVIDIA Container Toolkit，使用 `nvidia-ctk` 配置 Docker 并启动/重启服务。它不会自动删除已有冲突包或覆盖已存在的仓库配置；遇到这些情况会停下。若安装中途失败，请先根据报错检查已完成的系统修改，不要盲目删除已有配置。脚本不修改 NVIDIA 驱动，也不自动将用户加入 `docker` 组；该组具有相当于 root 的权限，因此本指南使用 `sudo docker`。安装方法可对照 [Docker Ubuntu 官方文档](https://docs.docker.com/engine/install/ubuntu/)和 [NVIDIA 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

在本机终端、仓库根目录执行以下命令，构建项目镜像并验证 Python 和 NVIDIA GPU：

```bash
# 本机使用 127.0.0.1:7890 代理；无代理的机器可省略变量
SGMEM_BUILD_PROXY=http://127.0.0.1:7890 bash scripts/verify-docker.sh
```

脚本使用 `sudo docker`，会下载基础镜像和 Python 依赖，构建最终镜像 `sgmem:local`，并使用 `--rm` 清理测试容器。最后必须看到 `CUDA available: True` 和目标 GPU 名称，才表示镜像可以访问 NVIDIA GPU；失败时保留终端错误信息以便排查。安装、构建和验证需要网络及足够的磁盘空间。

已有镜像和本机 Qwen/E5 文件后，可以继续验证容器内完整调用：

```bash
bash scripts/verify-docker-pipeline.sh
```

该脚本临时启动 Qwen 服务，使用 Finance 开发集的同一道题分别执行 BM25 和 E5 Dense、QA、judge 和失败分析命令；检查可解析判分和检索结果后关闭测试服务并删除临时输出。它不重建镜像，不把单题的输出当作正式实验结果。失败时请保留终端报错；若模型位置或端口不同，可设置 `SGMEM_HF_CACHE`、`SGMEM_QWEN_CONTAINER_MODEL` 和 `SGMEM_VERIFY_PORT`。

### 本机 Docker Hub DNS / 代理问题

本机当前无法直接解析 Docker Hub，但本地 HTTP 代理 `127.0.0.1:7890` 可以访问。Docker daemon 不会自动继承终端代理；可在确保该代理正在运行后，在项目根目录执行：

```bash
bash scripts/configure-docker-proxy.sh
```

脚本先测试代理，然后写入 `/etc/systemd/system/docker.service.d/sgmem-proxy.conf` 并重启 Docker；重启会影响正在运行的 Docker 容器。随后它会使用 Linux host 网络和临时 build proxy 参数执行 `scripts/verify-docker.sh`，让构建中的 apt 和 pip 也能连接网络。代理地址若变化，可运行 `SGMEM_DOCKER_PROXY=http://127.0.0.1:端口 bash scripts/configure-docker-proxy.sh`；脚本不会覆盖不一致的现有配置。

该代理只适用于提供本地监听的机器，不应直接照搬到 HPC。恢复 Docker daemon 的原有网络配置时，可在确认无需此代理后执行 `sudo rm /etc/systemd/system/docker.service.d/sgmem-proxy.conf`，再运行 `sudo systemctl daemon-reload` 和 `sudo systemctl restart docker`。代理值通过 Docker 预定义 build 参数传递，不写入 Dockerfile 或镜像环境。

需要更新依赖时，先修改 `pyproject.toml`，然后使用 uv 重新解析并审查 `requirements-docker.txt`：

```bash
uv pip compile pyproject.toml --extra dense --extra serving \
  --python-version 3.11 --python-platform x86_64-unknown-linux-gnu \
  --no-header --no-annotate --output-file requirements-docker.txt
```

本次修复缺失的 `requests` 时复用了已有镜像的 CUDA 层；更新后的全部 79 项锁定依赖已在最终镜像内核对，`pip check` 和单题端到端验证通过。更新后的完整 Dockerfile 尚未从空缓存重新构建；正式实验前应按需要复核构建环境与模型版本。离线 HPC 可在允许联网的构建机准备镜像和模型；不能假设计算节点可访问外网。

## Docker

首次使用请先按上面的 `scripts/verify-docker.sh` 构建镜像；随后可在仓库根目录运行：

```bash
# 本机使用代理时，建议通过上面的 verify-docker.sh 构建。
sudo docker run --rm sgmem:local --help
sudo docker run --rm --runtime=nvidia --gpus all --entrypoint python sgmem:local \
  -c 'import torch; print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available()'
```

第二条命令验证镜像中的 PyTorch 能否使用 GPU；若只需 CPU 可跳过。容器里的 Python 和依赖来自镜像，不读取宿主机 Python 环境。下面的例子使用 **Linux 主机网络**连接本地模型服务；其他系统应使用其可访问的服务地址。运行 GPU 任务需宿主驱动及 NVIDIA Container Toolkit 支持 `--gpus all`。

在一个终端启动 Transformers 的 OpenAI 兼容服务，模型文件需预先下载到挂载的缓存中。服务与客户端必须使用相同的模型名；以下命令使用本机缓存的 Qwen2.5-0.5B 作为连通性验证示例，缓存位置按实际路径调整。两个终端均需设置同一个 `QWEN_CONTAINER_MODEL`：

```bash
export QWEN_CONTAINER_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
sudo docker run --rm --network host --runtime=nvidia --gpus all \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -e HF_HUB_OFFLINE=1 --entrypoint transformers sgmem:local \
  serve "$QWEN_CONTAINER_MODEL" --host 127.0.0.1 --port 8000 \
  --device cuda --dtype float16
```

在另一个终端运行客户端。挂载包含 Git 子模块的项目到 `/workspace`；Dense 模型从项目挂载路径 `/workspace/models/e5-base-v2` 加载，且 QA 模型名必须与服务启动时一致。输出路径必须事先不存在：

```bash
export QWEN_CONTAINER_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
mkdir -p data/splits results
sudo docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e USER=sgmem -e LOGNAME=sgmem -v "$PWD:/workspace" sgmem:local \
  split --output data/splits/finance-tech.json
sudo docker run --rm --network host --runtime=nvidia --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -e HOME=/tmp -e USER=sgmem -e LOGNAME=sgmem -e HF_HOME=/tmp/huggingface \
  -e HF_HUB_OFFLINE=1 -e NO_PROXY=127.0.0.1,localhost sgmem:local \
  run --method dense --device cuda --embedding-model /workspace/models/e5-base-v2 \
  --answer-model "$QWEN_CONTAINER_MODEL" \
  --domains Finance --categories multi_hop --limit 2 \
  --split data/splits/finance-tech.json --partition dev --output results/dense-smoke
```

BM25 可去掉 GPU、`--device cuda` 和 `--embedding-model`，并改为 `--method bm25`。客户端使用宿主用户的 UID/GID 写入，避免输出归 root；`USER` 和 `LOGNAME` 为未在镜像 passwd 中登记的 UID 提供名称，`HOME` 和缓存位于可写的容器临时目录。`split` 和 `run` 不覆盖已有输出，重跑时需指定新的路径。镜像在挂载了 `/workspace` 时优先运行工作区 `src/`，使代码与运行记录的工作区一致；依赖变化仍需重新构建镜像。上述流程已在本机使用单题验证；目标 HPC 的运行环境仍需单独验证。小模型只能验证连通性，正式实验须选择并固定适合任务的 QA/judge 模型。测试产物遵循 [AGENTS.md](../AGENTS.md) 清理。

## HPC 与 Apptainer

是否允许 Docker、Apptainer、镜像仓库及外网访问，以集群规定为准。可在构建机用 `docker save` 导出镜像并传至集群，再使用 `apptainer build sgmem.sif docker-archive:sgmem.tar` 转换；也可按集群规则从受信任仓库导入。[Apptainer OCI 文档](https://apptainer.org/docs/user/main/docker_and_oci.html)

在已分配的计算节点，可将项目目录绑定到 `/workspace`，运行 `apptainer exec --pwd /workspace --bind "$PWD:/workspace" sgmem.sif python -m sgmem --help`。使用 NVIDIA GPU 时，在集群支持的前提下加入 `--nv`，确认宿主驱动与容器内 PyTorch/CUDA 兼容，遵循调度器分配的可见设备。[Apptainer GPU 文档](https://apptainer.org/docs/user/main/gpu.html)

数据和模型可挂载为只读；模型缓存、结果目录、服务网络和端口需按集群规定配置。作业脚本应根据实际调度器、分区、GPU、CPU 和时限编写；当前未验证 Slurm 作业或完整 HPC 部署。

## 已发现的 Slurm 集群与更大 Qwen

用户提供的 `mgmt-3` 信息显示：集群使用 Slurm，列有 `i64m1tga40u`（A40）和 `i64m1tga800u`（A800）等 GPU 分区，提供 `apptainer-1.4.5` 模块。管理节点的 `nvidia-smi` 无法访问驱动，不代表计算节点没有 GPU；不要在管理节点启动模型。分区是否对当前账号开放、计算节点实际驱动与显存、容器网络和文件权限仍须实测。以下命令是可调整示例，尚未在该集群运行。

优先使用 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)：它是 non-thinking 指令模型，便于沿用现有问答和 judge 格式。其 4B 参数以 FP16/BF16 加载时仅权重约 8 GB，实际显存还包含运行时、上下文 KV cache 和 E5。若预算允许，也可试 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)；其默认 thinking，须明确关闭或验证服务输出与判分解析，并记录输出长度及额外成本。Qwen3.5 是另一架构，当前镜像与本项目还未测试，不能直接视为兼容。模型大小和官方指标不保证在本任务上效果更好。

### 1. 准备源码、镜像和模型

将当前项目的实际代码、`external/GroupMemBench` 子模块、E5 文件和需要使用的 split 文件传到集群共享目录；不要只传未包含本次改动的旧提交。正式实验应记录项目 commit、未提交改动、上游版本与模型 revision。模型文件、容器镜像和原始数据不纳入普通 Git。

如果集群允许 Apptainer 从 Docker archive 构建，可在本机导出当前已验证的镜像，传输 archive 后在集群转换；文件需要单独的足够空间，完成并校验转换后清理本次产生的临时 archive。路径需替换为两台机器实际可用的位置：

```bash
# 本机；选择合适的临时传输路径
sudo docker save -o /path/to/transfer/sgmem.tar sgmem:local
# 传输后，在集群允许构建镜像的节点
module load apptainer-1.4.5
apptainer build /path/to/sgmem.sif docker-archive:/path/to/sgmem.tar
```

在集群网络可访问 Hugging Face 且允许下载的节点，可用镜像中的 `hf` 命令将模型下载到项目目录。下例指定官方模型的一次固定 revision（来源：[模型仓库](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)）；若集群无法联网，在有网的机器下载完整目录后按集群要求传输。不要只复制 Hugging Face cache 的 `snapshots` 子目录而丢失其指向的 `blobs`。

```bash
# 在集群按实际位置设置；模型存储需要足够空间
SGMEM_PROJECT=/path/to/sgmem
SGMEM_IMAGE=/path/to/sgmem.sif
module load apptainer-1.4.5
apptainer exec --pwd /workspace --bind "$SGMEM_PROJECT:/workspace" "$SGMEM_IMAGE" \
  hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir /workspace/models/qwen3-4b-instruct
```

`models/qwen3-4b-instruct` 是实际运行所需的权重目录；下载器产生的中间缓存可在下载完成且确认完整后清理，勿删除模型文件。E5 同样需要在集群准备为 `models/e5-base-v2`。若模型放在其他共享路径，需按站点策略绑定，并保证服务和客户端使用同一个容器内模型路径。

### 2. 申请 GPU 并检查驱动

例如，若当前账号有权使用该 A40 分区，可在管理节点提交一个短检查；队列可能等待，GPU 占用计入课程预算。若集群要求 account、QoS 或其他参数，请按站点规定增加，拒绝访问时不要绕过限制：

```bash
srun -p i64m1tga40u --gres=gpu:a40:1 -n 1 -c 2 --mem=4G -t 00:02:00 nvidia-smi
```

进入计算节点后记录 GPU 实际显存和 **Driver Version**。当前镜像的 PyTorch 使用 CUDA 13.0；NVIDIA 的常规 CUDA 13.x 兼容要求为驱动 **580 或更高**，实际运行仍须检查完整兼容性。若节点驱动过旧，应先与管理员确认可用驱动或重建使用兼容 CUDA/PyTorch 的镜像；不要自行修改集群驱动。[NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

下列交互式分配仅为示例，时间上限和资源均按课程及分区政策调整。退出 shell 会释放该分配：

```bash
srun -p i64m1tga40u --gres=gpu:a40:1 -n 1 -c 4 --mem=24G -t 00:30:00 --pty bash
```

在**已分配的 GPU 计算节点**中加载模块，设置集群上的实际路径，然后检查容器是否看到 GPU。不要手动覆盖 Slurm 提供的 `CUDA_VISIBLE_DEVICES`：

```bash
module load apptainer-1.4.5
SGMEM_PROJECT=/path/to/sgmem
SGMEM_IMAGE=/path/to/sgmem.sif
apptainer exec --nv "$SGMEM_IMAGE" python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

### 3. 在同一计算节点运行 Qwen 和基线

以下在上述 GPU 作业 shell 内执行；变量均应替换为集群路径，`$SGMEM_IMAGE` 和 `$SGMEM_PROJECT` 必须对计算节点可见。容器沿用共享的工作区代码；服务与客户端通过同节点 loopback 通信。正式共享多人的节点上需选未占用的端口，服务不要绑定到所有网络接口。

```bash
SGMEM_MODEL=/workspace/models/qwen3-4b-instruct
SGMEM_EXEC=(apptainer exec --nv --pwd /workspace --bind "$SGMEM_PROJECT:/workspace" \
  --env HF_HUB_OFFLINE=1 --env NO_PROXY=127.0.0.1 --env PYTHONPATH=/workspace/src "$SGMEM_IMAGE")
"${SGMEM_EXEC[@]}" transformers serve "$SGMEM_MODEL" \
  --host 127.0.0.1 --port 18767 --device cuda --dtype float16 &
SGMEM_QWEN_PID=$!
```

等待终端提示服务启动，再检查就绪状态；`curl` 成功只说明服务可访问，仍须真实运行 QA 和 judge：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:18767/health
```

为避免把调试结果当作实验结论，下面的单题验证写入新建的临时目录；BM25 与 Dense 使用同一模型、问题、Top-K 和 judge：

```bash
mkdir -p "$SGMEM_PROJECT/results"
SGMEM_SMOKE_HOST=$(mktemp -d "$SGMEM_PROJECT/results/hpc-smoke.XXXXXXXX")
SGMEM_SMOKE_CONTAINER="/workspace/results/${SGMEM_SMOKE_HOST##*/}"
"${SGMEM_EXEC[@]}" python -m sgmem split --domains Finance --output "$SGMEM_SMOKE_CONTAINER/split.json"
for SGMEM_METHOD in bm25 dense; do
  "${SGMEM_EXEC[@]}" python -m sgmem run --method "$SGMEM_METHOD" --device cuda \
    --embedding-model /workspace/models/e5-base-v2 \
    --answer-model "$SGMEM_MODEL" --base-url http://127.0.0.1:18767/v1 \
    --domains Finance --categories multi_hop --split "$SGMEM_SMOKE_CONTAINER/split.json" \
    --partition dev --limit 1 --top-k 10 --timeout 300 \
    --output "$SGMEM_SMOKE_CONTAINER/$SGMEM_METHOD"
done
```

检查两种方法的 `metrics.json`：服务错误数应为 0，judge 应能解析；不要将单题结果写成正式准确率。结束后停止本次服务并清理刚才 `mktemp` 创建的目录，再退出作业 shell：

```bash
cat "$SGMEM_SMOKE_HOST/bm25/metrics.json" "$SGMEM_SMOKE_HOST/dense/metrics.json"
kill "$SGMEM_QWEN_PID"
wait "$SGMEM_QWEN_PID" 2>/dev/null || true
rm -rf -- "$SGMEM_SMOKE_HOST"
exit
```

正式实验应使用持久的独立输出目录、固定 split、完整配置和实际资源记录，不使用 smoke 目录；先完成基线失败分析，再冻结方法并运行测试集。

### 可选：换成 Qwen3-8B

如果有足够显存和 GPU 时间，可在允许联网的节点下载 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) 的固定 revision；BF16/FP16 权重按约 8.2B 参数估算约 16.4 GB，实际运行还需显存存放上下文、运行时和 E5。A40 官方规格是 48 GB，但目标节点的实际可用显存应以分配后的 `nvidia-smi` 为准。

```bash
apptainer exec --pwd /workspace --bind "$SGMEM_PROJECT:/workspace" "$SGMEM_IMAGE" \
  hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /workspace/models/qwen3-8b
```

Qwen3-8B 默认启用 thinking；现有 QA 和 judge 的输出上限及解析器尚未针对该模式验证。先停止前一个服务，再以当前镜像的 Transformers 服务显式关闭 thinking，并验证输出和 judge 解析：

```bash
SGMEM_MODEL=/workspace/models/qwen3-8b
"${SGMEM_EXEC[@]}" transformers serve "$SGMEM_MODEL" --reasoning off \
  --host 127.0.0.1 --port 18767 --device cuda --dtype float16 &
SGMEM_QWEN_PID=$!
```

等待服务就绪后复用上面的单题测试，`--answer-model` 必须与服务加载路径一致。若正式比较 BM25、Dense 和后续方法，所有方法必须使用同一冻结模型、thinking 配置、prompt 和 judge；不能把 4B 与 8B 的结果混成同一对照。vLLM 可作为后续吞吐优化，但当前镜像没有安装它，需另行选择兼容的镜像和运行配置并验证。
