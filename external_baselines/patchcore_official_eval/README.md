# PatchCore 官方算法 + full-frame256 三数据集统一复评

本目录是本地交付的运行适配代码。**没有连接服务器、上传、启动正式实验，也没有执行 PyTorch/FAISS 数值测试。** 本地只执行 Python 标准库检查；服务器环境、真实权重加载、完整数据解析、双 T4 显存和数值一致性仍待验证。不能将本交付称为“已在双 T4 跑通”。

## 1. 先运行 BTAD 单类检查与 benchmark

以下全部是供用户在服务器项目根目录手动执行的 Bash 命令。先按第 4 节准备环境与权重。`server_commands` 本身不安装环境，不 SSH、不上传。它逐阶段保存日志、精确 argv 和退出码；任一阶段失败立即停止后续阶段。`--dry-run` 只输出命令。

```bash
cd /path/to/experiment
export BTAD_ROOT=/data/BTAD/BTech_Dataset_transformed
export MVTEC_ROOT=/data/mvtec
export VISA_ROOT=/data/VisA
export WEIGHTS=/weights/wide_resnet50_2-95faca4d.pth
export RUNS=/results/patchcore_official_fullframe256_seed42_v1
export GPU_MODEL=0
export GPU_CORESET=1

# 检查 BTAD/01 的完整数据计数、对应关系、GT、权重、设备与依赖。
# 小型合成 FAISS/指标探针不使用测试统计选择模型配置。
python -m external_baselines.patchcore_official_eval.server_commands \
  --phase check --dataset btad --category 01 \
  --dataset-root "$BTAD_ROOT" --output-dir "$RUNS/btad" --weights "$WEIGHTS" \
  --gpu-model "$GPU_MODEL" --gpu-coreset "$GPU_CORESET"

# 必须是另一个输出目录；默认仅 2 张训练正常图和 2 张测试图。
# 仍保留 256、1024维、0.1 coreset、FP32；不作为正式模型或正式指标。
python -m external_baselines.patchcore_official_eval.server_commands \
  --phase benchmark --dataset btad --category 01 \
  --dataset-root "$BTAD_ROOT" --output-dir "$RUNS/btad_01_benchmark" --weights "$WEIGHTS" \
  --gpu-model "$GPU_MODEL" --gpu-coreset "$GPU_CORESET" --batch-size 2
```

`benchmark_report.json` 汇总实际阶段耗时、两卡各自的 PyTorch 峰值和可用的整卡/进程采样最大值；`01/{fit,predict}/resources.csv` 和 `nvidia_smi_samples.jsonl` 是详细记录。没有实际执行前，不存在实测耗时/显存结果。全量 greedy 计算量可非常大，不能按两张图的 benchmark 线性承诺全量时间。

日志在输出目录旁的 `<输出目录名>_logs/<UTC时间>/`，每阶段 `.log`、`.argv.json`、`.exit.json` 各一份。不要在未看退出码的情况下继续手动执行依赖阶段。多个独立 Bash 命令顺序运行时可先 `set -e`。

## 2. BTAD 三类正式命令

检查和 benchmark 达到预期、数值测试通过之后，再手动执行。`full` 严格按 **fit → predict → evaluate**，失败停止；类别在每个阶段内串行，不启动 DDP/DataParallel。

```bash
python -m external_baselines.patchcore_official_eval.server_commands \
  --phase full --dataset btad --category all \
  --dataset-root "$BTAD_ROOT" --output-dir "$RUNS/btad" --weights "$WEIGHTS" \
  --gpu-model "$GPU_MODEL" --gpu-coreset "$GPU_CORESET" --batch-size 8
```

独立运行各阶段也使用同一日志包装器，将 `--phase full` 分别改为 `--phase fit`、`--phase predict`、`--phase evaluate`。独立入口完整示例如下；直接入口的退出码由 shell 返回，若需自动保存日志/退出码优先使用上面的包装器。

```bash
python -m external_baselines.patchcore_official_eval fit \
  --dataset btad --category all --dataset-root "$BTAD_ROOT" --output-dir "$RUNS/btad" \
  --backbone-weights "$WEIGHTS" --model-device "cuda:$GPU_MODEL" \
  --coreset-device "cuda:$GPU_CORESET" --nn-device "cuda:$GPU_MODEL" --metric-device "cuda:$GPU_MODEL"

python -m external_baselines.patchcore_official_eval predict \
  --dataset btad --category all --dataset-root "$BTAD_ROOT" --output-dir "$RUNS/btad" \
  --model-device "cuda:$GPU_MODEL" --coreset-device "cuda:$GPU_CORESET" \
  --nn-device "cuda:$GPU_MODEL" --metric-device "cuda:$GPU_MODEL"

python -m external_baselines.patchcore_official_eval evaluate \
  --dataset btad --category all --dataset-root "$BTAD_ROOT" --output-dir "$RUNS/btad" \
  --metric-device "cuda:$GPU_MODEL"
```

上面两种形式是替代用法，**不要在同一个完成目录重复执行**。已完成类别需显式 `--skip-completed`，且必须匹配协议、训练/测试清单、几何与必要保存文件。未完成类别不能静默覆盖，使用新输出目录；此前成功保存的 bank/预测仍保留。多个独立进程不能同时写同一输出目录。

## 3. MVTec AD 15 类与 VisA 12 类

三个数据集使用同一 `protocol.py` 冻结配置，不根据 BTAD 测试结果调整参数。

```bash
python -m external_baselines.patchcore_official_eval.server_commands \
  --phase check --dataset mvtec --category all \
  --dataset-root "$MVTEC_ROOT" --output-dir "$RUNS/mvtec" --weights "$WEIGHTS" \
  --gpu-model "$GPU_MODEL" --gpu-coreset "$GPU_CORESET"

python -m external_baselines.patchcore_official_eval.server_commands \
  --phase full --dataset mvtec --category all \
  --dataset-root "$MVTEC_ROOT" --output-dir "$RUNS/mvtec" --weights "$WEIGHTS" \
  --gpu-model "$GPU_MODEL" --gpu-coreset "$GPU_CORESET"

python -m external_baselines.patchcore_official_eval.server_commands \
  --phase check --dataset visa --category all \
  --dataset-root "$VISA_ROOT" --output-dir "$RUNS/visa" --weights "$WEIGHTS" \
  --gpu-model "$GPU_MODEL" --gpu-coreset "$GPU_CORESET"

python -m external_baselines.patchcore_official_eval.server_commands \
  --phase full --dataset visa --category all \
  --dataset-root "$VISA_ROOT" --output-dir "$RUNS/visa" --weights "$WEIGHTS" \
  --gpu-model "$GPU_MODEL" --gpu-coreset "$GPU_CORESET"
```

单类可将 `all` 改为合法类别名。VisA 复用现有 `1cls.csv` 解析器：即使选择单类，公共解析器仍验证全部 12 类清单存在。BTAD 公共入口仍要求根目录包含 01/02/03 目录，但只读取指定类别的样本。数据根目录可以是现有适配入口支持的外层目录。

首次评价将原始 GT 按现有最近邻函数转为二值 mask 并压缩保存；之后可以不提供数据根目录、权重或模型材料复算：

```bash
python -m external_baselines.patchcore_official_eval evaluate \
  --dataset btad --category all --output-dir "$RUNS/btad" \
  --metric-device "cuda:$GPU_MODEL" --evaluation-name metrics_recomputed
```

`--evaluation-name` 必须是新名称，不覆盖旧指标。默认正式评价必须 CUDA；`--cpu-check` 是显式非正式小规模检查选项，不隐式回退。没有 CPU 训练/预测的正式入口。

## 4. 服务器环境与离线权重

建议的 Linux 环境声明是 `environment.server.yml`，**未在本地求解或在服务器验证**。它使用现代兼容依赖，非官方 2021 年 requirements 的逐项环境复刻。驱动必须支持所安装 CUDA runtime。不要同时安装 `faiss-cpu` 和 `faiss-gpu`，也不要直接覆盖现有实验环境。

```bash
conda env create -f external_baselines/patchcore_official_eval/environment.server.yml
conda activate patchcore-official
python -c 'import torch, torchvision, faiss; print(torch.__version__, torchvision.__version__); print(torch.cuda.device_count()); assert hasattr(faiss, "StandardGpuResources")'
```

准备的是完整本项目代码树，而不只是此目录：公共评价导入依赖 `DINOv3/`、`DINOv2/nvs/`、`external_baselines/superadd_external/` 的现有代码。它们不需要训练或加载 DINO 模型，但必须能被导入。所有命令从项目根目录执行。

官方源码位于 `external_baselines/patchcore_official/`，来源 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)，commit **fcaa92f124fb1ad74a7acf56726decd4b27cbcad**，Apache-2.0，保留 `LICENSE` 和 `NOTICE`。`verify_source()` 检查 commit 及 `src/`/许可证未修改。未调用官方仓库附带的示例预训练 bank。

若服务器尚未准备此 checkout，以下命令仅拉代码并跳过无用的大型示例 bank（本交付不执行）：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/amazon-science/patchcore-inspection.git external_baselines/patchcore_official
git -C external_baselines/patchcore_official checkout fcaa92f124fb1ad74a7acf56726decd4b27cbcad
```

WRN50-2 必须使用 [官方 V1 权重文件](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth)，对应历史 `pretrained=True`。现代 `Wide_ResNet50_2_Weights.DEFAULT` 是 V2，因此禁止使用 DEFAULT；见 [torchvision 文档](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.wide_resnet50_2.html)。

在有网络的准备机器手动下载这份文件，再通过自己的交付流程提供给服务器；本代码从不联网下载权重。`--backbone-weights` 接受本地路径；单一 SHA256 前缀 `95faca4d` 验证原始分发文件，完整摘要保存在 `run.json`。重打包 state_dict 的文件即使参数一样也被拒绝，以避免不明确的权重来源。模型用 `weights=None` 构造，再严格加载这份离线 state_dict；模型及运算为 FP32、冻结、eval。

## 5. 科学协议与适配边界

这次是 **“官方算法实现 + full-frame256 外部数据集统一复评”**，不是官方论文原始输入协议的逐项复现。官方 README 示例是 resize256 后 center crop224；这里 RGB 整图直接双线性 resize256×256，不裁剪、无增强，训练/测试同样 ImageNet normalization。seed 从示例中的 0 改为指定的 42，每类别重新设置。

固定参数：WRN50-2 V1；layer2/3；patchsize3、stride1；pretrain/target embedding 均1024；官方 ApproximateGreedy ratio0.1、随机 Linear 投影128、10个起始点；单模型、1近邻、FP32，无 AMP/FP16/IVF/PQ。

| 路径 | 复用与变化 |
|---|---|
| 特征 | 原始 `PatchCore._embed`、`PatchMaker.patchify`、跨层 bilinear、`Preprocessing`、`MeanMapper`、`Aggregator`。子类仅把最终整个 batch 一次转 CPU，避免每 patch 的 Python 数组列表。 |
| 构库 | 每批直接写 CPU `.npy` memmap，训练图全部使用；没有 list+全库 concatenate，也没有 index%8 校准划分。 |
| coreset | 子类继承官方 `ApproximateGreedyCoresetSampler`；一次初始化相同 dense `torch.nn.Linear(..., bias=False)`，所有块共享。GPU1 持有 N×128 和 N×1；距离临时块至多 chunk×10/1。 |
| greedy | 起始点 `np.random.choice`；初始为到10起点的欧氏距离均值；每轮全局 argmax，再用到新点距离取 minimum；没有每块独立选择。复用官方距离函数的 sqrt（**仅 coreset 距离**）。 |
| 最终 bank | 用索引从 CPU **1024维**原特征取行保存；128维不参与最终检索。无须永久再存一份含相同向量的 FAISS 文件，predict 从 bank 重建 FlatL2 即可离线重推。 |
| 检索/评分 | 官方 `NearestNeighbourScorer.predict` 和 `PatchCore._predict`。FAISS wrapper 绑定设备/资源并分块 add/search，保持精确平方 L2。官方 image score 原样保存，不开平方、不除1280、不做MAD/LOCAL/GUIDED。 |
| 定位 | 原始 `RescaleSegmentor`：先双线性上采样至256，再 Gaussian sigma4；再把完整256图双线性映射至原始 H//4,W//4，align_corners=False。不逐图归一化。 |

浮点说明：分块 GEMM 与整块 GEMM、不同 CUDA/FAISS 后端可以有浮点末位差异；接近并列的点可能改变后续 greedy 路径。精确并列沿用 PyTorch argmax 的第一个行号，不加扰动、不擅自去重。测试在相同投影值/起点下对照索引，投影与预测用数值容差，不宣称跨设备逐位相同。禁止为了 OOM 降低科学参数。

## 6. 双 T4、工程参数与资源口径

默认 model/NN/metric=`cuda:0`，coreset=`cuda:1`；允许显式改成其他两张卡。两个索引是当前进程的逻辑 CUDA 编号，受 `CUDA_VISIBLE_DEVICES` 影响；不是两卡显存合并。正式路径检查两个不同设备，绝不静默使用 CPU FAISS。

GPU0 每批提取后立即转 CPU memmap；提取结束释放模型，GPU1 完成投影/选择；bank 立即持久化；GPU1 临时张量释放。predict 新进程将 WRN 和最终 bank 对应 FlatL2 放到 GPU0。FAISS 的 `StandardGpuResources` 由 wrapper 持有，`GpuIndexFlatConfig.device` 明确指定，临时池默认256MiB，FP32索引。见 [FAISS GPU 资源说明](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)。

工程参数可调：`--batch-size 8`、`--projection-chunk 8192`、`--distance-chunk 65536`、`--query-chunk 4096`、`--faiss-temp-mb 256`、直接入口的 `--num-workers 0`。显存不足时先降 batch/chunk，用新输出路径重试；不会自动改配置。约 N×1024×4 字节的全特征缓存和 N×128×4 字节的投影需求应按真实类别评估，磁盘与 CPU RAM 也不能忽略。

阶段分别记录 feature_extraction、cpu_cache_write、coreset_projection、coreset_selection、bank_save、index_build、model_inference、prediction_save、metric_evaluation 等。每阶段同步所用 CUDA 卡，分别 reset/记录每张卡的 allocated/reserved 峰值；不相加写成“双卡总峰值”。`environment.json` 记录型号、总显存、启动时可用显存、逻辑设备、UUID（PyTorch可提供时）、环境版本。

PyTorch 统计不覆盖全部 FAISS/CUDA 分配。另通过 `nvidia-smi` 在阶段开始和其后约每0.5秒采样整卡显存及本进程显存，按**物理 UUID**记录 raw 与 observed maximum。查询自身有延迟，短阶段可能仅一个样本；WDDM/MIG/容器可能无进程值，缺失表示 unavailable，不能当0，也不能称精确峰值。阶段计时包含该阶段的数据传输及同步；fit/evaluate 总时长不是纯推理延迟，包含 Gaussian CPU 后处理的 model_inference 也不是仅网络 forward。

## 7. 数据和统一评价

复用现有 `relation_reliability.datasets.mvtec_ad_records`、`visa_task_decoupled.dataset.visa_one_class_records`、`btad_validation.dataset.records` 的清单/划分/标签/mask对应。忽略 BTAD 公共入口附带的 memory/calibration manifest，自己保存全部正常训练的角色 `memory_fit_all_official_train_normal`。训练和预测的 `Images` 在边界丢弃标签及mask，只返回图片。

BTAD 强制核对：01=(400,21,49)，02=(399,30,200)，03=(1000,400,41)，顺序为 train normal/test normal/test anomaly。不会删除空mask异常图或改标签。评价保存原始前景像素数与评价坐标空mask标记，区分官方原始空标注与下采样后消失。逐图键固定为 category + relative_path。

评价复用：

- `mvtec_broad6_compose2.metrics.evaluate_fast`：所有测试正常+异常图的像素 AUROC、average precision、AUPRO@0.3；现有快速 CUDA，200个类别全局线性阈值，4连通，FPR上限0.3，不调用 ADEval。
- `mvtec_task_decoupled.runner.image_metrics`：用**官方 image_score**覆盖前者默认 map.max 得到的图像指标。输出列重命名为 `image_AP`/`pixel_AP`，明确 AP 而非PR曲线梯形面积。
- `mvtec_broad6_compose2.runner._load_mask`：GT最近邻缩放再 `>0`；完整视野。
- `normal_reconstruction.scoring.fixed_fpr_diagnostics`：FPR≤1%、≤5%的实际FPR、缺陷像素召回、区域平均覆盖、小区域覆盖与数量。沿用 whole-tie `score > boundary`、4连通、小区域面积≤图像评价像素数0.001；无区域为 N/A。
- `guided_validation.summary.mean_available`：固定FPR类间等权，排除N/A并报告有效类别数；数量单独求和。五指标类间等权，不跨三个数据集混合macro。非全类执行的macro明确标为 selected_categories、NOT_full_dataset。

所有阈值仅事后评价，不写回模型、构库、分数或配置。没有 bootstrap、多seed或调参扫描。

## 8. 保存材料与恢复

默认全部保存，无“只存五指标”选项。输出结构：

```text
run.json                         协议/数据根/权重摘要/来源
shared/                          一份原始V1权重、适配源码、官方src与LICENSE/NOTICE
checks/<类别列表>/               清单、计数、mask审计、依赖探针状态
<category>/fit/
  bank.npy                       最终FP32 1024维bank（唯一权威bank）
  coreset_indices.npy            全特征库行号
  model.json                     配置、起点、行数、共享权重路径
  train_manifest.json            全量正常训练清单与几何
  full_features.npy              CPU memmap中间缓存
  complete.json / failure.json   阶段状态
<category>/predict/
  maps/*.npz                     FP32 input_map(256×256)、evaluation_map(H//4,W//4)
  samples.json, sample_scores.csv 官方image_score、键、标签、mask相对路径、三组尺寸
  test_manifest.json             预测前锁定的测试元数据
  complete.json / failure.json
<category>/ground_truth/maps/*.npz 压缩bool评价mask、键、标签、原始前景计数
evaluations/<名称>/              逐类/macro五指标、固定FPR及macro、mask审计、状态
各阶段 arguments/environment/command.json、resources.csv、nvidia_smi_samples.jsonl
```

bank/索引/模型配置在构库结束即保存；预测每批落盘并更新清单，评价失败不丢失这些材料。`full_features.npy` 可以在 fit 完成后由用户手动管理/删除；代码不自动清理最终bank或预测，也不自动删除缓存。predict 重建 FAISS 索引只读取最终 bank，不重新提训练特征或选 coreset。原图尺寸和输入/评价尺寸保存在清单，不用热力图PNG或量化图替代连续输出。

可分别归档：**轻量报告包**包含 JSON/CSV/日志/源码来源；**模型/预测包**包含 shared权重、每类bank/coreset/model、所有连续预测、压缩GT和清单。只有前者不能叫“已完整归档”。可排除中间全特征缓存。独立重新推理需要完整项目公共代码树及原始测试图；只复算指标在GT快照完整后不需要原始图、权重、bank、FAISS或官方源码导入。

## 9. 测试与未验证清单

本地仅运行标准库：

```bash
python -m unittest external_baselines.patchcore_official_eval.tests.test_stdlib -v
```

本地13项标准库检查通过，详情见仓库根目录 README 的验证状态。语法/CLI通过不代表模型数值或服务器显存通过。本代码仓库不提交本地实验报告或运行日志。

服务器有依赖后手动执行数值检查（不构建真实数据集的正式bank）：

```bash
export PC_V1_WEIGHTS="$WEIGHTS"
python -m pytest external_baselines/patchcore_official_eval/tests/test_runtime.py -q

export PC_DUAL_T4=1
export PC_MODEL_DEVICE="cuda:$GPU_MODEL"
export PC_CORESET_DEVICE="cuda:$GPU_CORESET"
python -m pytest external_baselines/patchcore_official_eval/tests/test_runtime.py -q
```

覆盖：官方embedding/图像分数/连续图一致性；同投影/起点的分块coreset及精确并列；FAISS分块与平方L2；新进程重载bank复现且禁止fit/coreset；无模型/权重/原图的保存结果复算；非正方形坐标、GT最近邻、空mask、macro/N/A；标签/mask元数据改变不影响特征/bank输入或预测。默认用小型合成骨干验证机制；`PC_V1_WEIGHTS` 打开真实WRN50-2对照；`PC_DUAL_T4=1` 打开真实两张T4与GPU FAISS检查，不以CPU冒充GPU。

截至本地交付，上述全部数值测试、三数据集真实扫描、V1实际加载、双T4完整类别资源/吞吐和CUDA指标均**未执行**。环境创建命令也未在服务器执行。需要先数值测试、check、benchmark，再决定是否手动启动正式命令。
