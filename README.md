# patchcore-unified-eval

基于作者官方 PatchCore 源码，对 MVTec AD、VisA、BTAD 使用统一的 full-frame256 输入和评价坐标，并提供双 GPU 分阶段执行。

**本仓库只保存代码、测试、文档和环境配置。** 数据集、预训练权重、memory bank、连续预测、实验结果、运行日志和缓存均不提交。官方源码以固定 commit 的 Git submodule 引用；本仓库不上传上游附带的示例模型。

## 克隆与 Kaggle 同步

在 Kaggle notebook 的代码单元中运行以下命令，需要启用网络及对本仓库的读取权限：

```python
!GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/YYK001/patchcore-unified-eval.git /kaggle/working/patchcore-unified-eval
%cd /kaggle/working/patchcore-unified-eval
```

后续同步（不覆盖有冲突的本地修改）：

```python
%cd /kaggle/working/patchcore-unified-eval
!git pull --ff-only
!GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

`GIT_LFS_SKIP_SMUDGE=1` 避免拉取上游仓库的大型示例 bank；这些 bank 不参与本实现。官方源码固定为 `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`，来源和许可证参见 [SOURCE.json](external_baselines/patchcore_official_eval/SOURCE.json)。

自行在 Kaggle 挂载数据和原始 V1 权重；将数据/权重路径指向 `/kaggle/input/...`，输出目录指向代码仓库之外的 `/kaggle/working/...`。不要将凭据写入 notebook 或代码库。运行前检查当前会话实际 GPU 配置与依赖；正式路径需要两个不同 CUDA 设备和 FAISS GPU，不会静默回退 CPU。

## 运行

[完整运行文档](external_baselines/patchcore_official_eval/README.md) 包括环境声明、BTAD 单类检查、benchmark、三数据集正式命令，以及保存材料和指标定义。

独立入口：`check`、`benchmark`、`fit`、`predict`、`evaluate`。例如只查看参数：

```bash
python -m external_baselines.patchcore_official_eval --help
python -m external_baselines.patchcore_official_eval.server_commands --help
```

公共数据解析和评价模块保留原包路径；这里只包含静态导入所需的代码，不包含原实验工作区的全部内容，也不要求 DINO 模型权重。

## 验证状态

本地只执行 Python 标准库验证，13 项通过；源码语法编译通过。PyTorch/FAISS 数值测试、真实数据扫描、V1 权重实际加载、双 T4 显存/耗时和 Kaggle 正式运行均尚未验证。

```bash
python -m unittest external_baselines.patchcore_official_eval.tests.test_stdlib -v
```

数值测试代码和服务器执行方式见完整运行文档。此协议使用整图256×256，不是官方224中心裁剪输入协议的逐项复刻。
