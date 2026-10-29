# AEGIS 开源代码说明

本仓库对应 MICCAI 2026 论文 *AEGIS: Anatomy-Embedded Group-Invariant
Segmentation for Fair Medical Foundation Models*，由现有实验代码和论文整理而成。
英文主页提供完整命令与接口：[README.md](README.md)。

## 包含什么

- 精简的 SAM / MedSegX / ConMoAE 骨干；只训练适配器和 mask decoder。
- 按论文重建的属性条件原型对齐、历史 EMA 群体权重，以及 5 epoch 预热。
- 数据预处理、患者级划分检查、训练、断点恢复、推理和公平性评估。
- 合成数据示例、测试、依赖配置、Apache-2.0 许可证及上游声明。

默认参数见 `configs/hnscc.yaml`。使用 `configs/smoke.yaml` 可以在 CPU 上运行
完整软件流程；它使用小型测试网络，不能用来证明论文效果。

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
bash scripts/smoke.sh
pytest -q
```

## 使用自己的数据

1. 按 [数据说明](docs/DATA.md) 准备 NPY 切片和 manifest，患者不能跨训练、验证、测试集。
2. 从 SAM 和 MedSegX 官方仓库获取权重，填写配置中的本地路径。
3. 运行英文主页的训练命令；每次选择一种敏感属性，例如 site 或 age。
4. 用验证集选择 `best.pt`，再对固定测试集报告 Dice 和 EA Gap。

EA Gap 按论文定义为各组平均 Dice 的最大值减最小值，单位为百分点。
没有人口学标签时可以使用 `aegis.infer`，但必须提供 bounding box。
默认评估从真值 mask 生成框，属于有框提示的分割评估。

## 复现边界

这是一份补齐并修正的发布实现，不是已经验证可恢复论文全部表格的实验存档。
已有代码与论文在原型归一化、配对求和尺度、历史权重和总损失等方面有冲突；
默认实现按论文公式处理，差异见 [METHOD.md](docs/METHOD.md)。

官方 MedSegX 权重没有 HNSCC 任务行。默认配置显式从 PharynxCancer 初始化该行，
这是为补齐运行链路作出的选择，不能说成已确认的原实验配置。原始训练权重、
患者划分和 Harvard-FairSeg / PI-CAI 的完整转换流程没有随现有目录提供，
因此没有虚构下载地址、原始划分或已复现指标。

患者索引 CSV、临床影像、实验结果、权重、论文 PDF 和旧脚本没有打入源码包。
