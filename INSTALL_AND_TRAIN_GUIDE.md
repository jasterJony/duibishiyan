# MMDetection 环境配置与训练指南

> 适用于 Windows + NVIDIA RTX 4060 Ti (CUDA 13.0) 环境

---

## 一、完整安装方案（从零开始）

打开 **PowerShell** 或 **Anaconda Prompt**，按顺序执行：

```bash
# ============================================
# 步骤 1：创建新环境
# ============================================
conda create -n mmdet python=3.10 -y

# ============================================
# 步骤 2：激活环境
# ============================================
conda activate mmdet

# ============================================
# 步骤 3：安装 PyTorch（CUDA 12.1）
# ============================================
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# ============================================
# 步骤 4：安装 MMCV 和 MMEngine
# ============================================
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmengine>=0.7.1

# ============================================
# 步骤 5：安装其他依赖
# ============================================
pip install matplotlib numpy pycocotools scipy shapely six terminaltables tqdm

# ============================================
# 步骤 6：克隆项目
# ============================================
git clone https://github.com/jasterJony/duibishiyan.git
cd duibishiyan

# ============================================
# 步骤 7：安装 MMDetection
# ============================================
pip install -e .

# ============================================
# 步骤 8：验证安装
# ============================================
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "import mmdet; print(f'MMDetection: {mmdet.__version__}')"

# ============================================
# 步骤 9：开始训练
# ============================================
python tools/train.py coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/faster_rcnn_coco8
```

---

## 二、一键复制版

### 第一段：创建环境和安装依赖

```bash
conda create -n mmdet python=3.10 -y && conda activate mmdet && pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 && pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html && pip install mmengine>=0.7.1 matplotlib numpy pycocotools scipy shapely six terminaltables tqdm
```

### 第二段：克隆项目和安装

```bash
git clone https://github.com/jasterJony/duibishiyan.git && cd duibishiyan && pip install -e . && python -c "import torch, mmdet; print('安装成功!')"
```

---

## 三、基于现有 YOLOv11 环境安装（更快）

如果已有 YOLOv11 环境，直接在其中安装：

```bash
# 激活 yolo 环境
conda activate yolo11

# 安装 MMDetection 依赖
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmengine>=0.7.1 pycocotools shapely terminaltables

# 克隆项目
git clone https://github.com/jasterJony/duibishiyan.git
cd duibishiyan

# 安装 mmdet
pip install -e .
```

---

## 四、验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "import mmdet; print(f'MMDetection: {mmdet.__version__}')"
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
```

预期输出：
```
PyTorch: 2.4.0, CUDA: True, GPU: NVIDIA GeForce RTX 4060 Ti
MMDetection: 3.3.0
MMCV: 2.1.0
```

---

## 五、数据集说明

项目已包含 `coco8` 小数据集用于测试，位于 `data/coco8/`：

```
data/coco8/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   └── *.jpg (4张训练图片)
└── val2017/
    └── *.jpg (4张验证图片)
```

---

## 六、可用模型列表

| 模型 | 配置文件 | 说明 |
|------|----------|------|
| Faster R-CNN | `coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py` | 经典两阶段检测器 |
| Cascade R-CNN | `coco8_configs/configs/cascade_rcnn/cascade-rcnn_r50_fpn_2e_coco8.py` | 级联检测器 |
| Mask R-CNN | `coco8_configs/configs/mask_rcnn/mask-rcnn_r50_fpn_2e_coco8.py` | 实例分割 |
| HTC | `coco8_configs/configs/htc/htc_r50_fpn_2e_coco8.py` | 混合任务级联 |
| SCNet | `coco8_configs/configs/scnet/scnet_r50_fpn_2e_coco8.py` | 自校准网络 |
| Grid R-CNN | `coco8_configs/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2e_coco8.py` | 网格引导检测 |

---

## 七、训练命令

### 7.1 单模型训练

```bash
# 激活环境
conda activate mmdet

# 进入项目目录
cd duibishiyan

# 训练 Faster R-CNN
python tools/train.py coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/faster_rcnn_coco8

# 训练 Cascade R-CNN
python tools/train.py coco8_configs/configs/cascade_rcnn/cascade-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/cascade_rcnn_coco8

# 训练 Mask R-CNN
python tools/train.py coco8_configs/configs/mask_rcnn/mask-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/mask_rcnn_coco8

# 训练 HTC
python tools/train.py coco8_configs/configs/htc/htc_r50_fpn_2e_coco8.py --work-dir work_dirs/htc_coco8

# 训练 SCNet
python tools/train.py coco8_configs/configs/scnet/scnet_r50_fpn_2e_coco8.py --work-dir work_dirs/scnet_coco8

# 训练 Grid R-CNN
python tools/train.py coco8_configs/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2e_coco8.py --work-dir work_dirs/grid_rcnn_coco8
```

### 7.2 批量训练所有模型

双击运行 `train_all_models.bat`，或在命令行执行：

```bash
train_all_models.bat
```

---

## 八、测试/推理命令

### 8.1 模型测试

```bash
python tools/test.py coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py work_dirs/faster_rcnn_coco8/epoch_2.pth
```

### 8.2 图片推理

```bash
python demo/image_demo.py demo/demo.jpg coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py --weights work_dirs/faster_rcnn_coco8/epoch_2.pth --out-dir results/
```

---

## 九、常用训练参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--work-dir` | 模型保存目录 | `--work-dir work_dirs/my_model` |
| `--resume` | 从断点恢复训练 | `--resume work_dirs/my_model/epoch_1.pth` |
| `--amp` | 混合精度训练（省显存） | `--amp` |

---

## 十、常见问题

### Q1: CUDA out of memory（显存不足）

```bash
# 使用混合精度训练
python tools/train.py config.py --amp
```

或修改配置文件中的 batch_size：
```python
train_dataloader = dict(batch_size=1)  # 从 2 改为 1
```

### Q2: 找不到数据集

确保 `data/coco8/` 目录存在且结构正确。

### Q3: mmcv 版本不兼容

```bash
pip uninstall mmcv mmcv-full -y
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

### Q4: Windows 路径问题

确保项目路径中没有中文或空格。

---

## 十一、训练输出说明

训练完成后，`work_dirs/` 目录结构：

```
work_dirs/
└── faster_rcnn_coco8/
    ├── epoch_1.pth             # 第1轮权重
    ├── epoch_2.pth             # 第2轮权重（最终）
    ├── last_checkpoint         # 最新检查点路径
    └── *.py                    # 配置备份
```
