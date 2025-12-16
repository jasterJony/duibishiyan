# MMDetection 环境配置与训练指南

> 适用于 Windows + NVIDIA RTX 4060 Ti (CUDA 13.0) 环境
> 基于已有 YOLOv11 训练环境复制

---

## 一、环境配置

### 方案 A：复制 YOLOv11 环境（推荐）

如果你已有 YOLOv11 的 conda 环境，可以直接复制：

```bash
# 1. 查看现有环境名称
conda env list

# 2. 复制环境（假设 YOLOv11 环境名为 yolo11）
conda create --name mmdet --clone yolo11

# 3. 激活新环境
conda activate mmdet

# 4. 安装 MMDetection 额外依赖
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmengine>=0.7.1
pip install pycocotools shapely terminaltables

# 5. 克隆项目
git clone https://github.com/jasterJony/duibishiyan.git
cd duibishiyan

# 6. 安装 mmdet
pip install -e .
```

### 方案 B：全新安装

```bash
# 1. 创建新环境
conda create -n mmdet python=3.10 -y
conda activate mmdet

# 2. 安装 PyTorch (CUDA 12.1，兼容 CUDA 13.0)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 mmcv 和 mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmengine>=0.7.1

# 4. 安装其他依赖
pip install matplotlib numpy pycocotools scipy shapely six terminaltables tqdm

# 5. 克隆项目
git clone https://github.com/jasterJony/duibishiyan.git
cd duibishiyan

# 6. 安装 mmdet
pip install -e .
```

---

## 二、验证安装

```bash
# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 检查 MMDetection
python -c "import mmdet; print(f'MMDetection: {mmdet.__version__}')"

# 检查 mmcv
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
```

预期输出：
```
PyTorch: 2.4.0
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
MMDetection: 3.3.0
MMCV: 2.1.0
```

---

## 三、数据集准备

项目已包含 `coco8` 小数据集用于测试，位于 `data/coco8/`。

### 数据集结构
```
data/coco8/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   ├── 000000000030.jpg
│   └── 000000000034.jpg
└── val2017/
    ├── 000000000036.jpg
    ├── 000000000042.jpg
    ├── 000000000049.jpg
    └── 000000000061.jpg
```

### 使用自定义数据集

如果要使用自己的 COCO 格式数据集：

```bash
data/
└── your_dataset/
    ├── annotations/
    │   ├── instances_train.json
    │   └── instances_val.json
    ├── train/
    │   └── *.jpg
    └── val/
        └── *.jpg
```

---

## 四、可用模型列表

| 模型 | 配置文件 | 说明 |
|------|----------|------|
| Faster R-CNN | `coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py` | 经典两阶段检测器 |
| Cascade R-CNN | `coco8_configs/configs/cascade_rcnn/cascade-rcnn_r50_fpn_2e_coco8.py` | 级联检测器，精度更高 |
| Mask R-CNN | `coco8_configs/configs/mask_rcnn/mask-rcnn_r50_fpn_2e_coco8.py` | 实例分割 |
| HTC | `coco8_configs/configs/htc/htc_r50_fpn_2e_coco8.py` | 混合任务级联 |
| SCNet | `coco8_configs/configs/scnet/scnet_r50_fpn_2e_coco8.py` | 自校准网络 |
| Grid R-CNN | `coco8_configs/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2e_coco8.py` | 网格引导检测 |

---

## 五、训练命令

### 5.1 单模型训练

#### Windows PowerShell / CMD

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

### 5.2 批量训练所有模型（Windows BAT 脚本）

创建 `train_all_models.bat` 文件：

```bat
@echo off
echo ==========================================
echo Batch Training Script for MMDetection
echo Dataset: coco8
echo Models: 6 models
echo ==========================================

cd /d %~dp0

REM Activate conda environment
call conda activate mmdet

REM Create work_dirs
if not exist work_dirs mkdir work_dirs

echo.
echo [1/6] Training Faster R-CNN...
python tools/train.py coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/faster_rcnn_coco8
if %errorlevel% neq 0 echo [FAILED] Faster R-CNN

echo.
echo [2/6] Training Cascade R-CNN...
python tools/train.py coco8_configs/configs/cascade_rcnn/cascade-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/cascade_rcnn_coco8
if %errorlevel% neq 0 echo [FAILED] Cascade R-CNN

echo.
echo [3/6] Training Mask R-CNN...
python tools/train.py coco8_configs/configs/mask_rcnn/mask-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/mask_rcnn_coco8
if %errorlevel% neq 0 echo [FAILED] Mask R-CNN

echo.
echo [4/6] Training HTC...
python tools/train.py coco8_configs/configs/htc/htc_r50_fpn_2e_coco8.py --work-dir work_dirs/htc_coco8
if %errorlevel% neq 0 echo [FAILED] HTC

echo.
echo [5/6] Training SCNet...
python tools/train.py coco8_configs/configs/scnet/scnet_r50_fpn_2e_coco8.py --work-dir work_dirs/scnet_coco8
if %errorlevel% neq 0 echo [FAILED] SCNet

echo.
echo [6/6] Training Grid R-CNN...
python tools/train.py coco8_configs/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_2e_coco8.py --work-dir work_dirs/grid_rcnn_coco8
if %errorlevel% neq 0 echo [FAILED] Grid R-CNN

echo.
echo ==========================================
echo Training Complete!
echo Check work_dirs/ for trained models
echo ==========================================
dir work_dirs
pause
```

### 5.3 Linux/Mac 批量训练脚本

已有脚本 `tools/batch_train_coco8.sh`：

```bash
# 添加执行权限
chmod +x tools/batch_train_coco8.sh

# 运行
./tools/batch_train_coco8.sh
```

---

## 六、测试/推理命令

### 6.1 模型测试

```bash
# 测试 Faster R-CNN
python tools/test.py coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py work_dirs/faster_rcnn_coco8/epoch_2.pth
```

### 6.2 图片推理

```bash
# 单张图片推理
python demo/image_demo.py demo/demo.jpg coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py --weights work_dirs/faster_rcnn_coco8/epoch_2.pth --out-dir results/
```

---

## 七、常用参数说明

### 训练参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--work-dir` | 模型保存目录 | `--work-dir work_dirs/my_model` |
| `--resume` | 从断点恢复训练 | `--resume work_dirs/my_model/epoch_1.pth` |
| `--amp` | 混合精度训练（省显存） | `--amp` |
| `--auto-scale-lr` | 自动调整学习率 | `--auto-scale-lr` |

### 显存不足解决方案

如果 8GB 显存不够，修改配置文件中的 `batch_size`：

```python
# 在配置文件中添加
train_dataloader = dict(batch_size=1)  # 从 2 改为 1
```

或使用混合精度训练：

```bash
python tools/train.py config.py --amp
```

---

## 八、训练输出说明

训练完成后，`work_dirs/` 目录结构：

```
work_dirs/
└── faster_rcnn_coco8/
    ├── 20241216_120000/
    │   └── vis_data/           # 可视化数据
    ├── epoch_1.pth             # 第1轮权重
    ├── epoch_2.pth             # 第2轮权重（最终）
    ├── last_checkpoint         # 最新检查点路径
    └── faster-rcnn_r50_fpn_2e_coco8.py  # 配置备份
```

---

## 九、快速开始（一键命令）

```bash
# 1. 激活环境
conda activate mmdet

# 2. 进入项目
cd duibishiyan

# 3. 验证环境
python -c "import torch, mmdet, mmcv; print('环境OK!')"

# 4. 开始训练第一个模型
python tools/train.py coco8_configs/configs/faster_rcnn/faster-rcnn_r50_fpn_2e_coco8.py --work-dir work_dirs/faster_rcnn_coco8
```

---

## 十、常见问题

### Q1: CUDA out of memory
```bash
# 解决方案：减小 batch_size 或使用混合精度
python tools/train.py config.py --amp
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

## 十一、联系方式

如有问题，请提 Issue 或联系项目维护者。
