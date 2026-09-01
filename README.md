# SwinIR‑Med：基于 SwinIR 的医学 CT 影像 4× 超分辨率重建

本项目实现了一个面向医学 CT 影像的 **4× 超分辨率（Super‑Resolution）** 模型 **SwinIR‑Med**，可将 128×128 的低分辨率 CT 切片重建为 512×512 的高分辨率影像，并以 DICOM（IMA）格式输出。

## 特性

- **高低频分治残差结构**：主干由高频分支（Swin Transformer 块，保留边缘细节）与低频分支（大核深度卷积，保证平滑区域连续性、抑制伪影）融合而成。
- **盲退化建模（Degradation‑Aware）**：通过轻量编码器提取退化特征，对浅层特征做 scale/shift 调制，提升对真实退化（如不同层厚、噪声）的鲁棒性。
- **CARAFE 风格上采样 + U‑Net 跳跃连接**：结合全局双三次上采样残差与浅层特征跳跃连接，稳定重建。
- **多损失约束**：L1 像素损失 + 边缘（Sobel）损失 + 一阶/二阶全变分平滑损失。
- **DICOM 全流程 IO**：读取 `.ima/.dcm` 并还原 HU 窗宽窗位，输出标准 IMA 文件（层厚 1mm，尺寸 512×512，HU 范围 ‑1000~400）。
- **混合精度训练（AMP）**：使用 `torch.amp` 的 `GradScaler` / `autocast` 加速训练并节省显存。

## 环境依赖

- Python 3.8+
- PyTorch 1.12+（含 `torch.amp`）
- torchvision
- pydicom
- opencv‑python (cv2)
- numpy
- scikit‑image
- tqdm

可使用项目内的虚拟环境 `venv`（Windows）：

```powershell
pip install torch torchvision pydicom opencv-python numpy scikit-image tqdm
```

## 目录结构

```
newCT/
├── model.py            # 网络定义（SwinIRMed）、损失（SwinIRMedLoss）、token/image 转换辅助
├── utils.py            # DICOM IO、数据变换、模型工厂 build_swinir_med、验证函数
├── train.py            # 训练入口
├── test.py             # 批量推理入口
├── dataset/
│   ├── 128x128/        # 低分辨率（LR）训练影像
│   ├── 512x512/        # 高分辨率（HR）训练影像
│   └── test_4x/        # 测试集（LR）
└── result_swinir/      # 训练指标与推理结果输出
```

## 数据准备

- LR / HR 影像为配对的 CT 切片，分辨率需为 **128×128** 与 **512×512**。
- 支持 `.ima` / `.dcm` / `.png` / `.jpg` / `.jpeg` 等常见格式（DICOM 会按 HU 窗宽窗位归一化到 [0,255]）。
- 数据集目录下 LR、HR 文件名需一一对应。

## 训练

```powershell
python train.py
```

默认配置（可在 `train.py` 的 `__main__` 中修改）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `LR_DIR` | `dataset/128x128` | 低分辨率目录 |
| `HR_DIR` | `dataset/512x512` | 高分辨率目录 |
| `batch_size` | 2 | 批大小 |
| `epochs` | 1 | 训练轮数（按需调大） |
| `learning_rate` | 1e‑4 | 学习率（AdamW） |
| `upscale_factor` | 4 | 超分倍数 |
| `val_ratio` | 0.1 | 验证集占比 |

训练过程会：
- 每轮在验证集上计算 PSNR / SSIM；
- 将指标写入 `result_swinir/.../training_metrics_amp.csv`；
- 在验证 PSNR 刷新时保存最佳模型 `*_best.pth`；
- 每 10 轮及末轮保存检查点（含 optimizer / scaler 状态，可断点续训）。

## 推理（测试）

```powershell
python test.py
```

默认读取 `dataset/test_4x` 下的影像，加载 `result_swinir/swinir_med_4x_medical_8/swinir_med_4x_sr_amp_best.pth`，输出到 `result_swinir/swinir_med_4x_medical_8/test_output_amp/`，文件名形如 `image25_4x_SR.ima`。

推理使用 tqdm 进度条，仅在处理失败/缩放提示时打印，结束输出一行汇总。

## 模型说明

默认超参（`utils.MODEL_CONFIG`）：

```python
img_size=128, embed_dim=60,
depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
window_size=8, mlp_ratio=2.0
```

如需修改模型规模，调整 `utils.build_swinir_med` 中的 `MODEL_CONFIG` 即可，训练与测试会自动同步。

## 许可证

仅供学习与科研使用。
