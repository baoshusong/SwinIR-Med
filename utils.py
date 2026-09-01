"""SwinIR‑Med 公共工具：DICOM IO、数据变换、模型工厂、验证函数"""
import datetime
import numpy as np
import torch
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import SwinIRMed

# ===================== 默认模型配置（test.py / train.py 共用） =====================
MODEL_CONFIG = dict(
    img_size=128,
    embed_dim=60,
    depths=[6, 6, 6, 6],
    num_heads=[6, 6, 6, 6],
    window_size=8,
    mlp_ratio=2.0,
)
# 归一化系数，必须与 get_default_transform 保持一致
NORM_MEAN = 0.5
NORM_STD = 0.5

# ===================== 数据变换 =====================
def get_default_transform():
    """统一的归一化流水线（与 denormalize_to_01 配套，mean/std 必须保持一致）"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[NORM_MEAN], std=[NORM_STD])
    ])

def denormalize_to_01(tensor):
    """将 [-1,1] 归一化的张量还原为 [0,1] 的 numpy 数组"""
    return (tensor.cpu().numpy() * NORM_MEAN + NORM_STD).clip(0, 1)

# ===================== 模型工厂 =====================
def build_swinir_med(upscale_factor=4, device=None):
    """按默认配置构造 SwinIRMed，避免各脚本重复书写超参"""
    model = SwinIRMed(upscale=upscale_factor, **MODEL_CONFIG)
    if device is not None:
        model = model.to(device)
    return model

# ===================== DICOM工具函数 =====================
def read_ima_image(path, return_hu=False):
    """读取 CT 切片。
    return_hu=False（默认）：返回按固定窗宽窗位(-1000~400)归一化到[0,255]的 uint8 图，供推理/可视化；
    return_hu=True：返回 HU（float32，已做 Rescale）原始数组，供训练时的随机窗宽窗位增强使用。
    """
    ds = pydicom.dcmread(path)
    hu = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        hu = hu * ds.RescaleSlope + ds.RescaleIntercept
    if return_hu:
        return hu, ds
    img = np.clip(hu, -1000, 400)
    img = (img - (-1000)) / 1400 * 255.0
    img = img.astype(np.uint8)
    return img, ds

def save_as_ima(pixel_array, original_ds, save_path, verbose=False):
    if pixel_array.shape not in [(512, 512), (128, 128), (256, 256)]:
        raise ValueError(f"像素数组尺寸必须为512×512/256×256/128×128，当前：{pixel_array.shape}")
    if pixel_array.dtype != np.uint8:
        pixel_array = pixel_array.astype(np.uint8)
    ds = Dataset()
    file_meta = Dataset()
    if hasattr(original_ds, 'PatientID'):
        ds.PatientID = original_ds.PatientID
    if hasattr(original_ds, 'StudyID'):
        ds.StudyID = original_ds.StudyID
    if hasattr(original_ds, 'SeriesNumber'):
        ds.SeriesNumber = original_ds.SeriesNumber
    elif hasattr(original_ds, 'SeriesID'):
        ds.SeriesNumber = original_ds.SeriesID
    ds.Modality = original_ds.Modality if hasattr(original_ds, 'Modality') else 'CT'
    if hasattr(original_ds, 'Manufacturer'):
        ds.Manufacturer = original_ds.Manufacturer
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SliceThickness = 1.0
    now = datetime.datetime.now()
    ds.StudyDate = now.strftime('%Y%m%d')
    ds.StudyTime = now.strftime('%H%M%S.%f')[:-3]
    ds.AcquisitionDate = ds.StudyDate
    ds.AcquisitionTime = ds.StudyTime
    ds.RescaleSlope = 5.5556
    ds.RescaleIntercept = -1000.0
    if hasattr(original_ds, 'PixelSpacing'):
        pixel_spacing = original_ds.PixelSpacing
        if isinstance(pixel_spacing, (str, float, int)):
            pixel_spacing = [float(pixel_spacing)] * 2
        elif len(pixel_spacing) == 1:
            pixel_spacing = [pixel_spacing[0], pixel_spacing[0]]
        ds.PixelSpacing = [float(x) for x in pixel_spacing[:2]]
    else:
        ds.PixelSpacing = [0.5, 0.5]
    ds.WindowCenter = original_ds.WindowCenter if hasattr(original_ds, 'WindowCenter') else 0
    ds.WindowWidth = original_ds.WindowWidth if hasattr(original_ds, 'WindowWidth') else 400
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()
    file_meta.FileMetaInformationGroupLength = len(file_meta)
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    fd = FileDataset(
        save_path,
        ds,
        file_meta=FileMetaDataset(),
        preamble=b"\0" * 128
    )
    fd.save_as(save_path, write_like_original=False)
    if verbose:
        print(f"成功保存full‑1mm CT IMA文件：{save_path}")

# ===================== 验证函数 =====================
def validate_model(model, val_loader, device):
    """在验证集上计算平均 PSNR / SSIM（反归一化逻辑统一走 denormalize_to_01）"""
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            hr_pred = model(lr_imgs)
            hr_pred_np = denormalize_to_01(hr_pred)
            hr_imgs_np = denormalize_to_01(hr_imgs)
            for pred, gt in zip(hr_pred_np, hr_imgs_np):
                pred_gray = pred[0]
                gt_gray = gt[0]
                total_psnr += psnr(gt_gray, pred_gray, data_range=1.0)
                total_ssim += ssim(gt_gray, pred_gray, data_range=1.0)
                count += 1
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim
