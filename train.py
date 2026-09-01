import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from model import SwinIRMed, SwinIRMedLoss
from utils import read_ima_image, validate_model, build_swinir_med, get_default_transform

# ===================== 数据集 =====================
class SRDataset(data.Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.img_names = sorted([f for f in os.listdir(lr_dir)
                                 if f.lower().endswith(('ima', 'dcm', 'png', 'jpg', 'jpeg'))])
        assert len(self.img_names) > 0, f"LR文件夹 {lr_dir} 中无有效医学影像！"
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)
        lr_img, _ = read_ima_image(lr_path)
        hr_img, _ = read_ima_image(hr_path)
        if lr_img.shape[:2] != (128, 128):
            raise ValueError(f"LR影像 {img_name} 尺寸错误，需为128×128，当前: {lr_img.shape[:2]}")
        if hr_img.shape[:2] != (512, 512):
            raise ValueError(f"HR影像 {img_name} 尺寸错误，需为512×512，当前: {hr_img.shape[:2]}")
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        return lr_img, hr_img

# ===================== 训练入口函数 =====================
def train_model(lr_dir, hr_dir, save_model_path, csv_save_path,
                batch_size=4, epochs=100, learning_rate=1e-4, upscale_factor=4, val_ratio=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"使用设备: {device} | 训练数据：full‑1mm CT影像 | 模型：SwinIR‑Med | 模式：混合精度训练 (AMP) | 超分倍数：4x (128→512)")
    transform = get_default_transform()
    full_dataset = SRDataset(lr_dir, hr_dir, transform)
    n_val = int(len(full_dataset) * val_ratio)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_swinir_med(upscale_factor, device)
    loss_fn = SwinIRMedLoss(
        use_edge_loss=True,
        edge_weight=0.01,
        use_tv_loss=True,
        tv_weight=0.08,
        use_smooth_loss=True,
        smooth_weight=0.04
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scaler = GradScaler('cuda')
    start_epoch = 0
    best_psnr = 0.0
    best_model_path = save_model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        best_psnr = best_checkpoint.get('val_psnr', 0.0)
        print(f"加载最佳模型基准 PSNR: {best_psnr:.2f} dB")
    if os.path.exists(save_model_path):
        checkpoint = torch.load(save_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("已加载混合精度缩放器状态")
        print(f"加载已有模型，从第 {start_epoch} 轮继续训练")
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    csv_header = ['epoch', 'train_loss', 'val_psnr', 'val_ssim']
    if not os.path.exists(csv_save_path):
        with open(csv_save_path, 'w', newline='', encoding='utf‑8') as f:
            csv.writer(f).writerow(csv_header)
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (AMP)")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                hr_pred = model(lr_imgs)
                total_loss = loss_fn(hr_pred, hr_imgs)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()
            pbar.set_postfix({'total_loss': total_loss.item()})
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        val_psnr, val_ssim = validate_model(model, val_loader, device)
        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
        with open(csv_save_path, 'a', newline='', encoding='utf‑8') as f:
            csv.writer(f).writerow([epoch + 1, avg_train_loss, val_psnr, val_ssim])
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr
            }, best_model_path)
            print(f"新最佳模型已保存 (PSNR: {val_psnr:.2f} dB)")
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_psnr': val_psnr
            }, save_model_path)
            print(f"检查点模型已保存至: {save_model_path}")

if __name__ == "__main__":
    LR_DIR = "dataset/128x128"
    HR_DIR = "dataset/512x512"
    SAVE_MODEL_PATH = "result_swinir/swinir_med_4x_medical_8/swinir_med_4x_sr_amp.pth"
    CSV_SAVE_PATH = "result_swinir/swinir_med_4x_medical_8/training_metrics_amp.csv"
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    train_model(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        save_model_path=SAVE_MODEL_PATH,
        csv_save_path=CSV_SAVE_PATH,
        batch_size=2,
        epochs=2,
        learning_rate=1e-4,
        upscale_factor=4,
        val_ratio=0.1
    )
