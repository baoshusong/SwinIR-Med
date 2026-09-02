import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

from model import SwinIRMed
from utils import (read_ima_image, save_as_ima, build_swinir_med,
                   denormalize_to_01, build_25d_test_input)

def test_batch_images(model_path, input_folder, output_folder, upscale_factor=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_swinir_med(upscale_factor, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    img_names = sorted([f for f in os.listdir(input_folder)
                         if f.lower().endswith(('ima', 'dcm'))])
    assert len(img_names) > 0, "测试文件夹中无有效IMA/DICOM影像！"
    os.makedirs(output_folder, exist_ok=True)
    saved_count = 0
    with torch.no_grad():
        for idx in tqdm(range(len(img_names)), desc="SwinIR‑Med CT 2.5D 4x超分推理 (128→512)"):
            try:
                lr_tensor, original_ds, img_name = build_25d_test_input(input_folder, img_names, idx, device)
            except Exception as e:
                print(f"警告：无法读取 {img_names[idx]}，错误：{e}，跳过")
                continue
            hr_pred = model(lr_tensor)
            hr_pred_np = denormalize_to_01(hr_pred.squeeze(0))
            hr_pred_gray = (hr_pred_np[0] * 255).astype(np.uint8)
            save_name = os.path.splitext(img_name)[0] + '_4x_SR.ima'
            save_path = os.path.join(output_folder, save_name)
            save_as_ima(hr_pred_gray, original_ds, save_path)
            saved_count += 1
    print(f"\nSwinIR‑Med CT影像2.5D 4x超分完成！共生成 {saved_count} 张，结果保存至: {output_folder}")
    print(f"输出格式：IMA（DICOM）| 层厚：1mm | 尺寸：512×512 | HU值范围：‑1000~400")

if __name__ == "__main__":
    TEST_BATCH_INPUT = "dataset/test_4x"
    TEST_BATCH_OUTPUT = "result_swinir/swinir_med_4x_medical_12/test_output_amp"
    BEST_MODEL = "result_swinir/swinir_med_4x_medical_12/swinir_med_4x_sr_amp_best.pth"
    os.makedirs(TEST_BATCH_OUTPUT, exist_ok=True)
    test_batch_images(
        model_path=BEST_MODEL,
        input_folder=TEST_BATCH_INPUT,
        output_folder=TEST_BATCH_OUTPUT,
        upscale_factor=4
    )
