"""
check.py — 统一输入 (B=2, T=1, C=15, H=128, W=128) 检查所有模型 I/O
预期输出: (2, 1, 128, 128)
"""
import torch
import torch.nn as nn


def check_model(name, model, input_tensor):
    """检查单个模型的输入输出。"""
    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"  输入形状: {tuple(input_tensor.shape)}")
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        print(f"  输出形状: {tuple(output.shape)}  ✓")
        # 额外检查
        B, C, H, W = 2, 1, 128, 128
        assert output.shape[0] == B, f"batch 不匹配: {output.shape[0]} != {B}"
        assert output.shape[1] == C, f"channel 不匹配: {output.shape[1]} != {C}"
        assert output.shape[2] == H, f"H 不匹配: {output.shape[2]} != {H}"
        assert output.shape[3] == W, f"W 不匹配: {output.shape[3]} != {W}"
        print(f"  ✓ 形状完全匹配 (B={B}, C={C}, H={H}, W={W})")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"统一输入形状: (2, 1, 15, 128, 128)  [B, T, C, H, W]")
    print(f"预期输出形状: (2, 1, 128, 128)     [B, num_classes, H, W]")

    # 统一输入: [B, T, C, H, W] = [2, 1, 15, 128, 128]
    x = torch.randn(2, 10, 15, 128, 128).to(device)

    results = []

    # ---- UTAE ----
    from models.utae import UTAE
    model = UTAE(in_channels=15, num_classes=1).to(device)
    results.append(check_model('UTAE', model, x))

    # ---- SwinUNetHeadWithTemporal ----
    from models.swinutae import SwinUNetHeadWithTemporal
    model = SwinUNetHeadWithTemporal(
        img_size=128, in_channels=15, num_classes=1
    ).to(device)
    results.append(check_model('SwinUTAE', model, x))

    # ---- ConvGRU_Seg ----
    from models.convgru import ConvGRU_Seg
    model = ConvGRU_Seg(
        num_classes=1, img_res=128, in_channels=15,
        kernel_size=(3, 3), hidden_dim=16
    ).to(device)
    results.append(check_model('ConvGRU', model, x))

    # ---- UNet3D ----
    from models.Unet3d import UNet3D
    model = UNet3D(in_channels=15, num_classes=1, img_res=128).to(device)
    results.append(check_model('UNet3D', model, x))

    # ---- CMXSeg (from CMXSegTemporal) ----
    from models.CMXSegTemporal import CMXSeg
    model = CMXSeg(num_classes=1, img_size=128).to(device)
    results.append(check_model('CMXSeg', model, x))

    # ---- CMNextSeg ----
    from models.CMNextSegTemporal import CMNextSeg
    model = CMNextSeg(num_classes=1, img_size=128).to(device)
    results.append(check_model('CMNextSeg', model, x))

    # ---- ESASeg (class名也是 CMXSeg，用别名) ----
    from models.ESASegTemporal import CMXSeg as ESASeg
    model = ESASeg(num_classes=1, img_size=128).to(device)
    results.append(check_model('ESASeg', model, x))

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("检查汇总:")
    passed = sum(results)
    total = len(results)
    for i, (name, ok) in enumerate(zip(
        ['UTAE', 'SwinUTAE', 'ConvGRU', 'UNet3D', 'CMXSeg', 'CMNextSeg', 'ESASeg'],
        results
    )):
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    print(f"\n通过: {passed}/{total}")


if __name__ == '__main__':
    main()
