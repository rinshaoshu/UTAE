from models.Unet3d import UNet3D
from models.QMF import QMF
from models.MCANet import MACANet3DWrapper
def get_model():
    """获取 QMF 模型"""
    model = UNet3D(num_classes=2,in_channels=17)
    return model
