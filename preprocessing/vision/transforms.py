from torchvision import transforms

def get_base_transforms(size: int = 224):
    """画像を指定サイズに整える基本処理"""
    return [
        transforms.Resize(size),
        transforms.CenterCrop(size)
    ]

def get_tensor_and_normalize(mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)):
    """PyTorchで計算するための必須処理（最後に行う）"""
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
