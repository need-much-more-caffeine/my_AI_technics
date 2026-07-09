from torchvision import transforms

def get_hard_augmentations(size: int = 224):
    # 実戦向けのスパルタなデータ水増し手法
    return [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
    ]
