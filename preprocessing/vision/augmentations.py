from torchvision import transforms

def get_train_augmentations(size: int = 224):
    """
    学習用のデータ水増し手法のストック
    今後、色調変更(ColorJitter)や回転(RandomRotation)などをここに追加する
    """
    return [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2), # ←将来の追加例
    ]
