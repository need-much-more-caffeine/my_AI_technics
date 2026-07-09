from torchvision.transforms import functional as F
from PIL import Image

class LetterboxPad:
    # 縦横比を崩さずにリサイズし、余白をパディングするカスタム変換
    def __init__(self, size: int = 224, fill: int = 0):
        self.size = size
        self.fill = fill
        
    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
