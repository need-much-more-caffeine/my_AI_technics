class EEGNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        F1: int = 8,        # 時間フィルタの数
        D: int = 2,         # 空間フィルタの深さ乗数
        F2: int = 16,       # Pointwiseフィルタの数
        kernel_length: int = 32, # 時間フィルタの長さ
        p_drop: float = 0.25
    ) -> None:
        super().__init__()

        # --- 1. 時間的特徴抽出 (Temporal Convolution) ---
        # input: (B, 1, Channels, Time)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            # --- 2. 空間的特徴抽出 (Depthwise Spatial Convolution) ---
            # 電極間(in_channels)を縦に一気に畳み込む
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p_drop)
        )

        # --- 3. 時間と空間統合 (Separable Convolution) ---
        self.block2 = nn.Sequential(
            # Depthwise (各チャネルごとに時間の畳み込み)
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            # Pointwise (チャネル間の特徴を混ぜる)
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # Time=100の場合、2回のプーリング
            nn.Dropout(p_drop)
        )

        # 特徴量の次元数を自動計算、全結合層へ接続（次元エラー防止工夫）
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, in_channels, seq_len)
            dummy_out = self.block2(self.block1(dummy_input))
            self.feature_dim = dummy_out.numel()

        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, Channels, Time) -> (B, 17, 100)
        """
        # 1D信号を無理やり2D画像(1チャンネル)の形に変換する
        # (B, 17, 100) -> (B, 1, 17, 100)
        X = X.unsqueeze(1)

        X = self.block1(X)
        X = self.block2(X)

        X = X.flatten(start_dim=1)
        return self.head(X)
