# 学習ループ スニペット集

コードの引き出しであり、うまく学習ができないときにチェックする。

## Label Smoothing (過学習防止 / 確信度の緩和)
CrossEntropyの引数に追加。ノイズの多いデータに有効。

```python
# label_smoothing=0.1 を追加
loss = F.cross_entropy(y_pred, y, label_smoothing=0.1)
```

## 学習率スケジューラの使用

```python
# Optimizerの定義後に宣言
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 各エポックの最後に実行
for epoch in range(epochs):
    # ... train / val loop ...
    scheduler.step()
```
