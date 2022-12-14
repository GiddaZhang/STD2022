## 12.14

```
Test checkpoint epoch 150.
Top1 accuracy for sample 50 is: 0.51376.
Top5 accuracy for sample 50 is: 0.88424
```

- top5~88%
- 取训练集的0~499为验证集，在`dataset.py`文件里加了一句判断，如果`index<500`就让`index+=500`避免用验证集训练，更好的办法还没想到；
- 网络结构粗暴地加了一些层数；
- 提高了训练轮数。

