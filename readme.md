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

## 12.29

TODO：
- 去噪声
- 修改网络

## 12.30-31
- 去噪声
  - 图像每帧做滤波后提取特征
  - 音频用谱减法预处理后提取特征
- 修改网络结构
  - 新增`test_train.py`文件，将训练集的15%划分出来做测试，效果不理想，top5准确率只有60%。

## 1.2
### 炼丹记录

- Prob两层+助教给定特征+不做数据增强+LSTM输出接Afeat+layers_num=3, dropout=.1

  ```
  Test checkpoint epoch 80.
  Top1 accuracy for sample 50 is: 0.2922.
  Top5 accuracy for sample 50 is: 0.72112.
  Test checkpoint epoch 90.
  Top1 accuracy for sample 50 is: 0.28732.
  Top5 accuracy for sample 50 is: 0.71428.
  Test checkpoint epoch 100.
  Top1 accuracy for sample 50 is: 0.30147999999999997.
  Top5 accuracy for sample 50 is: 0.72772.
  Test checkpoint epoch 110.
  Top1 accuracy for sample 50 is: 0.27408.
  Top5 accuracy for sample 50 is: 0.70096.
  Test checkpoint epoch 120.
  Top1 accuracy for sample 50 is: 0.29908.
  Top5 accuracy for sample 50 is: 0.73584.
  ```

- Prob多层+畅畅提取特征+不做数据增强+LSTM输出接Afeat+layers_num=3, dropout=.1
  ```
  Test checkpoint epoch 80.
  Top1 accuracy for sample 50 is: 0.26404.
  Top5 accuracy for sample 50 is: 0.71052.
  Test checkpoint epoch 100.
  Top1 accuracy for sample 50 is: 0.28056000000000003.
  Top5 accuracy for sample 50 is: 0.72456.
  ```
### 特征说明

- `Train/afeat&vfeat` 特征替换为畅畅提取的特征
- `Test/Clean/vfeat` 特征替换为畅畅提取的特征
- `Test/Noise/vfeat` 特征替换为畅畅提取的特征
- `Test/Denoise/vfeat` 为畅畅提取的特征，似乎还没有提取音频去噪的特征。

## 1.3

## 完成clean测试
（结果被ignore掉了）以下是训练+测试结果
```
Test checkpoint epoch 100.
Top1 accuracy for sample 50 is: 0.26892.
Top5 accuracy for sample 50 is: 0.77008.
```

## TODO：Noise测试

- 训练用的特征存到`Train/{}feat_denoise`中，{}是a或v；
- `train_test.py`用于一边训练一边测试，为了尽可能多得训练，只分2%的训练集出来测试（将将够50个）。
  - 参数存在`checkpoints/noise`里；
  - 训好后打印若干轮的结果，从中选最好的即可。
- 最后用最好的参数运行`test.py`