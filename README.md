# Deep Learning - Assignment 2

---
You can download dataset on  https://cchsu.info/files/images.zip 


## Task 1 : Designing a Convolution Module for Variable Input Channels

Compare  the  performance  of the  model  using the dynamic convolution module with naive models across different input channel combinations, evaluating accuracy and computational cost.

### Method

1. Naive Model : Resnet34

* Training

```bash
python training.py --model_name naive --use_channel RGB
```

* Inference

```bash
python inference.py --timestamp 06_08_16 --model_name naive
```

> 在Naive Model中，我將所有訓練資料用3通道的形式去訓練Resnet34，只有在測試時才處理動態通道的問題。
>
> 我處理動態通道的方法為Padding，也就是將缺失的通道數用0補足。
>
> EX：如果該圖片的通道為RG，我就將B通道補0。

2. Dynamic Convolution Module : Dy_CNN + Resnet34

```bash
./shell_script/dy_cnn_train.sh
```

* Inference

```bash
python inference.py --timestamp 06_08_16 --model_name dy_cnn
```

> 在Dynamic Convolution Module中，我沿用Naive Model的Resnet34當作backbone網路。不過我在前面加入一層Dynamic Convolution，將通道數不為3的圖片轉換成3通道形式。
> 在訓練時我固定Naive Model的Resnet34的參數不訓練，只訓練前面的一層Dynamic Convolution。

結果如下

* Accuarcy

|        | RGB | RG | RB | GB | R  | G  | B  |
|:------:|:---:|:--:|:--:|:--:|:-: |:-: |:-: |
|  Naive |0.68 |0.24|0.05|0.04|0.02|0.04|0.02|
| Dy_CNN |0.64 |0.22|0.56|0.19|0.48|0.45|0.45|

* Parmater

---

## Task 2 : Designing a Two-Layer Network for Image Classification

---

### Tips

1. training and testing phase 要能自適應 image size and channel
最簡單的方式是做data preprocess (全部resize成256x256以及channel設為3)
當然老師希望(期盼)同學可以設計一個新的Conv Module可以達到dynamically去吃input
最後的實驗流程由同學自己從testing set中去竄改RGB三通道

2. 與第一題無關，這題需設計模型(僅能使用兩層神經網路)
Fully connected 也算一層
目的在於僅設計2層的模型來達到90%的ResNet34效能
老師有說明不要拿網路上pretrained好的ResNet34
因為他一定很強，我們要自己設計實驗流程
固定epoch, optimizer, loss的情況
自己的模型與ResNet34的差異

---

### TODO

先訓練一個resnet34(naive)的基礎模型，在用prefix finetuning的方式訓不同head的Dy_cnn（7個）

* [ ] Task 1
  * [x] training&inference  bcakbone Resnet34

  * [x] training mutiple dy_cnn head

  * [X] inference

    * [X] 在inference中實在testing naive model
    * [X] 整合naive test and dy_cnn test
    * [X] 將原本用shell的執行的naive inference寫入inference.py
    * [X] 測試整合

  * [ ] plot

* [ ]Task 2
  
  * [ ]
