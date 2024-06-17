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

 在Naive Model中，我將所有訓練資料用3通道的形式去訓練Resnet34，只有在測試時才處理動態通道的問題。

 我處理動態通道的方法為Padding，也就是將缺失的通道數用0補足。

 EX：如果該圖片的通道為RG，我就將B通道補0。

2. Dynamic Convolution Module : Dy_CNN + Resnet34

```bash
.dy_cnn_train.sh
```

* Inference

```bash
python inference.py --timestamp 06_08_16 --model_name dy_cnn
```

 在Dynamic Convolution Module中，我沿用Naive Model的Resnet34當作backbone網路。不過我在前面加入一層Dynamic Convolution，將通道數不為3的圖片轉換成3通道形式。
 在訓練時我固定Naive Model的Resnet34的參數不訓練，只訓練前面的一層Dynamic Convolution。

結果如下

* Accuarcy

|        | RGB | RG | RB | GB | R  | G  | B  |
|:------:|:---:|:--:|:--:|:--:|:-: |:-: |:-: |
|  Naive |0.68 |0.24|0.05|0.04|0.02|0.04|0.02|
| Dy_CNN |0.64 |0.22|0.56|0.19|0.48|0.45|0.45|

* Parmater

在Resnet34中有2130萬個參數量，但因為我們只有訓練一層Dynamic Convolution，所以只需要儲存一次Resnet34以及7種不同的Head，三通道的Head參數量為192個，兩通道的Head參數量為90 $\times$ 3， 一通道的Head參數量為65 $\times$ 3 ，需要額外存的參數量為657個。

---

## Task 2 : Designing a Two-Layer Network for Image Classification

---

* Training Custom Model

```bash
python training --model_name custom
```

我模型中總共使用四層CNN，前面兩層為Dynamic Convolution搭配MaxPooling，後面兩層為正常Convolution搭配Residual Block使用，最後再接一層Self-attention。

---

### TODO

先訓練一個resnet34(naive)的基礎模型，在用prefix finetuning的方式訓不同head的Dy_cnn（7個）

* [X] Task 1
  * [x] training&inference  bcakbone Resnet34

  * [x] training mutiple dy_cnn head

  * [X] inference
    * [X] 在inference中實在testing naive model
    * [X] 整合naive test and dy_cnn test
    * [X] 將原本用shell的執行的naive inference寫入 <code>inference.py</code>
    * [X] 測試整合
  * [x] plot

* [x] Task 2
    * [x] construct model
    * [x] training model
    * [x] plot