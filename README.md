# 通过Seq2Seq完成一个文本摘要模型

> 数据集来源：
>
> 百度AI 《常规赛：问答摘要与推理》   <https://aistudio.baidu.com/aistudio/competition/detail/3>

## 数据处理阶段

1. 用jieba进行切词

2. 用gensim训练词向量

3. 填充字符串 

   > <start> <stop> <pad> <unk>

## Encoder&Decoder Layers

> [model_layers.py](https://github.com/Abner1zhou/Seq2Seq_tf2/blob/master/model_layers.py)

1. embedding都采用预先训练好的词向量，直接传入



# How to Run

下载数据：<https://aistudio.baidu.com/aistudio/datasetdetail/1407>

数据存放目录   data/AutoMaster_TrainSet

- 数据预处理

```bash
python utils/data_processing.py
```

[data_processing.py](https://github.com/Abner1zhou/Seq2Seq_tf2/blob/master/utils/data_processing.py)

- 调整参数

[params_utils.py](<https://github.com/Abner1zhou/Seq2Seq_tf2/blob/master/utils/params_utils.py>)

- 开始训练

```bash
python train.py
```

[train.py](<https://github.com/Abner1zhou/Seq2Seq_tf2/blob/master/train.py>)

