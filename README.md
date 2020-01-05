# 通过Seq2Seq完成一个文本摘要模型

> 数据集来源：
>
> 百度AI 《常规赛：问答摘要与推理》   <https://aistudio.baidu.com/aistudio/competition/detail/3>

## 数据处理阶段

1. 用jieba进行切词

2. 用gensim训练词向量

3. 填充字符串 

   > <start> <stop> <pad> <unk>

