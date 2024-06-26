# Protein MSA Fold project
## 运行环境
```shell
conda create -n MSA python=3.8.18
pip install -r requirements.txt
```

## 修改记录
将msadata.py根据Protein的部分进行了修改

运行流程：seq(batch_size, seq_length) -> ESM(batch_size, seq_length, 1280) -> repeat * num_alignment(batch_size, num_alignment, seq_length, 1280) <-> label(batch_size, num_alignment, seq_length)

除此之外，在model部分针对ESM优化了Embedding部分，即将nn.Embedding换成了nn.Linear以映射到hidden_dim

## 6.26更新
在finetune.py加入了threshold以控制最大Protein Seq长度，可以在finetune.sh的--threshold参量进行调整

在msadata.py中加入了进度条以显示数据处理进度，并结合threshold控制数据长度