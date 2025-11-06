## 项目简介

本项目完全使用 PyTorch 实现了一个 **Transformer 编码器 (Encoder-only)** 模型，支持以下特性：

- 模块化实现：自注意力、多头注意力、前馈层、层归一化、残差连接  
- 位置编码（可关闭）、残差连接（可关闭）、LayerNorm（可关闭）  
- Masked LM 预训练任务  
- 自动加载 Parquet 格式数据集  
- 多组消融实验配置  
- 绘制 Loss 与 Perplexity 曲线  

该模型可以直接在 WikiText-2 或自定义文本数据上进行训练与验证。


## 环境配置

建议使用 Conda 创建独立环境（Python ≥ 3.10，CUDA ≥ 11.7）：

```bash
conda create -n transformer python=3.10 -y
conda activate transformer
pip install torch datasets matplotlib tqdm

```
## 数据准备
你可以在官网或该链接https://huggingface.co/datasets/carlosejimenez/wikitext__wikitext-2-raw-v1 中下载WikiText-2数据集

## 运行命令
python -m scripts.train_encoder
程序将自动进行以下工作：
- 读取 Parquet 格式数据集
- 构建词表并编码文本
- 在多组消融配置上进行训练
- 保存模型与曲线到 ./results/ 目录

## 实验环境与硬件要求
| 组件      | 配置                     |
| ------- | ---------------------- |
| GPU     | NVIDIA RTX 3090 (24GB) |
| CPU     | AMD Ryzen 7 / Intel i7 |
| CUDA    | 11.7+                  |
| PyTorch | 2.3.0                  |
| Python  | 3.10                   |
| 运行时长    | 约 0.5 小时 / 10 epochs   |
| 显存需求    | ≥ 12GB 推荐              |



