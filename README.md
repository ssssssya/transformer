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

## 数据准备
你可以在官网或该链接https://huggingface.co/datasets/carlosejimenez/wikitext__wikitext-2-raw-v1中下载WikiText-2数据集



