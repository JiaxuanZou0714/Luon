# Luon: Low-Rank Muon Optimizer 📉
### Fusing Nuclear Norm Regularization into Muon for Scale-Invariant Transformers

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-blue)]()

> **TL;DR:** 在现代 LLM 架构（RMSNorm, QK-Norm）中，标准 L2 Weight Decay 因尺度不变性而无法有效控制模型复杂度。我们提出两种基于 Newton-Schulz 迭代的低秩正则化方法，并将其融合到 Muon 优化器中，形成 **Luon (Low-rank Muon)**，在 Grokking 任务上实现更快的泛化。

---

## 🔥 核心方法

本仓库实现了三种正则化策略的对比实验：

| 方法 | 优化器 | 低秩衰减方式 | 更新公式 |
|------|--------|-------------|----------|
| **L2 Baseline** | AdamW | Weight Decay | $W \leftarrow W - \lambda W$ |
| **Explicit LowRank** | AdamW + 回调 | 解耦的 Sign(W) 衰减 | $W \leftarrow W - \alpha \cdot \text{Sign}(W)$ |
| **Luon (Fused)** | Muon + AdamW | 融合到动量中 | $W \leftarrow W - \eta \cdot \text{NS}(\text{Momentum}(G + \lambda W))$ |

---

## 🧠 动机

### L2 Decay 在尺度不变网络中失效

现代 Transformer（如 Gemma 3, LLaMA）大量使用归一化层。在这些**尺度不变架构**中：

$$\text{Norm}(\alpha W x) = \text{Norm}(W x)$$

L2 Weight Decay 惩罚的是 $\|W\|_F^2$，优化器可以简单地缩小权重而不改变函数行为，无法真正降低复杂度（秩）。

### 解决方案：Nuclear Norm ($\|W\|_*$)

要在尺度不变网络中控制复杂度，必须控制**秩**而非幅度。核范数正则化：

$$\mathcal{L} = \mathcal{L}_{task} + \lambda \sum_i \sigma_i(W)$$

相当于对奇异值施加 L1 惩罚，促进**谱稀疏性**。

### 高效实现：Newton-Schulz 迭代

每步计算 SVD 太慢。我们使用 **Newton-Schulz 迭代** 逼近矩阵符号函数：

$$\text{Sign}(W) = W (W^T W)^{-1/2} \approx U V^T$$

这是核范数的次梯度，只需矩阵乘法即可高效计算。

---

## 🚀 快速开始

### 环境依赖
```bash
pip install torch numpy matplotlib tqdm seaborn
```

### 运行实验
```bash
python Luon.py --steps 3000 --device cuda
```

这将生成 `mechanism_analysis_final.png`，包含：
- Grokking 速度对比
- Effective Rank 演化
- 奇异值谱分布
- Attention Pattern 可视化

---

## 💻 核心算法

### 1. Newton-Schulz 迭代

```python
def newton_schulz_robust(M, steps=5, epsilon=1e-7):
    """计算矩阵符号函数: Sign(M) = M * (M^T * M)^(-1/2)"""
    M = M / (M.norm() + epsilon)  # 谱范数归一化

    for _ in range(steps):
        A = M @ M.T
        M = 0.5 * (3.0 * I - A) @ M

    return M
```

### 2. 显式低秩衰减（解耦方式）

```python
class NewtonSchulzLowRankDecay:
    def step(self):
        for W in self.params:
            sign_W = newton_schulz_robust(W.clone())
            W.sub_(self.decay_rate * sign_W)  # W <- W - α·Sign(W)
```

### 3. Luon：融合低秩衰减的 Muon

```python
class HybridLowRankMuon(Optimizer):
    def step(self):
        # 对目标参数 (Q, K) 使用 Muon + 融合低秩
        g_fused = grad + lambda * W           # 融合梯度
        momentum.mul_(mu).add_(g_fused)       # 动量更新
        update = newton_schulz(momentum)      # 正交化
        W.sub_(lr * update)                   # 参数更新

        # 其他参数使用标准 AdamW
```

---

## 🧪 实验设置

| 配置 | 值 |
|------|-----|
| **任务** | 模加法 $a + b \pmod{113}$ |
| **架构** | 2层 Transformer (dim=128, heads=4) |
| **归一化** | Pre-RMSNorm + QK-Norm |
| **数据划分** | 50% 训练 / 50% 验证 |

---

## 📊 评估指标

- **Grokking Speed**: 验证集准确率达到 99% 所需的步数
- **Stable Rank**: $\|W\|_F^2 / \|W\|_2^2$，衡量有效秩
- **Singular Value Spectrum**: QK 权重矩阵的奇异值分布

---

## 📁 项目结构

```
Luon/
├── Luon.py          # 主实验代码（自包含）
├── README.md        # 本文件
└── mechanism_analysis_final.png  # 生成的分析图
```

---

## 📝 引用

```bibtex
@misc{luon2025,
  title={Luon: Low-Rank Muon Optimizer for Scale-Invariant Transformers},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/JiaxuanZou0714/Luon}}
}
```

---

## 🔗 相关工作

- [Muon Optimizer](https://github.com/KellerJordan/Muon) - 原始 Muon 实现
- [Grokking](https://arxiv.org/abs/2201.02177) - Grokking 现象研究

---

*This is a research prototype.*
