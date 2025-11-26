import math
import argparse
import random
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import seaborn as sns

sns.set_theme(style="whitegrid")

# ==============================================================================
# 0. Global Setup
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. Newton-Schulz Implementation
# ==============================================================================
def newton_schulz_robust(M, steps=5, epsilon=1e-7):
    """
    Computes the Matrix Sign Function: Sign(M) = M * (M^T * M)^(-1/2)
    Using Newton-Schulz iteration.
    """
    orig_dtype = M.dtype
    M = M.float()

    # 1. Spectral Norm Estimation & Normalization
    f_norm = M.norm()
    M.div_(f_norm + epsilon)

    r, c = M.shape
    if r > c:
        M = M.T
        r, c = c, r
        transposed = True
    else:
        transposed = False

    I = torch.eye(r, device=M.device, dtype=M.dtype)

    for _ in range(steps):
        A = M @ M.T # (r, r)
        M = 0.5 * (3.0 * I - A) @ M

    if transposed:
        M = M.T

    return M.to(orig_dtype)

# ==============================================================================
# 2. Experiment 2 Logic: Explicit LowRank Callback（栀的方案）
# ==============================================================================
class NewtonSchulzLowRankDecay:
    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords
        self.params_to_decay = []

        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if self.target_keywords and not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append(param)

    @torch.no_grad()
    def step(self):
        for W in self.params_to_decay:
            # Explicitly compute Sign(W) independent of gradients
            X = W.clone()
            sign_W = newton_schulz_robust(X, steps=self.num_iterations)

            # Update: W <- W - rate * Sign(W)
            W.sub_(self.decay_rate * sign_W)

# ==============================================================================
# 3. Experiment 3 Logic: Strict LowRank Muon Optimizer（基于栀的方法改进的Muon）
# ==============================================================================
class HybridLowRankMuon(optim.Optimizer):
    """
    Hybrid Optimizer:
    - Target Params (Q, K): Muon with fused LowRank objective.
      Update = - lr * NS( Momentum( Grad + lambda * W ) )
    - Other Params: Standard AdamW.
    """
    def __init__(self, params, target_keywords=None,
                 muon_lr=0.02, muon_wd=0.0, muon_momentum=0.95,
                 adam_lr=1e-3, adam_wd=0.0):

        target_keywords = target_keywords or ["q_proj", "k_proj"]
        muon_params = []
        adam_params = []

        for name, p in params:
            if not p.requires_grad:
                continue
            # Logic: If param is 2D AND matches target, use Muon
            if p.ndim == 2 and any(k in name for k in target_keywords):
                muon_params.append(p)
            else:
                adam_params.append(p)

        defaults = dict(muon_lr=muon_lr, muon_wd=muon_wd, muon_momentum=muon_momentum,
                        adam_lr=adam_lr, adam_wd=adam_wd)

        super().__init__([{'params': muon_params, 'type': 'muon'},
                          {'params': adam_params, 'type': 'adam'}], defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['type'] == 'muon':
                # === Experiment 3 Core Logic ===
                lr = group['muon_lr']
                wd = group['muon_wd'] # This is 'lambda'
                mu = group['muon_momentum']

                for p in group['params']:
                    if p.grad is None: continue
                    g = p.grad

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']

                    # -----------------------------------------------------------
                    # Strict Implementation of: W_{t+1} = W_t - eta * Op(G + lambda*W)
                    # -----------------------------------------------------------

                    # 1. Construct the Fused Gradient: G_tilde = G + lambda * W
                    g_fused = g + wd * p

                    # 2. Update Momentum: M_{t+1} = mu * M_t + G_tilde
                    buf.mul_(mu).add_(g_fused)

                    # 3. Apply Operator: U = NewtonSchulz(M_{t+1})
                    # Note: We clone buf to avoid modifying the momentum buffer during NS
                    update_direction = newton_schulz_robust(buf.clone(), steps=5)

                    # 4. Apply Update: W_{t+1} = W_t - lr * U
                    p.sub_(lr * update_direction)

            elif group['type'] == 'adam':
                # === Standard AdamW (Simplified) ===
                lr = group['adam_lr']
                wd = group['adam_wd']
                beta1, beta2 = 0.9, 0.999
                eps = 1e-8

                for p in group['params']:
                    if p.grad is None: continue
                    g = p.grad
                    if wd > 0: p.mul_(1 - lr * wd) # Decoupled WD

                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_cor1 = 1 - beta1 ** state['step']
                    bias_cor2 = 1 - beta2 ** state['step']
                    step_size = lr * math.sqrt(bias_cor2) / bias_cor1

                    p.addcdiv_(exp_avg, denom, value=-step_size)

# ==============================================================================
# 4. Model & Data
# ==============================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 114
    dim: int = 128
    depth: int = 2
    heads: int = 4
    mlp_dim: int = 512
    use_qk_norm: bool = True

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.head_dim = config.dim // config.heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

        # --- FIX: Save use_qk_norm to self ---
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(config.dim)
            self.k_norm = RMSNorm(config.dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.o_proj((attn @ v).transpose(1, 2).reshape(B, T, C))

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Save config for later access
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, config.dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': RMSNorm(config.dim),
                'attn': Attention(config),
                'norm2': RMSNorm(config.dim),
                'mlp': nn.Sequential(
                    nn.Linear(config.dim, config.mlp_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(config.mlp_dim, config.dim, bias=False)
                )
            }) for _ in range(config.depth)
        ])
        self.norm_final = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :T, :]
        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['mlp'](layer['norm2'](x))
        x = self.norm_final(x)
        return self.lm_head(x[:, -1, :])

class ModularAdditionDataset(Dataset):
    def __init__(self, p=113, split='train', train_frac=0.5, seed=42):
        data = [(i, j, p, (i + j) % p) for i in range(p) for j in range(p)]
        random.seed(seed)
        random.shuffle(data)
        split_idx = int(len(data) * train_frac)
        self.data = data[:split_idx] if split == 'train' else data[split_idx:]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        i, j, eq, res = self.data[idx]
        return torch.tensor([i, j, eq], dtype=torch.long), torch.tensor(res, dtype=torch.long)

# ==============================================================================
# 5. Training Loop
# ==============================================================================

def get_stable_rank(model):
    ranks = []
    for name, param in model.named_parameters():
        if "q_proj" in name or "k_proj" in name:
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            # Stable Rank = ||W||_F^2 / ||W||_2^2
            ranks.append((S.norm()**2 / (S[0]**2 + 1e-9)).item())
    return sum(ranks) / len(ranks) if ranks else 0

def train_run(args, decay_type, decay_val, device):
    p = 113
    train_frac = 0.5
    config = ModelConfig(vocab_size=p+1, use_qk_norm=True)
    model = Transformer(config).to(device)

    lrd_opt = None

    # 1. Experiment 1: Baseline L2
    if decay_type == 'L2':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay_val)

    # 2. Experiment 2: Explicit LowRank
    elif decay_type == 'LowRank':
        # AdamW with NO weight decay for structural params (handled by lrd_opt)
        decay_params, nodecay_params = [], []
        target = ["q_proj", "k_proj"]
        for name, p_val in model.named_parameters():
            if any(t in name for t in target): nodecay_params.append(p_val)
            else: decay_params.append(p_val)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ], lr=1e-3)
        # Apply strict W <- W - alpha * Sign(W)
        lrd_opt = NewtonSchulzLowRankDecay(model.named_parameters(), decay_rate=decay_val, target_keywords=target)

    # 3. Experiment 3: LowRank Muon (Fused)
    elif decay_type == 'LowRankMuon':
        optimizer = HybridLowRankMuon(
            model.named_parameters(),
            target_keywords=["q_proj", "k_proj"],
            muon_lr=0.02,
            muon_wd=decay_val,   # lambda passed here
            adam_lr=1e-3,
            adam_wd=0.0
        )

    # Data
    train_ds = ModularAdditionDataset(p=p, split='train', train_frac=train_frac)
    val_ds = ModularAdditionDataset(p=p, split='val', train_frac=train_frac)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)

    history = {'step': [], 'val_acc': [], 'rank': []}

    pbar = tqdm(range(args.steps), desc=f"{decay_type}", leave=False)
    iter_loader = iter(train_loader)

    for step in pbar:
        try: x, y = next(iter_loader)
        except: iter_loader = iter(train_loader); x, y = next(iter_loader)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrd_opt: lrd_opt.step()

        if step % 50 == 0:
            model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    corr += (model(vx).argmax(-1) == vy).sum().item()
                    tot += vy.size(0)
            val_acc = corr / tot
            model.train()

            rank = get_stable_rank(model)

            history['step'].append(step)
            history['val_acc'].append(val_acc)
            history['rank'].append(rank)
            pbar.set_postfix({'acc': f"{val_acc:.2f}", 'rank': f"{rank:.2f}"})

    return history, model

# ==============================================================================
# 6. Analysis Execution
# ==============================================================================
def run_mechanism_analysis(args):
    print(">>> Starting Comparison Analysis...")

    # --- 1. L2 Baseline ---
    print("1. Baseline (L2)...")
    hist_l2, model_l2 = train_run(args, 'L2', 0.1, args.device)

    # --- 2. Explicit LowRank ---
    # rate = 0.005 is subtracted DIRECTLY from weights.
    print("2. Explicit LowRank...")
    hist_lr, model_lr = train_run(args, 'LowRank', 0.01, args.device)

    # --- 3. LowRank Muon (Fused) ---
    # Using high lambda to ensure NS(G + lambda*W) feels the W.
    print("3. LowRank Muon (Fused)...")
    hist_muon, model_muon = train_run(args, 'LowRankMuon', 0.1, args.device)

    # --- Visualization ---
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4)

    # Plot Acc
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hist_l2['step'], hist_l2['val_acc'], label='L2', alpha=0.5)
    ax1.plot(hist_lr['step'], hist_lr['val_acc'], label='Explicit', linewidth=2)
    ax1.plot(hist_muon['step'], hist_muon['val_acc'], label='Muon (Fused)', linewidth=2, linestyle='-.', color='green')
    ax1.set_title("Grokking Speed")
    ax1.legend()

    # Plot Rank
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hist_l2['step'], hist_l2['rank'], label='L2', linestyle='--', alpha=0.5)
    ax2.plot(hist_lr['step'], hist_lr['rank'], label='Explicit', linewidth=2)
    ax2.plot(hist_muon['step'], hist_muon['rank'], label='Muon', linewidth=2, linestyle='-.', color='green')
    ax2.set_title("Effective Rank")
    ax2.legend()

    # Plot Spectrum
    ax3 = fig.add_subplot(gs[0, 2:])
    def get_svd(model):
        W = model.layers[0]['attn'].q_proj.weight.detach()
        S = torch.linalg.svdvals(W.float()).cpu().numpy()
        return S / S[0]

    ax3.plot(get_svd(model_l2), label='L2', marker='.', alpha=0.3)
    ax3.plot(get_svd(model_lr), label='Explicit', marker='o', alpha=0.6)
    ax3.plot(get_svd(model_muon), label='Muon', marker='x', linestyle='-', linewidth=2, color='green')
    ax3.set_yscale('log')
    ax3.set_title("Singular Value Spectrum")
    ax3.legend()

    # Plot Patterns
    def plot_attn(model, ax, title):
        p = 113
        device = args.device
        model.eval()
        with torch.no_grad():
            tokens = torch.arange(p, device=device)
            emb = model.embedding(tokens) + model.pos_embedding[:, 0, :]
            layer = model.layers[0]
            x = layer['norm1'](emb)
            attn_layer = layer['attn']
            Q = attn_layer.q_norm(attn_layer.q_proj(x))
            K = attn_layer.k_norm(attn_layer.k_proj(x))
            head_dim = model.config.dim // model.config.heads
            Q = Q.view(p, model.config.heads, head_dim)[:, 0, :]
            K = K.view(p, model.config.heads, head_dim)[:, 0, :]
            Attn = (Q @ K.T) / (head_dim**0.5)
            Attn = Attn.softmax(dim=-1).cpu().numpy()

        if sns: sns.heatmap(Attn, ax=ax, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
        else: ax.imshow(Attn, cmap="viridis")
        ax.set_title(title)

    ax4 = fig.add_subplot(gs[1, 0])
    plot_attn(model_l2, ax4, "L2 Pattern")
    ax5 = fig.add_subplot(gs[1, 1:3])
    plot_attn(model_lr, ax5, "Explicit LR Pattern")
    ax6 = fig.add_subplot(gs[1, 3])
    plot_attn(model_muon, ax6, "Muon LR Pattern")

    plt.tight_layout()
    plt.savefig("mechanism_analysis_final.png", dpi=150)
    print("Done. Saved to mechanism_analysis_final.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(42)
    run_mechanism_analysis(args)