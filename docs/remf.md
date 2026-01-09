# iMF-Reflow: 基于 Reflow 机制的改进平均流技术文档

## 1. 核心理论：Reflow 与 iMF 的结合

### 1.1 总体目标

我们的目标是训练一个 ODE 模型，使其轨迹尽可能平直，从而实现低误差的一步生成（1-NFE）。

1. **Stage 1 (1-Rectified Flow)**: 学习从标准高斯噪声 到真实数据分布 的映射。此时轨迹可能是弯曲的。
2. **Stage 2 (Reflow / 2-Rectified Flow)**: 利用 Stage 1 模型的 ODE 轨迹生成成对数据 。其中 是噪声， 是 Stage 1 模型生成的“伪数据”。Stage 2 模型学习这对数据之间的**直线路径** 。

### 1.2 损失函数 (iMF Formulation)

无论是在 Stage 1 还是 Stage 2，我们都统一使用 iMF 的复合目标损失函数。这能利用解析梯度的边界条件来稳定速度场的学习。

训练目标是最小化：

其中 的构造利用了 iMF 的雅可比-向量积 (JVP) 技巧：

**符号定义的通用化：**
为了适配 Reflow，我们定义一对训练数据 ：

- **Stage 1**: , 。
- **Stage 2**: , 。

插值公式统一为（ 为数据端， 为噪声端）：

---

## 2. 系统架构 (PyTorch)

网络架构保持轻量级 MLP 设计，但在 Reflow 过程中，我们需要保存和加载模型权重。

### 2.1 MLP 网络结构

```python
import torch
import torch.nn as nn
import numpy as np
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (Batch,)
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class IMF_MLP(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)

        # 输入映射
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # 残差块
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim), # z_emb + t_emb + r_emb
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(3)
        ])

        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, z, r, t):
        # z: (B, data_dim), r: (B,), t: (B,)
        z_emb = self.input_proj(z)
        t_emb = self.time_embed(t)
        r_emb = self.time_embed(r)

        h = z_emb
        for block in self.blocks:
            cond = torch.cat([h, t_emb, r_emb], dim=-1)
            h = h + block(cond)

        return self.output_proj(h)

```

---

## 3. 核心训练逻辑 (通用化 Step)

我们将训练步 `train_step` 改造为接受成对的 `(data, noise)`，以便复用于 Reflow 阶段。

```python
from torch.func import jvp, vmap

def imf_loss_step(model, optimizer, x_data, x_noise):
    """
    通用 iMF 训练步。
    x_data: 数据的端点 (t=0)
    x_noise: 噪声的端点 (t=1)
    """
    optimizer.zero_grad()
    batch_size = x_data.shape[0]
    device = x_data.device

    # 1. 采样时间 t 和 r
    t = torch.rand(batch_size, device=device)
    r = torch.rand(batch_size, device=device)

    # 2. 构造插值数据 (Linear Interpolation)
    [cite_start]# [cite: 1213] Reflow 核心：沿直线路径插值
    t_expand = t.view(-1, 1)
    z_t = (1 - t_expand) * x_data + t_expand * x_noise

    # 3. 预测瞬时速度 (Boundary Condition)
    # u_theta(z_t, t, t)
    v_theta = model(z_t, t, t)

    # 4. JVP 计算 (PyTorch 2.0+ func)
    def model_fn_for_jvp(z_in, r_in, t_in):
        return model(z_in.unsqueeze(0), r_in.unsqueeze(0), t_in.unsqueeze(0)).squeeze(0)

    # Tangents: (v_theta, 0, 1) -> 对应 (z, r, t) 的变化率
    tangents = (v_theta.detach(), torch.zeros_like(r), torch.ones_like(t))

    u_pred, dudt = vmap(jvp, in_dims=(None, (0,0,0), (0,0,0)))(
        model_fn_for_jvp, (z_t, r, t), tangents
    )

    # 5. 构造复合目标 V_target
    # V = u + (t - r) * stop_grad(dudt)
    tr_term = (t - r).view(-1, 1)
    V_target = u_pred + tr_term * dudt.detach()

    # 6. 计算 Loss
    # 目标速度场 flow_target 是 (x_noise - x_data)
    # 因为 z_t = (1-t)x + te, dz/dt = e - x
    flow_target = x_noise - x_data
    loss = torch.mean((V_target - flow_target) ** 2)

    loss.backward()
    optimizer.step()
    return loss.item()

```

---

## 4. Reflow 流程控制

Reflow 的关键在于：**先训练，再生成，再训练**。

### 4.1 ODE 生成器 (用于构造 Reflow 数据集)

为了在 Stage 1 结束后生成高质量的 Reflow 配对数据，我们建议使用多步 Euler 求解器，以确保轨迹的准确性，尽管最终目标是 1-NFE。

```python
@torch.no_grad()
def ode_generate(model, z_start, steps=100):
    """
    从 t=1 (噪声) 积分到 t=0 (数据)。
    z_start: 噪声数据 x_noise
    """
    model.eval()
    device = z_start.device
    B = z_start.shape[0]
    dt = -1.0 / steps # 时间倒流: 1 -> 0

    z = z_start.clone()
    t_curr = torch.ones(B, device=device) # t=1

    for _ in range(steps):
        # 在 Reflow 论文中，生成数据时 r 通常设为当前时间 t，或者直接预测 drift
        # 这里我们用 iMF 的 standard 模式: u(z, 1, 0) 代表平均速度?
        # 不，ODE 求解需要瞬时速度 v(z, t)。
        # 根据 iMF 理论，v(z, t) = model(z, t, t)

        v = model(z, t_curr, t_curr)
        z = z + v * dt
        t_curr = t_curr + dt

    return z # 这就是生成的 x_data (pseudo)

```

### 4.2 完整训练与 Reflow 脚本

```python
import torch
import torch.optim as optim
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# --- 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGE1_STEPS = 4000
STAGE2_STEPS = 4000  # Reflow 训练步数
BATCH_SIZE = 256
REFLOW_DATA_SIZE = 5000 # 用于 Reflow 训练的数据池大小

# --- 数据获取 ---
def get_real_data(n_samples):
    X, _ = make_moons(n_samples=n_samples, noise=0.05)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return torch.from_numpy(X).float().to(DEVICE)

# --- 主程序 ---
def main():
    # ==========================
    # Stage 1: 1-Rectified Flow
    # ==========================
    print(">>> Stage 1: Training on (Noise, Real Data)...")
    model_1 = IMF_MLP().to(DEVICE)
    opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)

    for step in range(STAGE1_STEPS):
        # 1. 准备数据对
        x_real = get_real_data(BATCH_SIZE)
        x_noise = torch.randn_like(x_real)

        # 2. 训练
        loss = imf_loss_step(model_1, opt_1, x_data=x_real, x_noise=x_noise)

        if step % 500 == 0:
            print(f"Stage 1 Step {step}, Loss: {loss:.4f}")

    # ==========================
    # Intermediate: Generate Reflow Dataset
    # ==========================
    print("\n>>> Generating Reflow Dataset (Z0, Z1) from Model 1...")
    [cite_start]# [cite: 1229] Reflow: Construct (Z0, Z1) pairs using the 1-rectified flow
    # 我们固定一批噪声 Z_noise (即 Z_1 in paper notation)
    reflow_noise = torch.randn(REFLOW_DATA_SIZE, 2, device=DEVICE)

    # 使用 Model 1 的 ODE 求解生成对应的 Z_data (即 Z_0)
    # 使用较多的步数 (N=100) 以获得精确的轨迹端点
    reflow_data = ode_generate(model_1, reflow_noise, steps=100)

    # 此时 (reflow_data, reflow_noise) 构成了一个新的“拉直”了的耦合

    # ==========================
    # Stage 2: 2-Rectified Flow (Reflow)
    # ==========================
    print("\n>>> Stage 2: Training Reflow Model on (Z0, Z1)...")
    [cite_start]# [cite: 1230] Train a new model on the induced pairs to straighten paths
    model_2 = IMF_MLP().to(DEVICE) # 初始化新模型
    opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)

    # 将数据封装以便 Batch 采样
    dataset = torch.utils.data.TensorDataset(reflow_data, reflow_noise)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    step = 0
    while step < STAGE2_STEPS:
        for x_data_batch, x_noise_batch in loader:
            # 这里的 x_data_batch 是 model_1 生成的数据
            loss = imf_loss_step(model_2, opt_2, x_data=x_data_batch, x_noise=x_noise_batch)

            if step % 500 == 0:
                print(f"Stage 2 (Reflow) Step {step}, Loss: {loss:.4f}")
            step += 1
            if step >= STAGE2_STEPS: break

    # ==========================
    # Evaluation: 1-NFE Sampling
    # ==========================
    print("\n>>> Evaluating 1-NFE Generation...")
    model_2.eval()
    with torch.no_grad():
        test_noise = torch.randn(1000, 2, device=DEVICE)
        ones = torch.ones(1000, device=DEVICE)
        zeros = torch.zeros(1000, device=DEVICE)

        # 1-NFE: 直接预测从 t=1 到 t=0 的平均速度
        # 在 iMF 框架下，跨度为 1 的平均速度即为位移
        # u = model(z_noise, 1, 0)
        u = model_2(test_noise, ones, zeros)

        # z_0 = z_1 - u * (1 - 0)
        z_gen = test_noise - u

        z_gen = z_gen.cpu().numpy()
        real_data = get_real_data(1000).cpu().numpy()

    # 绘图对比
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Real Data")
    plt.scatter(real_data[:,0], real_data[:,1], s=5, alpha=0.5)
    plt.subplot(122)
    plt.title("iMF + Reflow (1-NFE Generated)")
    plt.scatter(z_gen[:,0], z_gen[:,1], s=5, alpha=0.5, c='red')
    plt.show()

if __name__ == "__main__":
    main()

```

## 5. 实现细节与注意事项

1. **Reflow 数据集的构建**:

- 在 `Intermediate` 阶段，必须使用 Model 1 生成数据。我们通过 ODE 求解器（如 100 步 Euler）从噪声回推到数据。这样生成的 对实际上通过 Model 1 的流场连接 。

- Stage 2 的训练仅仅是在学习连接这两点的**直线**。因为 Model 1 已经把噪声映射到了（近似）数据流形上，Stage 2 的任务只是把这个映射过程“拉直” 。

2. **JVP 的作用保持不变**:

- 即使是在 Reflow 阶段，我们依然使用 iMF 的 JVP Loss。这是因为我们希望 Stage 2 模型不仅学会直线轨迹，还要在每一点都满足 的物理约束，这有助于模型在 1-NFE 推理时更加鲁棒。

3. **1-NFE 的实现**:

- 在最后的评估中，我们直接调用 `model(z, 1, 0)`。由于经过了 Reflow，轨迹变得极度接近直线，因此使用单步欧拉法（即直接减去平均速度）的误差非常小，从而实现了高质量的快速采样 。
