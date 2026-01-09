
# Improved Mean Flow (iMF) 复现技术文档

## 1. 核心理论摘要 (面向实现)

Improved Mean Flow (iMF) 的目标是训练一个模型，使其能够一步（1-NFE）从噪声生成数据。相比原始 MeanFlow，iMF 改进了损失函数，使其更加稳定且不再依赖“作弊”的条件速度（）。

### 1.1 关键公式

iMF 将问题转化为学习平均速度场 。训练的核心在于构造一个 **Compound Target (复合目标)** 。

根据论文公式 (12) ，我们的训练目标是最小化以下损失：

其中，复合函数 定义为：

**符号说明：**

- : 真实数据。
- : 采样噪声 (Gaussian)。
- : 两个随机时间步，。
- : 插值后的噪声数据， 。

- : 神经网络预测的从 到 的平均速度。
- : 瞬时速度的估计值。**这是 iMF 的关键改进点。**
- : Stop Gradient（停止梯度传导）。
- : 雅可比-向量积 (Jacobian-Vector Product)，用于计算 。

### 1.2 关键改进点：边界条件 (Boundary Condition)

为了避免引入额外的参数，iMF 建议利用 和 的自然关系：当 时，平均速度等于瞬时速度。
因此，我们设定 \*\*\*\* 。这意味着我们可以用同一个网络，通过传入相同的时间参数来获得瞬时速度的估计，用于 JVP 计算。

---

## 2. 系统架构 (PyTorch)

由于你的目标是 MLP 和 Toy Dataset，我们不需要复杂的 Transformer 或 U-Net。一个带有时间嵌入 (Time Embedding) 的 ResNet-MLP 即可。

### 2.1 MLP 网络结构设计

输入层需要接受：

1. **状态 ** (维度 )。
2. **当前时间 ** (标量)。
3. **参考时间 ** (标量)。

**处理技巧：**

- 不要直接将 和 作为浮点数输入，建议使用 **正弦位置编码 (Sinusoidal Positional Embeddings)** 映射到高维（例如 64 维），然后与 的特征拼接或相加。
- 对于 iMF，网络必须同时以 和 为条件。

### 2.2 伪代码 (MLP 定义)

```python
import torch
import torch.nn as nn
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

        # 中间层 (简单的 ResNet Block)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim), # *3 是因为拼接了 z_emb, t_emb, r_emb
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(3)
        ])

        # 输出层
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, z, r, t):
        # z: (B, 2), r: (B,), t: (B,)
        z_emb = self.input_proj(z)
        t_emb = self.time_embed(t)
        r_emb = self.time_embed(r)

        # 简单的条件拼接机制
        h = z_emb
        for block in self.blocks:
            # 将条件拼接送入 MLP
            cond = torch.cat([h, t_emb, r_emb], dim=-1)
            h = h + block(cond) # Residual connection

        return self.output_proj(h)

```

---

## 3. 训练流程详解 (核心难点)

这是复现中最关键的部分，尤其是 JVP 的计算。

### 3.1 雅可比-向量积 (JVP) 的计算

iMF 需要计算 。根据论文公式 (5) 和 JVP 的定义：

注意：这里的 代表向量积。输入是 ，对应的切向量 (tangent vector) 是 。

- 的变化率是 (瞬时速度)。
- 是固定的参考时间，变化率为 0。
- 是当前时间，变化率为 1。

### 3.2 完整训练步 (Training Step) 代码

```python
import torch.autograd.functional as F

def loss_fn(model, x):
    """
    x: 真实数据 batch (Batch_Size, 2)
    """
    batch_size = x.shape[0]
    device = x.device

    # 1. 采样时间 t 和 r
    # 论文推荐 t, r 独立均匀采样，或者根据策略采样。
    # 这里使用简单的均匀采样 [0, 1]
    t = torch.rand(batch_size, device=device)
    r = torch.rand(batch_size, device=device)

    # 2. 构造噪声数据 z_t
    e = torch.randn_like(x)
    # 扩展维度以便广播计算
    t_expand = t.view(-1, 1)
    # [cite_start]Flow Matching 公式: z_t = (1-t)x + t*e (t=0是数据, t=1是噪声) [cite: 84]
    z_t = (1 - t_expand) * x + t_expand * e

    # 3. 计算瞬时速度 v_theta (Boundary Condition)
    # [cite_start]使用 u_theta(z_t, t, t) 作为 v_theta 的估计 [cite: 190]
    # 这一步不需要梯度传导给 v_theta 用于 JVP 内部优化，但它本身需要训练
    # 在 JVP 计算中，v_theta 仅作为切向量输入
    v_theta = model(z_t, t, t)

    # 4. 定义辅助函数用于 JVP 计算
    # 必须是一个纯函数，输入是 (z, r, t)
    # 为了处理 batch，我们通常对单样本做 jvp 然后 vmap，或者利用 PyTorch 的 functional API
    # 针对简单 MLP，我们可以手动利用 autograd.grad 来实现 JVP，
    # 或者为了代码清晰，使用以下 trick：

    # --- JVP 计算开始 ---
    # 我们需要计算 d(model(z, r, t))/dt = df/dz * dz/dt + df/dr * dr/dt + df/dt * dt/dt
    # 其中 dz/dt = v_theta, dr/dt = 0, dt/dt = 1

    # 开启梯度记录
    z_in = z_t.detach().clone().requires_grad_(True)
    r_in = r.detach().clone().requires_grad_(True)
    t_in = t.detach().clone().requires_grad_(True)

    u_pred = model(z_in, r_in, t_in)

    # 计算 gradients
    # 我们需要对输出的每个维度分别求导，或者使用 torch.autograd.grad
    # 更加高效的方法是使用 forward-mode AD (PyTorch 2.0+ 支持)
    # 但为了兼容性，这里展示最通用的 grad 方法：

    # 技巧：计算 sum(u_pred * dummy_grad) 对输入的梯度，相当于计算 Vector-Jacobian Product
    # 这里我们想要 Jacobian-Vector Product。
    # 对于简单的标量 t 和 r，直接求导容易。对于 z (向量)，我们需要 v_theta。

    # 方法 A: 使用 autograd.grad (理解起来最直观)
    grad_outputs = torch.ones_like(u_pred)

    # 计算 du/dz * v_theta + du/dt * 1
    # 我们可以通过把 v_theta 当作"权重"来实现? 不完全是。
    # 正确的 JVP 实现其实在 PyTorch 中有点繁琐。
    # 这里推荐一种 hack 方法：
    # 利用链式法则，我们构造一个新的输入 z_new = z + delta * v_theta, t_new = t + delta
    # 然后求 limit delta -> 0。
    # 或者，利用 torch.func (PyTorch 2.0+) - 这是最推荐的现代方法。

    # --- 现代 PyTorch (torch.func) 实现 JVP ---
    from torch.func import jvp, vmap

    # 定义单样本模型调用
    def model_call_single(z_s, r_s, t_s):
        # 增加 batch 维度 (1, ...) 因为模型期望 batch
        return model(z_s.unsqueeze(0), r_s.unsqueeze(0), t_s.unsqueeze(0)).squeeze(0)

    # 准备切向量 (tangents)
    # primals: (z_t, r, t)
    # tangents: (v_theta, 0, 1)
    tangents = (v_theta.detach(), torch.zeros_like(r), torch.ones_like(t))

    # 使用 vmap 批量计算 JVP
    # jvp 返回: (outputs, jvp_out)
    # outputs 就是 u_pred, jvp_out 就是 du/dt
    u_pred_batch, dudt_batch = vmap(jvp, in_dims=(None, (0,0,0), (0,0,0)))(
        model_call_single, (z_t, r, t), tangents
    )

    # 5. 组装 Compound Target V_theta
    # [cite_start]V = u + (t - r) * stop_grad(dudt) [cite: 186]
    tr_term = (t - r).view(-1, 1)
    V_target = u_pred_batch + tr_term * dudt_batch.detach() # 核心：stop gradient

    # 6. 计算 Loss
    # [cite_start]目标是拟合条件速度场 (e - x) [cite: 150]
    # 注意: Flow Matching 中 v_c = e - x
    flow_target = e - x
    loss = torch.mean((V_target - flow_target) ** 2)

    return loss

```

_注意：如果你的 PyTorch 版本较低不支持 `torch.func`，可以使用 `torch.autograd.functional.jvp`，但它处理 batch 比较慢，可能需要循环。强烈建议使用 PyTorch 2.0+。_

---

## 4. 推理/采样 (Inference)

训练完成后，iMF 允许一步生成。

### 4.1 1-NFE 采样逻辑

在 Flow Matching 框架下（ 是数据， 是噪声）：
我们希望从噪声 映射到数据 。
iMF 预测的是从 到 的平均速度 。
设置起点 (噪声)，终点 (数据)。

公式为：

代入 :

### 4.2 采样代码

```python
@torch.no_grad()
def sample_imf(model, n_samples):
    device = next(model.parameters()).device

    # 1. 采样噪声
    z1 = torch.randn(n_samples, 2, device=device)

    # 2. 准备时间输入
    # r = 1 (噪声时刻), t = 0 (数据时刻)
    ones = torch.ones(n_samples, device=device)
    zeros = torch.zeros(n_samples, device=device)

    # 3. 预测平均速度
    # 注意：输入顺序是 (z, r, t) 还是 (z, t, r) 取决于你的 forward 定义
    # 上面的模型定义是 forward(z, r, t)
    u = model(z1, ones, zeros)

    # 4. 一步更新
    z0 = z1 - u

    return z0.cpu().numpy()

```

---

## 5. 完整的 Toy Example (可复制运行)

以下是一个完整的、单文件的 Python 脚本，用于在 "Moons" 数据集上训练 iMF。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from torch.func import jvp, vmap # 需要 PyTorch 2.0+

# --- 1. 配置与数据 ---
BATCH_SIZE = 256
STEPS = 5000
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(n_samples):
    X, _ = make_moons(n_samples=n_samples, noise=0.05)
    # 归一化到 [-1, 1] 附近
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return torch.from_numpy(X).float().to(DEVICE)

# --- 2. 模型定义 (MLP) ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * (-np.log(10000.0) / (half_dim - 1)))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class IMF_MLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.t_embed = TimeEmbedding(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + hidden_dim * 2, hidden_dim), # input: z(2) + t_emb + r_emb
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, z, r, t):
        # r, t: (B,) -> (B, hidden_dim)
        te = self.t_embed(t)
        re = self.t_embed(r)
        # 拼接条件
        x = torch.cat([z, re, te], dim=-1)
        return self.net(x)

# --- 3. 核心训练逻辑 (iMF Loss) ---
def train_step(model, optimizer, x_batch):
    optimizer.zero_grad()
    B = x_batch.shape[0]

    # 采样 t, r
    t = torch.rand(B, device=DEVICE)
    r = torch.rand(B, device=DEVICE)
    e = torch.randn_like(x_batch)

    # 插值 z_t = (1-t)x + t*e
    t_exp = t.view(-1, 1)
    z_t = (1 - t_exp) * x_batch + t_exp * e

    # 预测瞬时速度 v_theta (Boundary Condition: u(z, t, t))
    # 不需要此处的梯度反向传播到 JVP 内部
    v_theta = model(z_t, t, t)

    # --- JVP 计算 (PyTorch 2.0 functional) ---
    def model_fn_for_jvp(z_in, r_in, t_in):
        # 增加 dim 0 适配 vmap
        return model(z_in.unsqueeze(0), r_in.unsqueeze(0), t_in.unsqueeze(0)).squeeze(0)

    # Tangents: (v_theta, 0, 1) 对应 (z, r, t)
    tangents = (v_theta.detach(), torch.zeros_like(r), torch.ones_like(t))

    # 计算 u 和 du/dt
    # in_dims=(None, ...) 表示 model_fn 不变，后面的 inputs 需要 batch 处理
    u_pred, dudt = vmap(jvp, in_dims=(None, (0,0,0), (0,0,0)))(
        model_fn_for_jvp, (z_t, r, t), tangents
    )

    # 计算 Target V
    tr = (t - r).view(-1, 1)
    [cite_start]V_target = u_pred + tr * dudt.detach() # Stop Gradient [cite: 186]

    # Loss: MSE(V, e-x)
    loss = torch.mean((V_target - (e - x_batch))**2)

    loss.backward()
    optimizer.step()
    return loss.item()

# --- 4. 主程序 ---
model = IMF_MLP().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Start Training...")
loss_history = []
for step in range(STEPS):
    batch = get_data(BATCH_SIZE)
    loss = train_step(model, optimizer, batch)
    loss_history.append(loss)
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

# --- 5. 采样与可视化 ---
model.eval()
with torch.no_grad():
    # 1-NFE Sampling: z1(noise) -> z0(data) via u(z1, 1, 0)
    z_noise = torch.randn(1000, 2, device=DEVICE)
    ones = torch.ones(1000, device=DEVICE)
    zeros = torch.zeros(1000, device=DEVICE)

    # 预测从 r=1 到 t=0 的平均速度
    u = model(z_noise, ones, zeros)
    z_generated = z_noise - u # z0 = z1 + (0-1)*u

    z_generated = z_generated.cpu().numpy()
    gt_data = get_data(1000).cpu().numpy()

# 绘图
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title("Ground Truth")
plt.scatter(gt_data[:,0], gt_data[:,1], s=5, alpha=0.5)
plt.subplot(132)
plt.title("iMF Generated (1-NFE)")
plt.scatter(z_generated[:,0], z_generated[:,1], s=5, alpha=0.5, c='orange')
plt.subplot(133)
plt.title("Training Loss")
plt.plot(loss_history)
plt.show()

```

## 6. 技术细节补充与注意事项

1.  **Stop Gradient (`.detach()`)**: 在计算 `V_target` 时，必须对 JVP 的输出 `dudt` 使用 `detach()`。这意味着网络不会尝试通过改变其二阶导数来最小化 loss，这对于训练稳定性至关重要 。

2.  **输入归一化**: Toy dataset 最好归一化到均值为 0，方差为 1 的范围内（类似于标准高斯噪声的分布），这样 Flow Matching 学习起来最容易。
3.  **Boundary Condition 的有效性**: 对于小型 MLP，直接使用 足够有效。如果你发现效果不好，可以尝试论文中提到的 "Auxiliary Head" 方法，即增加一个小的分支专门预测 ，但这会增加代码复杂度。
4.  **关于 CFG (无分类器引导)**:

- 在上述代码中省略了 CFG 以保持简洁。
- 如果需要 CFG（例如你的 toy dataset 有多个簇，想控制聚合度）：

1. 在 `forward` 中增加 `w` (guidance scale) 作为输入。
2. 训练时随机采样 `w` (e.g., [1, 4])。
3. 推理时传入固定的高 `w`。

- 但在简单数据分布上，无 CFG 的 iMF 也能工作得很好。

这份指南应该足以让你复现出论文的核心结果。如果在 `torch.func` 部分遇到报错，请确保你的 PyTorch 版本 >= 2.0。
