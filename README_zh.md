# FlyBrain Snake: 连接组在环感觉运动伺服系统

[![Connectome](https://img.shields.io/badge/Connectome-MaleCNS%20v1.0-blue.svg)](https://codex.flywire.ai)
[![Neurons](https://img.shields.io/badge/Neurons-166%2C700%20Spiking%20LIF-green.svg)](https://github.com/alextitonis/flybrain)
[![Synapses](https://img.shields.io/badge/Synapses-125M%20Dataset%20%7C%20~25.6M%20Runtime-purple.svg)](https://codex.flywire.ai)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

[**English**](README.md) | [**中文说明**](README_zh.md)

一个具身计算神经科学探索项目：基于黑腹果蝇（*Drosophila melanogaster*）全中枢神经系统连接组动力学（[`flybrain`](https://github.com/alextitonis/flybrain)），实现闭环感觉运动寻路、外周净空避障反射与实时 WebGL 神经遥测 HUD。

---

## 🏛️ 系统架构：连接组在环感觉运动伺服系统 (System Architecture)

本项目探索并实现了一套典型的**『连接组在环感觉运动伺服系统（Connectome-in-the-Loop Sensorimotor System）』**，在控制层级上对应昆虫经典的感觉运动分工架构（头部回路负责定向 + 胸腹神经索负责近身避碰）。**中央复合体（CX）仪表盘仅作为状态监视与提取，闭环转向动作完全来自 DNa02 下行差模与外周净空 FSM**：

1. **全脑动力学层 / 下行意图层 (Whole-Brain Dynamics / Descending Intent Layer)**：  
   基于 **MaleCNS v1.0** 全脑连接组（16.67 万神经元，运行时稀疏突触阵约 2,558 万条），执行 4 步脉冲泄漏积分发放（LIF）网络动力学。作为多模态（嗅觉敏化门控 + 视网膜注视追踪）的**非线性生物动力学滤波器**，自发输出下行运动意图（`DNa02` 差模）。在每个决策 tick 开始时，通过 `brain.v.fill(0)` 复位全脑膜电位，清空跨 tick 运动迟滞，使每个 4 步 LIF 窗口成为纯粹的准静态前馈动力学滤波过程，无跨周期隐层记忆残留；
2. **低级反射层 (VNC 功能类比的净空 FSM)**：  
   功能类比昆虫**腹神经索（VNC）与外周反射弧**。由确定性有限状态机（FSM）提供双重生存防护：
   - **正面避碰覆写**：在正前方遭遇贴身障碍（`dist_front == 0`）时，强制执行反向/侧向紧急避碰覆写（Reflex Override）；
   - **侧向防贴身抑制**：在侧向紧贴障碍（`dist_left == 0` 或 `dist_right == 0`）时，主动抑制下行转向指令向该侧转向（Lateral Clearance Guard），防止下行命令误撞侧壁；
3. **空间接口层 (Spatial Interface Layer)**：  
   充当**虚拟复眼与触角**，将 2D 连续网格几何翻译为神经外加电压刺激（`brain.stimulate()`）：
   - 触角叶投射神经元（`AL`，DM1/VA2/DM2）：注入双侧气味浓度差 $\Delta c$；
   - 视小叶柱状神经元（`LC10a`）：注入视网膜偏航角 $\theta_{\text{bearing}}$；
   - 隐现逼近神经元（`LC4`/`LPLC2`）：在近距障碍（净空 $\le 1$）时注入视网膜扩张电压刺激。

```text
======================= 闭环感觉运动控制主干 =======================

 ┌────────────────────────────────────────────────────────────────────────┐
 │                      1. 空间接口层 (SPATIAL TRANSDUCTION)              │
 │   虚拟复眼 (Ommatidia) ──► 视网膜偏航角 θ (LC10a) & 近场隐现 (LC4)     │
 │   虚拟触角 (Antennae)  ──► 双侧气味浓度差 Δc (AL)                      │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ 神经膜电位电压刺激 (Voltage Depolarization)
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │       2. 全脑动力学层 / 下行意图层 (MaleCNS 166.7k, 约 2558 万突触)    │
 │                                                                        │
 │  嗅觉 (AL) + 气味门控视觉 (LC10a) + 隐现逼近 (LC4/LPLC2)               │
 │                                  │                                     │
 │                                  ▼ (4步 全脑 LIF 动力学, v.fill(0))     │
 │  下行运动意图：双侧下行神经元膜电位差模 (DNa02_L - DNa02_R)            │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ 下行转向意图 (Descending Steering Intent)
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                  3. 低级反射层 (VNC 功能类比的净空 FSM)                │
 │                                                                        │
 │   前方开阔 (dist_front > 0)    ──► 执行 DNa02 意图转向 / DNa01 巡航   │
 │   正面障碍 (dist_front == 0)   ──► 触发毫秒级外周紧急避碰 (FSM 转向)  │
 │   侧向贴身 (dist_lat == 0)     ──► 抑制下行转向指令撞向侧壁           │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ 离散执行动作
                                    ▼
                         [ 2D ARENA EMBODIMENT ]

=================== 神经遥测与状态监视 HUD (只读监控流) ===================
(提取自仿真运行过程，用于前端科学可视化与状态监视，不参与闭环动作决策)

  • E-PG 罗盘环  : 由当前朝向重建的 16 楔示意（非 E-PG 原生胞体读出）
  • PFL3 比较器  : 扇形体到外侧副叶（LAL）双侧航向差模平衡表
  • PAM 多巴胺簇 : 吃到食物时的瞬态进食奖赏神经烟花
  • 3D 全脑视图  : WebGL 按事件点亮解剖区位（视小叶火花、全脑光流与 VNC 轨迹）
```

---

## 🧬 生物神经回路映射机制 (Neurobiological Circuits)

1. **物理感觉通道严格解耦 (Sensory Decoupling)**：
   - **触角叶 (AL, DM1/VA2/DM2)**：只接收左右触角真实采样得到的化学浓度差 $\Delta c = c_L - c_R$，绝对不掺杂视觉空间方位角；
   - **视小叶 (LC10a)**：只编码食物目标相对于蛇头朝向的视网膜几何偏航角 $\theta_{\text{bearing}}$（设置 $\pm 0.05\text{ rad} \approx 2.9^\circ$ 直行死区以过滤微电位噪声）；
   - **近距隐现逼近 (LC4 / LPLC2)**：在贴近障碍物（净空 $\le 1$）时在空间接口层施加电压刺激。
2. **编码器侧气味敏化门控 (Odor-Gated Visual Sensitization)**：
   - 采用编码器侧的敏化公式，类比生物学中的气味门控视觉追踪机制：
     $$G_{\text{visual}} = G_{\text{base}} \times \left(1.0 + \alpha \cdot \frac{c_{\max}}{c_0}\right)$$
   - 远离食物时维持基线增益（~4.5）平稳大范围探索，近距捕食时敏锐度提升至 3~4 倍（~14.0），驱动果断向心定位置换。
3. **下行运动电位解码 (Descending Motor Readout)**：
   - 转向意图直接从连接组中真实的 **`DNa02`** 双侧电位差模解码（`diff = DNa02_L - DNa02_R > 0.03`），**无左右常值偏置**（去除了早期版本中的常数不对称偏移量；接口死区 $0.05\text{ rad}$ 与判定门限 $0.03$ 仍作为离散工程接口参数保留）；
   - 基础行进由 `DNa01` 提供前行推进偏置；
   - 每步开始时通过 `brain.v.fill(0)` 复位膜电位，防止跨步误差积累。
4. **VNC 功能类比的净空 FSM (Reflex Collision Override)**：
   - **正面避让**：在贴身撞墙或自交的致命危机（`dist_front == 0`）下，由 `MotorDecoder` 触发外周紧急避碰，并在前端仪表盘亮起 `⚡ REFLEX OVERRIDE (SPINAL COLLISION GUARD)` 警告；
   - **侧向抑制**：当检测到侧面贴壁（`dist_left == 0` 或 `dist_right == 0`）时，拦截向障碍侧转向的意图，保障侧向安全。
5. **只读神经遥测与状态 HUD (Read-Only Neuro-Telemetry HUD)**：
   - **E-PG 内部罗盘**：由贪吃蛇当前朝向角实时重建的 16 楔极坐标示意，便于观察朝向变化；
   - **PFL3 转向平衡仪**：扇形体双侧前运动电位差监视；
   - **PAM 多巴胺爆发**：食物捕获瞬态的奖赏神经光效；
   - **3D 脑体呈现**：WebGL 按事件点亮解剖区位（视小叶追踪火花、中枢脑光流与 VNC 轨迹）。

---

## 🚀 快速开始 (Quick Start)

### 1. 环境依赖与启动

本项目依赖极轻量（仅需 `flybrain` 与 `pytest`），Web 服务由 Python 标准库 `ThreadingHTTPServer` 与 Server-Sent Events (SSE) 驱动，无需大型深度学习框架：

```bash
# 安装依赖
pip install -r requirements.txt

# 启动仿真服务器
python3 run_server.py
```

在浏览器中打开：[http://localhost:8080](http://localhost:8080)。

### 2. 交互控制快捷键

- `Space`：播放 / 暂停自主仿真
- `S`：单步推进（前进一个 80ms 仿真窗口）
- `R`：重置游戏与脑回路
- `O`：开关 2D 嗅觉扩散羽流（Odor Plume）
- 鼠标拖拽 / 滚轮：自由旋转、平移与缩放 3D 16.6 万果蝇全中枢连接组模型

---

## 🧪 测试与基准评估 (Tests & Benchmark)

```bash
# 运行单元与集成测试套件 (65 passed)
pytest

# 运行 50-episode 感觉运动性能基准测试
python3 -m src.benchmark.runner --episodes 50
```

---

## 📚 详细技术文档 (Documentation)

- [**神经回路与工作原理指南 (中文)**](docs/NEURAL_CIRCUITS_zh.md): 详尽的果蝇全中枢连接组回路映射、多模态气味门控视觉追踪与下行控制说明。
- [**Neural Circuits & Architecture Guide (EN)**](docs/NEURAL_CIRCUITS.md): In-depth connectomic sensorimotor pathways, odor-gated visual pursuit, and descending motor decoding.
- [**English README (README.md)**](README.md): Complete project overview and architecture guide in English.
