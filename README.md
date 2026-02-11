# StockRL

**股票交易强化学习（PPO + LSTM/残差网络）**

这是一个基于强化学习（PPO，近端策略优化）与 LSTM 残差网络的单只股票日内/日间交易模拟与训练仓库。包含数据获取、预处理、环境封装、训练/测试脚本与结果可视化分析工具，方便在本地复现训练流程并将模型输出记录到 `log/` 与 `model/`。

**目录结构（主要文件）**
- `config.py`: 全局参数与目录/日志路径设置（修改此处即可改变代码/训练行为）。
- `dataProcess.py`: 使用 `akshare` 下载股票历史数据并拆分为训练/测试集（保存到 `database/`）。
- `myStockEnv.py`: 自定义环境 `SingleStockDayEnv`，负责状态构造、买卖逻辑与回报计算。
- `myAgent.py`: PPO 代理实现（包含网络结构、回报缓存、更新、保存/加载）。
- `myNN.py`: LSTM + 残差网络（模型骨干），供 `myAgent` 使用。
- `train.py`: 训练入口脚本，执行与 `SingleStockDayEnv` 的交互并训练 PPO。
- `test.py`: 测试/评估入口脚本，加载模型并运行评估回合，输出 csv 到 `log/`。
- `dataAnalysis.py`: 将训练/评估结果生成交互式 HTML（`action_*.html`, `asset_*.html`, `close_price_*.html` 等）。
- `database/`: 原始与拆分后的数据目录（`stockData/`, `trainDataset/`, `testDataset/`）。
- `log/`: 日志、CSV、HTML 可视化输出与 `params.txt`。
- `model/`: 训练中保存的模型文件（`.pth`）。

**快速开始（在 Windows PowerShell 下）**
1. 创建并激活 Python 虚拟环境（可选，但推荐）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安装依赖（推荐使用官方 PyTorch 安装命令根据 CUDA/CPU 选择合适版本）：

```powershell
pip install numpy pandas scikit-learn plotly matplotlib gymnasium akshare
# 安装 PyTorch：请参考 https://pytorch.org 获取与你环境匹配的安装命令，例如：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. 下载并准备数据（以 `config.py` 中 `code` 为准，默认 `600019`）：

```powershell
python dataProcess.py
```

该脚本会把原始数据保存到 `database/stockData/`，并将数据拆分为训练集与测试集，分别存放到 `database/trainDataset/` 与 `database/testDataset/`。

4. 训练模型：

```powershell
python train.py
```

训练过程会：
- 根据 `config.Param()` 的配置创建 `log/<envName>/` 目录并保存 `params.txt`。
- 在训练中不断将每步记录写入 `log/` 下的 CSV 文件，并在达到更新条件时调用 `PPO.update()`。
- 将训练得到的模型保存到 `model/`（例如 `PPO_<envName>_<episode>.pth`）。

5. 评估/测试模型：

```powershell
python test.py
```

该脚本会加载 `model/` 下对应的 `.pth` 文件并生成评估日志 CSV，之后可以使用 `dataAnalysis.py` 对结果做可视化。

6. 结果可视化（示例）：

```powershell
python dataAnalysis.py
```

输出的 HTML 文件会被写入对应 `log/<envName>/` 目录，打开浏览器查看 `asset_0.html`、`action_0.html` 等文件以交互方式审视结果。

**主要配置项（`config.Param`）**
- `code`: 股票代码（默认 `600019`）。
- `seq_length`: 状态序列长度（默认 10）。
- `initial_balance`, `initial_stock_owned`: 环境初始化资金与持股数量。
- `transaction_fee`: 交易手续费比例。
- `fixed_quantity`: 是否使用固定数量买卖（布尔）。
- `K_epochs`, `eps_clip`, `gamma`, `lr_actor`, `lr_critic`: PPO 相关超参。
- `updateNum`: PPO 每次更新所需的最少步数。
- `trainMode`: True=训练使用 `trainDataset`，False=使用 `testDataset`。

可通过编辑 `config.py` 中的 `Param()` 实例来修改以上参数，运行脚本时会自动读取这些设置并在 `log/` 中建立对应目录。

**数据格式要求**
环境代码期望 CSV 包含（中文列名）：
- `日期`, `股票代码`（将在读取时被删除）
- `成交额`, `成交量`, `收盘`, `开盘`, `最高`, `最低`

`myStockEnv` 会对 `成交额`、`成交量`、`收盘`、`开盘`、`最高`、`最低` 做 `MinMaxScaler` 归一化并构造成形状为 `(seq_length, features)` 的状态。

**常见问题与调试建议**
- 如果 PyTorch 找不到合适的二进制包，请按官方说明选择 CPU 或 CUDA 版本进行安装。
- 如果 `dataProcess.py` 下载失败，请确认 `akshare` 可正常访问行情接口并检查网络/代理设置。
- 训练过程可能需要较多时间与显存，建议在无 GPU 环境下将模型/批量参数调小或在云端/有 GPU 的机器运行。

**如何贡献**
- Fork 仓库 -> 创建分支 -> 提交 PR。
- 建议改进项：更通用的多只股票环境、回测框架、更多奖励设计以及模型/超参搜索脚本。

**许可**
本仓库建议使用 MIT 许可证（如需我可以为你添加 `LICENSE` 文件）。

---
如果你希望，我可以：
- 生成 `requirements.txt`（基于当前代码中的依赖）。
- 添加一个简单的 `run_example.ps1` PowerShell 脚本用于一键下载数据并训练一个小样例。

欢迎告诉我你想要的下一步。 
