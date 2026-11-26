# 沪深 300 多因子选股与交易成本回测

本人制作此项目的在于积累一些关于量化交易的基础实操经验，和熟悉python编程，学习过程中先是跟随微信公众号平台上关于baostock的一篇教程学会并编写了一个获取k线/基础财务数据的简单程序，接下来就切换到以deepseek为老师一边学一边开发的状态，sg0和sg1两个阶段，基本都是我向大模型提问后自己边理解边手敲到vscode，至于后面阶段滑点手续费以及接入joinquant的alpha101和华泰金工的部分因子，则是交给大模型来写。项目作为回测框架比较粗糙，仅仅作为一次学习的副产物，以后还是会转向backtesting等正式框架进行学习。

## 项目简介

本项目基于 Python、Baostock 和 Pandas，构建了一套从数据获取、清洗、因子计算、横截面检验到交易成本回测的日频量化研究流程。

研究范围：

```text
股票池：当前沪深 300 成分股，共 300 只
时间范围：2020-01-02 至 2025-12-31
数据频率：日频
价格口径：前复权
清洗后样本：416,691 条股票-日记录
```

项目重点不只是展示高收益结果，而是完整回答问题：

- 因子是否包含未来信息？
- 行情和基本面数据是否按可获得时间对齐？
- 分层收益是否在扣除成本后仍然存在？
- 换手率、滑点、停牌和涨跌停是否会使策略失效？
- 样本外是否仍能保留部分表现？

## 已实现内容

- 沪深 300 成分股、个股日线、指数和股票基础信息获取
- 复权口径、异常价格、重复记录和停牌处理
- BP、MOM20、VOL60 基础因子
- Rank IC、IR、五分组收益和多空组合
- MAD 去极值、横截面 Z-score 和等权合成分
- 成交量漂移后的换手率计算
- 佣金、过户费、历史印花税和滑点情景
- 次日开盘入场、隔日开盘出场
- ST、上市天数、停牌和涨跌停成交限制
- Alpha101 日频子集试点
- 华泰风格价量 + 基本面多因子试点
- 训练期与验证期分段评价

## 项目结构

```text
回测项目/
├─ code/
│  ├─ sg1_data_fetch.py
│  ├─ sg1_dataclean.py
│  ├─ sg1_factors.py
│  ├─ sg1_standardize.py
│  ├─ sg1_ic.py
│  ├─ sg1_backtest.py
│  ├─ sg1_performance.py
│  ├─ sg2_composite_analysis.py
│  ├─ sg3_turnover.py
│  ├─ sg4_cost_sensitivity.py
│  ├─ sg5_fetch_stock_basic.py
│  ├─ sg5_fetch_calendar.py
│  ├─ sg5_execution_frictions.py
│  ├─ sg6_long_only.py
│  ├─ alpha101_daily/
│  │  ├─ alphas.py
│  │  ├─ operators.py
│  │  ├─ engine.py
│  │  └─ run_pilot.py
│  └─ huatai_multifactor/
│     ├─ factors.py
│     ├─ fetch_fundamentals.py
│     └─ run_pilot.py
├─ data/
│  ├─ raw/
│  └─ processed/
├─ output/
└─ README.md
```

`data/` 和 `output/` 默认不进入 Git，原因是数据体积较大且结果可由代码重新生成。

## 数据说明

### 行情数据

使用 Baostock 获取：

```text
open, high, low, close, preclose
volume, amount, turn, pctChg
peTTM, pbMRQ, psTTM, pcfNcfTTM
tradestatus, isST
```

价格使用 `adjustflag="2"` 前复权。

数据质量处理：

- 删除重复的 `(date, code)` 记录；
- 检查非正价格和 OHLC 逻辑异常；
- 统一数值字段类型；
- 过滤少量复权标记异常的历史片段；
- 不使用空文件覆盖已有有效文件。

### 股票基础信息

从 Baostock 获取 `ipoDate`、`outDate`、股票类型和状态，用于：

- 计算上市后真实交易日数量；
- 排除上市不足 250 个交易日的股票；
- 控制新股早期异常波动。

### 基本面信息

第一版华泰风格基本面因子使用年度利润表数据：

```text
pubDate, statDate
roeAvg, npMargin, gpMargin
netProfit, MBRevenue
```

公告日 `pubDate` 会先对齐到下一个交易日，再用于当日选股，避免未来函数。当前版本尚未扩展到季度和 TTM。

## 基础因子

### BP

```text
BP = 1 / pbMRQ
```

`pbMRQ` 缺失、为 0 或为负时，BP 设为缺失值。

### MOM20

```text
MOM20 = close / close.shift(20) - 1
```

衡量过去 20 个交易日的价格动量。

### VOL60

```text
ret = close / preclose - 1
VOL60 = std(ret, 60)
```

衡量过去 60 个交易日收益波动。

三个因子在每个交易日内分别进行：

```text
MAD 去极值
→ Z-score 标准化
→ 等权合成分
```

## Alpha101 日频子集

代码目录：

```text
code/alpha101_daily/
```

第一版只实现 20 个不依赖行业、市值或分钟数据的日频因子，包括：

```text
alpha001, alpha002, alpha003, alpha004, alpha005
alpha006, alpha007, alpha008, alpha009, alpha010
alpha011, alpha012, alpha013, alpha014, alpha015
alpha016, alpha018, alpha020, alpha022, alpha025
```

该模块属于基于公开公式的日频子集实现，不是 JoinQuant 平台的逐行复制，也没有行业中性化。

初版结果中，`alpha005` 表现相对稳定：

```text
净年化收益：22.95%
净夏普：0.934
验证期净夏普：0.819
最大回撤：-30.07%
日均换手：4.83%
```

多数高换手 Alpha101 因子扣除成本后失效，说明成本约束不能省略。

## 华泰风格多因子

代码目录：

```text
code/huatai_multifactor/
```

这是一套基于常见华泰多因子分类思路实现的可复现试验，不是华泰官方因子代码。

### 日频价量因子

```text
MOM20, MOM60, MOM120
REV5
VOL20, VOL60
TURN20
ILLIQ20
HIGH52
```

### 基本面与估值因子

```text
EP, BP, SP, CFP
ROE, NPM, GPM
PROFIT_YOY, REVENUE_YOY
```

### 合成分

```text
PRICE_COMPOSITE_Z
FUNDAMENTAL_COMPOSITE_Z
TOTAL_COMPOSITE_Z
```

因子方向统一为“数值越高越偏多头”。合成分在每个交易日内进行 MAD 去极值和 Z-score。

部分结果：

| 因子 | 净年化 | 净夏普 | 验证期夏普 | 最大回撤 | 日均换手 |
|---|---:|---:|---:|---:|---:|
| ILLIQ20 | 28.03% | 1.246 | 1.087 | -25.47% | 4.69% |
| GPM | 16.03% | 0.725 | 0.575 | -30.36% | 1.24% |
| PRICE_COMPOSITE_Z | 13.81% | 0.811 | 0.738 | -20.78% | 10.03% |
| SP | 15.80% | 0.803 | 0.860 | -20.45% | 1.96% |
| TOTAL_COMPOSITE_Z | 10.18% | 0.683 | 0.826 | -19.23% | 4.89% |

`ILLIQ20` 是低流动性溢价，容量有限，不能仅凭历史收益判断可实施性。

## 回测口径

### 信号与执行

```text
t 日收盘生成信号
t+1 日开盘入场
t+2 日开盘出场
```

组合为因子最高 20% 的等权多头。

### 可交易过滤

- 信号日、入场日和出场日必须正常交易；
- 排除 ST；
- 排除上市不足 250 个交易日的股票；
- 排除停牌；
- 排除开盘即涨跌停锁定、无法成交的股票。

### 交易成本

```text
佣金：双边各 2.5 bp
过户费：双边各 0.1 bp
印花税：卖出收取
2020-01-01 至 2023-08-27：10 bp
2023-08-28 至 2025-12-31：5 bp
滑点：0 / 5 / 10 / 20 bp 情景测试
```

## 主要发现

1. 单因子 Rank IC 较小，但部分价量因子可在扣成本后保持正收益。
2. 高换手 Alpha101 因子通常在成本后失效。
3. 原多空组合的收益主要受空头腿拖累，单边做多后收益转正。
4. 基本面年频因子的增量有限，价量合成分表现更稳定。
5. 低流动性因子历史表现较好，但实际容量和冲击成本存疑。
6. 当前结果仍存在成分股偏差，不能视为严格历史可投资组合。

## 环境与运行

建议使用 Python 3.12 或更高版本，并建立独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install baostock pandas numpy pyarrow matplotlib
```

基础数据流程：

```powershell
cd code
python sg1_data_fetch.py
python sg1_dataclean.py
python sg1_factors.py
python sg1_standardize.py
python sg1_ic.py
python sg1_backtest.py
python sg1_performance.py
```

执行成本与单边对照：

```powershell
python sg2_composite_analysis.py
python sg3_turnover.py
python sg4_cost_sensitivity.py
python sg5_fetch_stock_basic.py
python sg5_fetch_calendar.py
python sg5_execution_frictions.py
python sg6_long_only.py
```

Alpha101 试点：

```powershell
python -m alpha101_daily.run_pilot
```

华泰风格试点：

```powershell
python -m huatai_multifactor.fetch_fundamentals
python -m huatai_multifactor.run_pilot
```

## 评价方式

```text
训练期：2020–2022
验证期：2023–2025
```

主要指标：

- Rank IC 均值与 ICIR；
- 净年化收益；
- 净年化波动；
- 净夏普；
- 最大回撤；
- 日均换手率；
- 扣费后净值曲线。

## 局限性

- 仅使用当前沪深 300 成分股，存在幸存者偏差；
- 未使用历史成分股和退市股票；
- 行业分类、行业中性化和市值中性化尚未加入；
- 基本面数据为年频，尚未升级到季度/TTM；
- 因子库属于公开公式和常见分类的研究型实现，不是平台原版；
- 未建模融券借券费、融资成本、冲击成本和组合容量；
- 未处理所有涨跌停排队、成交量和最低交易单位限制；
- 结果仅用于学习与研究，不构成投资建议。

## 后续计划

- 引入历史指数成分股与全市场股票池；
- 增加行业、市值和 Beta 中性化；
- 扩展季度/TTM 基本面因子；
- 做周度、月度调仓和换手率约束；
- 对候选因子做相关性和稳定性筛选；
- 增加更长样本和滚动样本外测试；
- 建立 notebook 展示和完整实验报告。

## 说明

本项目用于学习和研究，不应作为真实投资决策依据。第三方数据与因子思想的使用需遵守相应平台和研究机构的许可与使用条款。
