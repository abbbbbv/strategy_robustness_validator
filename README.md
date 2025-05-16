# strategy_robustness_validator
A versatile validation and robustness testing toolkit for algorithmic trading strategies.

## Description

Strategy-Robustness-Toolkit is a modular and extensible framework designed to rigorously validate and backtest any algorithmic trading strategy. The toolkit supports multiple robustness tests including train/test splits, walk-forward analysis, parameter stability checks, Monte-Carlo trade order shuffling, bootstrap confidence intervals, and advanced statistical measures like the Deflated Sharpe Ratio and CPCV-PBO.

While this repository includes an example RSI-based strategy applied to relative price data, the core framework is strategy-agnostic and can be adapted easily to other strategies by replacing the strategy class and relevant parameters.

## Features

- Flexible integration of custom trading strategies  
- Supports historical data loading and preprocessing (Binance Futures example provided)  
- Implements train/test split and walk-forward rolling optimization  
- Parameter robustness heatmaps and neighborhood stability analysis  
- Monte-Carlo simulations for randomness robustness  
- Bootstrap confidence intervals for CAGR and other metrics  
- Advanced overfitting detection via CPCV-PBO and Deflated Sharpe Ratio  
- Optional SPA / White Reality Check for rigorous multiple-testing correction  
- Minimal dependencies: only common data science Python packages required  

## Installation

```bash
pip install numpy pandas scipy matplotlib seaborn pandas_ta backtesting tqdm python-binance
