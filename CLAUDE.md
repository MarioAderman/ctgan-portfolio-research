# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring portfolio optimization using CTGAN (Conditional Tabular Generative Adversarial Networks) for synthetic data generation. The project compares traditional historical sampling with GAN-generated synthetic data for portfolio optimization using CVaR (Conditional Value at Risk) methodology.

## Environment Setup

```bash
# Activate the project-specific conda environment
zsh -c "source ~/.zshrc && conda activate ctgan-portfolio"
```

## Core Architecture

The project follows a modular architecture with these key components:

- **Entry Point**: `main.py` - Simple orchestration that loads config, data, and runs backtests
- **Configuration**: `config.json` - All parameters including model selection, risk parameters (alpha, CVaR), data paths, and optimization bounds
- **Backtester**: `src/backtester.py` - Main orchestration class that coordinates sample generation, portfolio optimization, and performance evaluation
- **Data Generators**: Two approaches in `src/generators/`:
  - `historical_generator.py` - Bootstrap sampling from historical data
  - `gan_generator.py` - CTGAN-based synthetic data generation with clustering and feature matching
- **Optimization**: `src/uryasev_optimization.py` - Implements Uryasev & Rockafellar CVaR optimization using linear programming
- **Utilities**: `src/utils.py` - Data loading, Z-score distance calculations, file operations

## Key Data Flow

1. **Data Loading**: Load asset prices and features from CSV files in `src/data/`
2. **Sample Generation**: For each rebalance date, generate samples using selected model(s)
3. **Portfolio Optimization**: Apply CVaR optimization to samples to get optimal weights
4. **Backtesting**: Evaluate portfolio performance over time
5. **Metrics Computation**: Calculate annualized returns, CVaR, HHI concentration, portfolio rotation

## Configuration System

The `config.json` file controls all aspects of the experiment:
- `model_names`: Array of generators to use ("historical", "CTGAN")
- `assets_path`/`features_path`: Data file locations
- `alpha`: Confidence level for CVaR (typically 0.95)
- `cvar`: Target CVaR constraint (typically 0.01-0.2)
- `sample_size`: Number of scenarios to generate (default 500)
- `lookback_years`: Historical window for training (default 5)
- `returns_timeframe`: Return calculation period in days (default 365)

## Common Commands

```bash
# Activate environment and run the main backtest
zsh -c "source ~/.zshrc && conda activate ctgan-portfolio && python main.py"
```

## Development Notes

- The project uses relative imports (`from src.generators.historical_generator import HistoricalGenerator`)
- Data files are expected in `src/data/` directory (asset_prices.csv, features.csv)
- The GAN generator uses SDV's CTGAN implementation with HDBSCAN clustering for feature matching
- Portfolio optimization uses scipy's linprog for the Uryasev CVaR formulation
- Results can be saved/loaded via config flags (`read_backtest`, `read_samples`)

## Research Context

This implements the methodology described in the paper "A Modified CTGAN-Plus-Features Based Method for Optimal Asset Allocation" (arXiv:2302.02269). The core innovation is using GAN-generated synthetic scenarios for portfolio optimization rather than traditional historical bootstrap sampling.