# CTGAN Portfolio Optimization - Extended Research

**An extended investigation and implementation of CTGAN-based portfolio optimization methodologies.**

## About This Project

This repository represents a further investigation and enhancement of the CTGAN portfolio optimization approach, building upon the foundational work and methodologies established by the original research team.

### Original Research Credit

**Original Authors**: Arturo Cifuentes, Domingo Ramirez & Fintual AGF

**Original Repository**: [chuma9615/ctgan-portfolio-research](https://github.com/chuma9615/ctgan-portfolio-research)

**Original Research Resources**:
- **Colab Notebook**: [Running Example](https://colab.research.google.com/drive/1MGgLu66FfDMpFQjNZuOK6z13eFikuz8x)
- **Research Papers**:
  - [arXiv Preprint](https://arxiv.org/abs/2302.02269)
  - [ResearchGate Publication](https://www.researchgate.net/publication/368288208_A_Modified_CTGAN-Plus-Features_Based_Method_for_Optimal_Asset_Allocation)
- **Article**: [Fintual Research Explanation](https://fintualist.com/chile/alpha/primer-articulo-de-investigacion-en-fintual-inversiones/?utm_source=intercom&utm_medium=email&utm_campaign=fintualist112)

### Enhanced Implementation Features

This extended research includes several key improvements and investigations:

- **CVaR Constraint Optimization**: Deep investigation into Conditional Value at Risk constraint calibration and its impact on portfolio diversification
- **Enhanced Visualizations**: Comprehensive portfolio performance dashboards with allocation tracking and concentration analysis
- **Performance Improvements**: Optimized backtesting methodology with proper annual rebalancing and feature-weighted scenario generation
- **Hacker-Style Progress Display**: Professional terminal interface with real-time progress tracking, ETA calculations, and colorized output
- **Modular Architecture**: Clean separation of concerns with dedicated generators, optimization, and visualization modules
- **Comprehensive Metrics**: HHI concentration index, portfolio rotation metrics, and ex-post CVaR analysis

### Research Focus

This investigation particularly focuses on:
1. **Portfolio Concentration Issues**: Solving extreme allocation concentration through proper CVaR constraint calibration
2. **Methodology Validation**: Comparing implementation results against paper benchmarks
3. **CTGAN vs Historical Sampling**: Detailed comparative analysis of synthetic vs bootstrap sampling approaches
4. **Feature Integration**: Optimal use of yield curve features for scenario weighting

## Setup and Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.8)
- conda (recommended) or virtualenv
- CUDA-capable GPU (optional, for accelerated training)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ctgan-portfolio-research.git
   cd ctgan-portfolio-research
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n ctgan-portfolio python=3.8
   conda activate ctgan-portfolio
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

**Execute the main optimization pipeline**:
```bash
python main.py
```

This will:
- Load and preprocess asset price data
- Train the CTGAN model for synthetic scenario generation
- Run portfolio optimization with CVaR constraints
- Generate comprehensive visualizations and performance metrics
- Output results with hacker-style progress tracking

### Project Structure

- `main.py` - Main execution script
- `src/` - Core implementation modules
- `src/data/` - Data files and preprocessing
- `src/generators/` - CTGAN and historical data generators
- `requirements.txt` - Python dependencies

---

*This project extends and builds upon the excellent foundational research by Cifuentes, Ramirez & Fintual AGF. All credit for the original CTGAN portfolio optimization methodology goes to the original research team.* 
