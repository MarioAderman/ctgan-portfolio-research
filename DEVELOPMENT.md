# DEVELOPMENT.md

This file provides detailed technical insights, debugging strategies, and optimization lessons learned during development of the CTGAN portfolio optimization system.

## Code Lessons Learned - Technical Insights

### 🔍 Critical Architecture Understanding

**1. Data Flow Pipeline:**
```
asset_prices.csv → pct_change(returns_timeframe) → generators → optimization → backtesting
```
- `returns_timeframe` in config controls whether daily (1) or annual (365) returns are calculated
- **Lesson**: This single parameter affects the entire pipeline - always check it first

**2. Portfolio Backtesting Logic:**
- **Wrong approach**: Daily compounding with annual weights → extreme returns
- **Correct approach**: Period-based returns between rebalancing dates → realistic returns
- **Key fix**: `backtest_portfolios()` method now calculates returns properly

### ⚠️ Common Pitfalls & Solutions

**1. Percentage Formatting Bug:**
```python
# WRONG (double percentage)
print(f"{value:.2%}")  # if value = 8.5, prints 850.00%

# CORRECT 
print(f"{value:.2f}%")  # if value = 8.5, prints 8.50%
```

**2. CVaR Configuration Logic:**
- **Higher CVaR limit = Higher allowed risk = Higher potential returns**
- `cvar: 0.01` = very conservative, `cvar: 0.30` = aggressive
- **Lesson**: Always test CVaR sensitivity - biggest performance lever

**3. CTGAN Training Time:**
- **10 epochs**: ~30-60 seconds per period
- **300 epochs**: ~2-3 minutes per period  
- **1500 epochs** (paper): ~5-10 minutes per period
- **Lesson**: Use reduced epochs for testing, full epochs for production

### 🏗️ Code Structure Insights

**1. Config-Driven Architecture:**
```json
{
  "model_names": ["historical", "CTGAN"],     // Controls which models run
  "sample_size": 100,                         // Scenarios per optimization
  "lookback_years": 3,                        // Historical window
  "cvar": 0.30,                              // Risk tolerance (key lever)
  "returns_timeframe": 365                    // Daily vs annual returns
}
```

**2. Generator Pattern:**
- Both `HistoricalGenerator` and `CTGANGenerator` implement same interface
- Easy to add new models by implementing `generate_sample()` method
- **Lesson**: Clean abstraction makes experimentation smooth

**3. Feature Integration:**
- Features automatically included if `use_features: true`
- Normalizer handles feature preprocessing (expects `f_` prefix)
- **Lesson**: Feature integration is seamless once data format is correct

### 🚀 Performance Optimization Patterns

**1. Testing vs Production Settings:**
```json
// Testing (fast)
"lookback_years": 3, "sample_size": 100, "epochs": 10

// Production (accurate)  
"lookback_years": 5, "sample_size": 500, "epochs": 1500
```

**2. Model Comparison Strategy:**
- Test with single model first (faster iteration)
- Add models incrementally once baseline works
- **Lesson**: Validate one model thoroughly before comparing multiple

**3. Git Workflow:**
- Commit core fixes separately from optimizations
- Keep production vs testing configs separate
- **Lesson**: Clean commits make rollback easier during experimentation

### 🎯 Debugging Strategies

**1. Return Calculation Issues:**
- Add debug prints to `backtest_portfolios()` method
- Check intermediate values: period returns should be ~5-15% annually
- **Lesson**: Visual inspection of portfolio values catches logic errors

**2. CTGAN Issues:**
- Check data format (features with `f_` prefix)
- Monitor training progress (add print statements)
- Start with minimal epochs for proof-of-concept
- **Lesson**: Verify data pipeline before optimizing hyperparameters

**3. Configuration Validation:**
- Always check `git diff` before major runs
- Verify CVaR direction (higher = more risk allowed)
- **Lesson**: Config mistakes waste hours - double-check critical parameters

## Performance Optimization Results

### CVaR Sensitivity Analysis

| CVaR Limit | Annualized Return | Improvement | Portfolio Behavior |
|------------|------------------|-------------|-------------------|
| 1% (0.01) | 6.58% | Baseline | Very defensive |
| 25% (0.25) | 8.46% | +28% | Moderate risk |
| 30% (0.30) | **9.26%** | **+41%** | Aggressive |

**Key Finding**: CVaR tuning provides the largest performance improvement lever.

### Model Performance Comparison

**Current Results (3-year lookback, 100 samples):**
- **Historical + CVaR 30%**: 9.26% annualized return
- **CTGAN + CVaR 1%**: 5.87% annualized return
- **Paper Target**: 15-17% annualized return

**Gap Analysis**: ~6 percentage points remaining to reach paper performance levels.

## Quick Wins for Future Optimization

1. **CVaR tuning** = biggest performance lever (41% improvement demonstrated)
2. **Feature integration** = paper showed ~2 percentage point gains  
3. **Production settings** = likely 2-4 point improvement from better data quality
4. **CTGAN hyperparameters** = potential 1-2 point gains from tuning

## Critical Bug Fixes Applied

### 1. Return Calculation Fix
**Problem**: Daily returns compounded over annual periods → 800%+ unrealistic returns  
**Solution**: Period-based returns between rebalancing dates  
**Impact**: Reduced returns from 800%+ to realistic 6-9% range

### 2. Percentage Display Fix
**Problem**: Double percentage formatting in output display  
**Solution**: Changed `.2%` to `.2f%` formatting  
**Impact**: Correct display of actual calculated values

### 3. CVaR Calculation Fix  
**Problem**: CVaR calculated on daily changes instead of period returns  
**Solution**: Updated `compute_cvar()` for annual rebalancing periods  
**Impact**: Realistic CVaR values aligned with portfolio behavior

## Technical Architecture Notes

### Data Processing Pipeline
1. **Asset Prices**: Daily prices normalized to 100 at start
2. **Returns Calculation**: Annual returns using `pct_change(365)`  
3. **Feature Processing**: Yield curve data with `f_` prefix normalization
4. **Sample Generation**: Historical bootstrap or CTGAN synthetic scenarios
5. **Optimization**: Uryasev CVaR linear programming
6. **Backtesting**: Period-based portfolio returns

### CTGAN Implementation Details
- **PCA preprocessing**: Orthogonalizes data to reduce trivial correlations
- **t-SNE reduction**: 2D embedding for clustering
- **HDBSCAN clustering**: Automatic regime detection
- **Conditional training**: Uses cluster labels as discrete conditioning variable
- **Feature integration**: Seamless yield curve context inclusion

### Configuration Parameters Impact
- `returns_timeframe`: Controls daily vs annual return calculations
- `cvar`: Primary performance tuning parameter
- `sample_size`: Quality vs speed tradeoff  
- `lookback_years`: Historical window for training
- `epochs`: CTGAN training quality vs time

This foundation enables rapid optimization and debugging for future development work.

## Session 2025-08-20: Bounds Constraint Investigation

### 🔍 Critical Discovery: Paper vs Implementation Gap

**Key Finding**: The paper's optimization formulation **does not include upper bounds constraints**, while our implementation uses `bounds: [0.0, 0.20]`.

**Paper Formulation:**
```
maximize E(x^T r)
subject to CVaR_α(x^T r) ≤ Λ, Σx_i = 1, x ≥ 0
```

**Our Implementation:**
```
Same + additional constraint: x_i ≤ 0.20 (20% max per asset)
```

### 📊 Bounds Impact Analysis

**Bounded Results (20% limits):**
- Historical: 11.53% return, HHI 0.8941 (concentrated but spread across 5+ assets)
- CTGAN: 10.58% return, HHI 0.8927
- **Pattern**: Artificial uniform distribution hitting bounds exactly (20% allocations)

**Unbounded Results (paper methodology):**
- Historical: 16.65% return, HHI 0.1765 (extreme concentration)
- CTGAN: 15.58% return, HHI 0.2084  
- **Pattern**: Natural concentration (95.76% in single asset in one period)

### 🎯 Root Cause Analysis

**Bounds Constraint Evolution:**
1. **Initial**: `[0.0, 1.0]` (essentially unbounded)
2. **"Fix"**: Changed to `[0.0, 0.20]` to prevent extreme concentration
3. **Result**: Created artificial "uniform distribution to bounds" pattern

**The Trade-off:**
- **With bounds**: Reasonable diversification, lower performance vs paper
- **Without bounds**: Paper-level performance, extreme concentration risk

### 🛠️ Technical Improvements

**Visualization System Enhanced:**
- Modified `src/visualization.py` to save plots without displaying when `create_visualizations: true`
- Added `plt.close()` for memory management
- Charts now save to `./charts/` directory automatically

**Configuration Alignment:**
- Upgraded to production settings: 5-year lookback, 500 samples
- Both Historical and CTGAN models now running simultaneously
- CVaR constraint at 10% (balanced between paper's 7.5%-30% range)

### 🔬 Next Investigation Priorities

1. **Implementation Deep-dive**: Identify what paper authors may have omitted
2. **Alternative Diversification**: Explore entropy penalties or other mechanisms  
3. **Feature Integration**: Verify yield curve weighting matches paper methodology
4. **Post-processing Analysis**: Check impact of small position removal (lines 78-81 in uryasev_optimization.py)

**Status**: Successfully reproduced paper-level performance but with concentration concerns that require further investigation of implementation details.