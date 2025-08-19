A Modified CTGAN-Plus-Features Based Method for Optimal Asset Allocation
Authors: José-Manuel Peña, Fernando Suárez, Omar Larrea, Domingo Ramírez, Arturo Cifuentes
Affiliations:
a Fintual Administradora General de Fondos S.A. Santiago, Chile. Fintual, Inc.
b Clapes UC, Pontificia Universidad Católica de Chile, Santiago, Chile.
Contact: research@fintual.com
Article history: Compiled May 17, 2024
arXiv:2302.02269v3 [q-fin.PM] 15 May 2024

Abstract
We propose a new approach to portfolio optimization that combines synthetic data generation with a CVaR-constraint. We formulate the problem as asset allocation across asset classes via passive (index) funds. Asset-class weights are determined by solving an optimization problem with a CVaR-constraint, using a Modified CTGAN algorithm that incorporates features (contextual information) to generate synthetic return scenarios, which feed the optimization engine. Context features are points along the U.S. Treasury yield curve. We demonstrate merits on ten asset classes (stocks, bonds, commodities) over January 2008–June 2022. The synthetic generator captures key characteristics of original data, and the optimization yields satisfactory out-of-sample performance, outperforming equal-weights (1/N) and historical-data-only formulations.

Keywords: Asset allocation; Portfolio optimization; Portfolio selection; Synthetic data; Synthetic returns; Machine learning; Features; Contextual information; GAN; CTGAN; neural networks

1. Motivation and Previous Work
The portfolio selection problem concerns allocating capital among investment options. Markowitz (1952) reframed the problem as quantitative optimization, highlighting diversification, risk–return tradeoff, and the efficient frontier. However, practical challenges in estimating the correlation matrix and reliance on standard deviation as a risk proxy limit mean-variance (MV) implementation. Practitioners have moved beyond classical MV due to stability, estimation error, and operational concerns.

Passive investing, popularized by John Bogle (1975), shifted emphasis from asset selection to asset allocation via index funds, reducing dimensionality and costs. Asset allocation across broad markets (e.g., U.S. stocks, EM stocks, HY bonds, commodities) becomes more tractable.

Modern advances include: CVaR as a preferred risk metric capturing tail risk and compatible with general distributions; synthetic data for stochastic optimization (GANs); regime-aware modeling; and the use of features/context to improve out-of-sample optimization.

We propose an asset allocation method with annual rebalancing, CVaR-based risk control, and synthetic returns generated via a Modified Conditional GAN with contextual features (U.S. Treasury yield curve). We detail the problem, the synthetic data generation, and present a numerical example.

2. Problem Formulation
Let n be the number of asset classes, each accessed via a price index. The decision vector x ∈ R^n denotes long-only weights (sum to 1). Returns r ∈ R^n have density π(r). The objective is to maximize expected return subject to a CVaR constraint at confidence α and tolerance Λ.

The expected return is:

𝐸
(
𝑥
⊤
𝑟
)
=
∑
𝑖
=
1
𝑛
𝑥
𝑖
𝐸
[
𝑟
𝑖
]
.
E(x 
⊤
 r)= 
i=1
∑
n
​
 x 
i
​
 E[r 
i
​
 ].
Optimization:

maximize E(x^\top r)
subject to CVaR_α(x^\top r) ≤ Λ, ∑_{i=1}^n x_i = 1, x ≥ 0.
CVaR at level α:

C
V
a
R
𝛼
(
𝑋
)
=
1
1
−
𝛼
∫
0
1
−
𝛼
V
a
R
𝛾
(
𝑋
)
 
𝑑
𝛾
,
CVaR 
α
​
 (X)= 
1−α
1
​
 ∫ 
0
1−α
​
 VaR 
γ
​
 (X)dγ,
with

V
a
R
𝛾
(
𝑋
)
=
−
inf
⁡
{
𝑥
∈
𝑅
∣
𝐹
𝑋
(
𝑥
)
>
1
−
𝛾
}
.
VaR 
γ
​
 (X)=−inf{x∈R∣F 
X
​
 (x)>1−γ}.
2.1. Discretization and linearization
Equivalent continuous formulation (Rockafellar and Uryasev, 2000):

minimize −E(x^\top r) over (x, ζ)
subject to 
𝜁
+
1
1
−
𝛼
∫
[
−
𝑥
⊤
𝑟
−
𝜁
]
+
𝜋
(
𝑟
)
 
𝑑
𝑟
≤
Λ
ζ+ 
1−α
1
​
 ∫[−x 
⊤
 r−ζ] 
+
​
 π(r)dr≤Λ, ∑ x_i = 1, x ≥ 0.
Discrete scenarios r_j, probabilities π_j:

minimize −E(x^\top r)
subject to 
𝜁
+
1
1
−
𝛼
∑
𝑗
=
1
𝑚
[
−
𝑥
⊤
𝑟
𝑗
−
𝜁
]
+
𝜋
𝑗
≤
Λ
ζ+ 
1−α
1
​
 ∑ 
j=1
m
​
 [−x 
⊤
 r 
j
​
 −ζ] 
+
​
 π 
j
​
 ≤Λ, ∑ x_i = 1, x ≥ 0.
Linear program with auxiliaries z_j and scenario matrix R ∈ R^{n×m}:

maximize 
𝑥
⊤
𝑅
𝜋
x 
⊤
 Rπ
subject to 
𝜁
+
1
1
−
𝛼
𝜋
⊤
𝑧
≤
Λ
ζ+ 
1−α
1
​
 π 
⊤
 z≤Λ, 
𝑧
≥
−
𝑅
⊤
𝑥
−
𝜁
1
z≥−R 
⊤
 x−ζ1, ∑ x_i = 1, x, z ≥ 0.
Feature-weighted scenarios: with features F ∈ R^{l×m} and current feature vector f, define distances d(f, f_q), inverse weights 
𝑑
𝑓
−
1
[
𝑞
]
=
1
/
𝑑
(
𝑓
,
𝑓
𝑞
)
d 
f
−1
​
 [q]=1/d(f,f 
q
​
 ), normalized 
𝜋
𝑓
=
𝑑
𝑓
−
1
/
(
1
⊤
𝑑
𝑓
−
1
)
π 
f
​
 =d 
f
−1
​
 /(1 
⊤
 d 
f
−1
​
 ). Sample returns and features jointly.

3. Synthetic Data Generation
Finance offers only a single realized path from a non-stationary, unknown DGP. We generate realistic, regime-aware synthetic returns via CTGAN using recent data.

Given historical Dh with returns 
𝑅
ℎ
R 
h
  and features 
𝐹
ℎ
F 
h
  (m_h samples), train an SDG to produce Ds = [R^s, F^s] on demand.

3.1. CTGAN overview
CTGAN models mixed tabular data, conditioning continuous variables on discrete regimes. It combats class imbalance via a conditional generator and training-by-sampling, and uses mode-specific normalization (via VGM) for continuous variables. It learns complex joint structure and multiple regimes.

3.2. Modified CTGAN-plus-features
We add unsupervised regime detection and feed the regime label to CTGAN:

Start with 
𝐷
ℎ
=
[
𝑅
ℎ
,
𝐹
ℎ
]
D 
h
 =[R 
h
 ,F 
h
 ].
Orthogonalize via PCA to reduce trivial correlations; store eigenvectors for inverse transform.
Identify regimes: reduce dimensionality via t-SNE to 2D, then cluster with HDBSCAN to obtain discrete labels C.
Train CTGAN on PCA-transformed data using C as discrete column.
Generate m_s synthetic samples in PCA space.
Invert PCA to original space to obtain 
𝐷
𝑠
=
[
𝑅
𝑠
,
𝐹
𝑠
]
D 
s
 =[R 
s
 ,F 
s
 ].
4. Example of Application
We consider ten asset classes (indices) with daily data from January 2003 to June 2022, annual rebalancing, and a 5-year lookback. Indices: S&P 500 (SPX), Nasdaq 100 (NDX), MSCI World (MXWO), MSCI EM (MXEF), High Yield (IBOXHY), Investment Grade (IBOXIG), EM Debt (JPEIDIVR), Bloomberg Commodities (BCOMTR), Long-term Treasuries (I01303US), Short-term Treasuries (LT01TRUU).

4.1. Feature selection
We use eight U.S. Treasury curve tenors: Fed funds (0M), 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y (tickers FDTR, I02503M, I02506M, I02501Y, I02502Y, I02505Y, I02510Y, I02530Y). They provide timely, interpretable macro context.

4.2. SDGP validation
CTGAN trained on 2017–2022 data; synthetic vs. original comparisons:

Visual pair plots suggest close match in returns and features.
KS-tests on marginals: high similarity; average complement score ~0.87 (most >0.82).
Correlation similarity across pairs: high (lowest ~0.83), preserving joint structure.
Cluster-level validation: 44 clusters; synthetic cluster frequencies correlate 97.2% with original; within-cluster KS scores highest when matching clusters.
Conclusion: Modified CTGAN preserves marginal, joint, and regime characteristics.

4.3. Testing strategy
We evaluate five strategies:

(i) CTGAN without features (Gw/oF)
(ii) CTGAN with features (GwF)
(iii) Historical sampling without features (Hw/oF)
(iv) Historical sampling with features (HwF)
(v) Equal Weights (EW)
Annual rebalancing each January using previous 5 years’ data; out-of-sample evaluation one year forward, rolling from Jan 2009 to mid-2022. CVaR limits Λ from 7.5% to 30%. CTGAN runs are stochastic; each optimization is run 5 times per Λ. Without features, π_j = 1/m. With features, use π_f based on feature distances.

4.4. Performance metrics
We compare annualized cumulative returns, ex-post CVaR vs. limit, annual rotation:

rotation
=
∑
𝑡
=
2
14
∑
𝑖
=
1
10
∣
𝑤
𝑖
,
𝑡
−
𝑤
𝑖
,
𝑡
−
1
∣
,
rotation= 
t=2
∑
14
​

i=1
∑
10
​
 ∣w 
i,t
​
 −w 
i,t−1
​
 ∣,
and diversification via complementary Herfindahl–Hirschman (HH) Index.

4.5. Performance comparison
Environment: MacBook Pro (M1 Pro, 16 GB RAM), CPU only. Historical strategies: ~0.001 s per rebalance; CTGAN strategies: ~203.5 s. For each window, CTGAN generates 500 synthetic scenarios; historical strategies sample 500 historical scenarios; EW is static.

Key findings:

Returns: Features materially improve performance. GwF and HwF outperform non-feature versions, especially at higher CVaR limits. GwF generally > HwF; at CVaR 0.25, GwF 16.78% vs. HwF 15.65% annualized. EW underperforms all.
Ex-post CVaR: Feature-based approaches meet/stay below limits and often lower than non-feature versions; non-monotonicity occurs when constraint inactive.
Diversification: Similar across strategies; higher Λ reduces diversification as portfolios tilt to riskier assets; note each “asset” is an index, so portfolios are inherently diversified.
Trading costs: Estimated via bid–ask spreads and observed rotations; annualized cost impact is small, leaving returns essentially unchanged after costs.
Overall, GwF delivers the strongest returns with controlled risk and reasonable diversification/turnover.

4.6. Discussion and potential biases
Model selection: NORTA considered but computationally expensive with many indices+features; CopulaGAN not pursued due to limited evidence.
Overfitting: Minimal hyperparameter tuning; architecture near defaults (Xu et al., 2019), learning rate 1e−4, epochs 1500 for smoother convergence; no tuning for PCA, t-SNE, HDBSCAN beyond standard settings.
Lookback and rebalancing: Five-year lookback consistent with prior experience (3–5y). Shorter risks insufficient variability; longer risks non-stationarity. Annual rebalancing aligns with passive investing.
5. Conclusions
A Modified CTGAN-plus-features approach generates realistic, regime-aware synthetic returns and, combined with a CVaR-constrained linear optimization, yields portfolios with strong out-of-sample performance. Contextual features (yield curve) significantly improve outcomes. Synthetic scenarios include feasible but unobserved paths, improving robustness beyond purely historical sampling.

Open directions: broader features (volatility, liquidity, FX) and applying the synthetic generator to other financial variables (default rates, FX).

Data and Code
Code and data: https://github.com/chuma9615/ctgan-portfolio-research
Historical data source: Bloomberg.

Tables
Table 1. Indices Employed in the Asset Allocation Example
Asset Class	Bloomberg Ticker	Name
US Equities	SPX	S&P 500 Index
US Equities Tech	NDX	Nasdaq 100 Index
Global Equities	MXWO	Total Stock Market Index
EM Equities	MXEF	Emerging Markets Stock Index
High Yield	IBOXHY	High Yield Bonds Index
Investment Grade	IBOXIG	Liquid Investment Grade Index
EM Debt	JPEIDIVR	Emerging Markets Bond Index
Commodities	BCOMTR	Bloomberg Commodity Index
Long-term Treasuries	I01303US	Long-Term Treasury Index
Short-term Treasuries	LT01TRUU	Short-Term Treasury Index
Table 2. Features (Yield Curve Tenors)
Bloomberg Ticker	Maturity
FDTR	0 Months (Fed funds rate)
I02503M	3 Months
I02506M	6 Months
I02501Y	1 Year
I02502Y	2 Years
I02505Y	5 Years
I02510Y	10 Years
I02530Y	30 Years
Table 3. KS-Test: Original vs. Synthetic Distributions (Complement Scores)
Variable	KS-test	Variable	KS-test
US Equities	91.89%	Fed Funds Rate	89.21%
US Equities Tech	86.30%	3 Months Treasury	82.85%
Global Equities	94.52%	6 Months Treasury	82.58%
EM equities	92.66%	1 Year Treasury	84.44%
High Yield	93.53%	2 Years Treasury	86.41%
Investment Grade	85.87%	5 Years Treasury	84.61%
EM Debt	86.47%	10 Years Treasury	85.87%
Commodities	76.61%	30 Years Treasury	85.21%
Long-term Treasuries	88.11%	Short-term Treasuries	80.55%
Table 4. Trading Expenses by Asset Class (Avg 30-Day Bid–Ask Spread, bps)
Asset Class	Selected ETF	Spread (bps)
US equities	SPY US	0.36
US equities tech	QQQ US	0.52
Global equities	VT US	0.54
EM equities	EEM US	2.69
US high yield	HYG US	1.35
US inv. grade	LQD US	0.96
EM debt	PCY US	5.66
Commodities	COMT US	14.1
Long-term treasuries	TLT US	1.03
Short-term treasuries	BIL US	1.25
Table 5. Annualized Transaction Expenses (bps)
CVaR	Gw/oF	GwF	Hw/oF	HwF	EW
0.075	0.54	1.32	0.19	1.52	0
0.10	0.43	1.50	0.23	1.72	0
0.125	0.44	1.20	0.23	1.64	0
0.15	0.47	1.61	0.23	1.82	0
0.175	0.53	1.31	0.24	1.76	0
0.20	0.47	1.46	0.25	1.56	0
0.225	0.49	0.99	0.28	1.49	0
0.25	0.53	1.51	0.28	1.30	0
0.275	0.45	0.80	0.30	1.07	0
0.30	0.45	0.85	0.27	0.93	0
Table 6. Annualized Returns (Net of Transaction Expenses)
CVaR	Gw/oF	GwF	Hw/oF	HwF	EW
0.075	12.53%	13.49%	12.90%	12.72%	7.89%
0.10	11.96%	13.28%	12.73%	12.96%	7.89%
0.125	12.46%	14.93%	13.04%	13.65%	7.89%
0.15	13.84%	15.41%	13.20%	14.01%	7.89%
0.175	12.94%	15.17%	14.04%	14.06%	7.89%
0.20	13.02%	15.20%	13.57%	14.69%	7.89%
0.225	12.51%	16.21%	13.26%	15.19%	7.89%
0.25	13.18%	16.76%	13.31%	15.64%	7.89%
0.275	13.60%	17.35%	13.59%	16.44%	7.89%
0.30	13.87%	17.76%	14.90%	16.63%	7.89%
Figure Placeholders
Insert your image files alongside this .md and update the filenames if needed.













References
Amenc, N., Martellini, L., et al. (2001). It’s time for asset allocation. Journal of Financial Transformation, 3, 77–88.
Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. Mathematical Finance, 9(3), 203–228.
Ban, G.-Y., & Rudin, C. (2019). The big data newsvendor: Practical insights from machine learning. Operations Research, 67(1), 90–108.
Bauer, M. D., Mertens, T. M., et al. (2018). Information in the yield curve about future recessions. FRBSF Economic Letter, 20, 1–5.
Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive analytics. Management Science, 66(3), 1025–1044.
Bogle, J. C. (2018). Stay the course: the story of Vanguard and the index revolution. John Wiley & Sons.
Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In PAKDD (pp. 160–172).
Chen, X., Owen, Z., Pixton, C., & Simchi-Levi, D. (2022). A statistical learning approach to personalization in revenue management. Management Science, 68(3), 1923–1937.
DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification. Review of Financial Studies, 22(5), 1915–1953.
Eckerli, F., & Osterrieder, J. (2021). Generative adversarial networks in finance: an overview. arXiv:2106.06364.
Elton, E. J., Gruber, M. J., & de Souza, A. (2019). Are passive funds really superior investments? Financial Analysts Journal, 75(3), 7–19.
Estrella, A., & Trubin, M. (2006). The yield curve as a leading indicator: Some practical issues. Current Issues in Economics and Finance, 12(5).
Evgenidis, A., Papadamou, S., & Siriopoulos, C. (2020). The yield spread’s ability to forecast economic activity. Journal of Business Research, 106, 221–232.
Fabozzi, F. J., Fabozzi, F. A., López de Prado, M., & Stoyanov, S. V. (2021). Asset management: Tools and issues. World Scientific.
Fahling, E. J., Steurer, E., Sauer, S., et al. (2019). Active vs. passive funds—German equity market. Journal of Financial Risk Management, 8(2), 73.
Friedman, D., Isaac, R. M., James, D., & Sunder, S. (2014). Risky curves: On the empirical failure of expected utility. Routledge.
Goodfellow, I. J., et al. (2014). Generative adversarial networks. arXiv:1406.2661.
Gutierrez, T., Pagnoncelli, B., Valladão, D., & Cifuentes, A. (2019). Asset allocation limits in DC pension schemes. Insurance: Mathematics and Economics, 86, 134–144.
Hamilton, J. D. (1988, 1989). Regime switching and macroeconomic time series. Journal of Economic Dynamics and Control; Econometrica.
Hu, Y., Kallus, N., & Mao, X. (2022). Fast rates for contextual linear optimization. Management Science.
Ibbotson, R. G. (2010). The importance of asset allocation. Financial Analysts Journal, 66(2), 18–20.
Kolm, P. N., Tütüncü, R., & Fabozzi, F. J. (2014). 60 years of portfolio optimization. European Journal of Operational Research, 234(2), 356–371.
Krokhmal, P., Uryasev, S., & Palmquist, J. (2002). Portfolio optimization with CVaR objective and constraints.
Kumar, R. R., Stauvermann, P. J., & Vu, H. T. T. (2021). Yield curve and economic activity: G7 countries. Journal of Risk and Financial Management, 14(2), 62.
Lommers, K., Harzli, O. E., & Kim, J. (2021). Confronting ML with financial research. Journal of Financial Data Science, 3(3), 67–96.
Lu, J., & Yi, S. (2022). Autoencoding conditional GAN for portfolio allocation diversification. arXiv:2207.05701.
Mariani, G., et al. (2019). PaGAN: Portfolio analysis with GANs. arXiv:1909.10578.
Markowitz, H. (1952). Portfolio selection. The Journal of Finance, 7(1), 77–91.
Massey, F. J. (1951). The Kolmogorov–Smirnov test for goodness of fit. JASA, 46(253), 68–78.
Pagnoncelli, B. K., Ramírez, D., Rahimian, H., & Cifuentes, A. (2022). Synthetic data-plus-features for portfolio optimization. Computational Economics.
Pflug, G. C. (2000). Remarks on VaR and CVaR. In Probabilistic Constrained Optimization (pp. 272–281).
Pun, C. S., Wang, L., & Wong, H. Y. (2020). GAN-based robust portfolio selection. IJCAI’20.
Rockafellar, R. T., & Uryasev, S. (2000, 2002). Optimization and properties of CVaR. Journal of Risk; Journal of Banking & Finance.
Schaller, H., & Norden, S. V. (1997). Regime switching in stock returns. Applied Financial Economics, 7(2), 177–191.
See, C.-T., & Sim, M. (2010). Robust approximation to multiperiod inventory management. Operations Research, 58(3), 583–594.
Sharpe, W. F. (1991). The arithmetic of active management. Financial Analysts Journal, 47(1), 7–9.
Takahashi, S., Chen, Y., & Tanaka-Ishii, K. (2019). Modeling financial time-series with GANs. Physica A, 527, 121261.
Thune, K. (2022). How and why John Bogle started Vanguard. thebalancemoney.com.
Tu, J., & Zhou, G. (2004). DGP uncertainty in portfolio decisions. Journal of Financial Economics, 72(2), 385–421.
Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. arXiv:1907.00503.