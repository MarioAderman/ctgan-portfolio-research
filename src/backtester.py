# Third party imports
import pandas as pd

# Local imports
from src.generators.historical_generator import HistoricalGenerator
from src.generators.gan_generator import CTGANGenerator
from src.metrics import compute_annualized_return, compute_cvar, compute_mean_hhi, compute_mean_rotation
from src.uryasev_optimization import UryasevOptimization
from src.utils import zscore_euclidean
from src.progress_display import HackerProgressDisplay


class Backtester():
    '''
    Entity responsible of backtests
    '''
    def __init__(self, asset_prices, asset_returns, config, rebalance_dates, features=None):
        self.asset_prices = asset_prices
        self.asset_returns = asset_returns
        self.config = config
        self.rebalance_dates = rebalance_dates
        self.features = features
        self.progress = HackerProgressDisplay()
        self.lookback_years = config['lookback_years']
        self.generators = self._instanciate_generators(config['model_names'])
        self.cvar = config['cvar']
        self.alpha = config['alpha']
        self.bounds = config['bounds']
        self.backtest_name = 'default'

    def run_backtests(self, save=False):
        '''
        Runs a backtest and saves it in a json file. The name is for the case the caller runs several backtests.
        '''

        # first we generate the samples for each rebalance date
        samples = self.generate_samples()

        # we compute the optimizations for each rebalance date and store the portfolio of each model for each date
        in_sample_portfolios = self.build_in_sample_portfolios(samples, self.rebalance_dates,  self.lookback_years, self.cvar, self.alpha, self.bounds)
        
        # we run the performance of the historical portfolios
        backtests  = self.backtest_portfolios(historical_portfolios=in_sample_portfolios)
        
        # we add some interesting metrics for analysis
        backtests = self.compute_metrics(backtests=backtests)

        return backtests

    def generate_samples(self):
        '''
        Generates the samples for each rebalance date and for each model.
        '''
        total_steps = len(self.rebalance_dates) * len(self.generators)
        self.progress.start_phase("SAMPLE GENERATION", total_steps)
        
        # for each date and model we generate samples and store them in a dictionary
        samples = {}
        for rebalance_date in self.rebalance_dates:
        
            start_date, end_date = self._get_start_end_dates(rebalance_date)
            samples[rebalance_date] = {}

            for generator in self.generators:
                self.progress.update_progress(
                    current_date=rebalance_date, 
                    model_name=generator.name,
                    sub_task="Generating scenarios"
                )
                sample = generator.generate_sample(sample_size=self.config['sample_size'],
                                                start_date=start_date,
                                                end_date=end_date)
                samples[rebalance_date][generator.name] = sample

        self.progress.complete_phase()
        return samples

    def _get_start_end_dates(self, rebalance_date):
        if self.features is None:
            end_date = rebalance_date
            start_date =  self.asset_returns[self.asset_returns.index<=end_date].resample('Y').last().index[-1*self.lookback_years-1]
        else:
            # this one is hard coded, won't work if we decide to change from 365 returns
            end_date = self.asset_returns[self.asset_returns.index<=rebalance_date].resample('Y').last().index[-2]
            start_date =  self.asset_returns[self.asset_returns.index<=end_date].resample('Y').last().index[-1*self.lookback_years-1]
        
        return start_date, end_date

    def build_in_sample_portfolios(self, samples, rebalance_dates, lookback_years, cvar, alpha, bounds):
        '''
        Given the samples, runs a uryasev optimisation for each rebalance date and model.
        '''
        total_steps = len(self.generators) * len(rebalance_dates)
        self.progress.start_phase("PORTFOLIO OPTIMIZATION", total_steps)

        # initialize optimitazion object
        uryasev_optimization = UryasevOptimization(alpha=alpha, cvar=cvar, bounds=bounds)
        portfolios = {}
        # for each date and model run an optimization problem
        for model in self.generators:
            portfolios[model.name] = {}
            model_portfolios = pd.DataFrame(columns=self.asset_returns.columns)
            for rebalance_date in rebalance_dates:
                self.progress.update_progress(
                    current_date=rebalance_date,
                    model_name=model.name,
                    sub_task="CVaR optimization"
                )
                sample = samples[rebalance_date][model.name]
                if self.features is not None:
                    density = self.compute_density(sample, rebalance_date)
                    sample_columns = self.asset_returns.columns.tolist() + self.features.columns.tolist()
                else:
                    density = None
                    sample_columns = self.asset_returns.columns.tolist()
                
                sample_assets = pd.DataFrame(sample, columns=sample_columns)[self.asset_returns.columns].values
                portfolio = uryasev_optimization.get_optimal_portfolio(sample=sample_assets, density=density)
                portfolio.index = self.asset_returns.columns
                model_portfolios.loc[rebalance_date] = portfolio
                portfolios[model.name] = model_portfolios

        self.progress.complete_phase()
        return portfolios
    
    def compute_density(self, sample, rebalance_date):
        columns = self.asset_returns.columns.tolist() + self.features.columns.tolist()
        sampled_features = pd.DataFrame(sample, columns=columns)[self.features.columns]
        spot_feature = self.features.loc[rebalance_date]
        # use zscore normalized euclidean
        distances = 1 / zscore_euclidean(spot_feature, sampled_features)
        density = distances / distances.sum()
        
        
        return density.values


    def backtest_portfolios(self, historical_portfolios):
        '''
        Given a historical portfolio and the total returns, computes the performance backtest.
        Uses proper annual rebalancing: calculates returns between rebalancing periods.
        '''
        total_steps = len(self.generators)
        self.progress.start_phase("PERFORMANCE CALCULATION", total_steps)
        
        backtests = {}
        for model in self.generators:
            self.progress.update_progress(
                model_name=model.name,
                sub_task="Computing returns"
            )
            backtests[model.name] = {}
            portfolios = historical_portfolios[model.name]
            
            # Convert portfolio weights from percentage to fractional
            portfolios_frac = portfolios / 100
            
            # Calculate portfolio performance using period returns between rebalancing dates
            portfolio_values = pd.Series(index=portfolios.index, dtype=float)
            portfolio_values.iloc[0] = 100  # Starting value of 100
            
            for i in range(len(portfolios.index) - 1):
                current_date = portfolios.index[i]
                next_date = portfolios.index[i + 1]
                
                # Get asset prices at start and end of period
                start_prices = self.asset_prices.loc[current_date]
                end_prices = self.asset_prices.loc[next_date]
                
                # Calculate period returns for each asset
                period_returns = (end_prices / start_prices) - 1
                
                # Apply portfolio weights to get portfolio return for this period
                portfolio_weights = portfolios_frac.loc[current_date]
                portfolio_return = (portfolio_weights * period_returns).sum()
                
                
                # Calculate portfolio value at end of period
                portfolio_values.iloc[i + 1] = portfolio_values.iloc[i] * (1 + portfolio_return)
            
            backtests[model.name]['total_return_serie'] = portfolio_values
            backtests[model.name]['portfolios'] = portfolios
        
        self.progress.complete_phase()
        return backtests
    
    def compute_metrics(self, backtests):
        '''
        Calculates some ex post metrics.
        '''
        for model in self.generators:
            serie = backtests[model.name]['total_return_serie']
            portfolios = backtests[model.name]['portfolios']
            backtests[model.name]['annualized_return'] = compute_annualized_return(serie)
            backtests[model.name]['cvar_expost'] = compute_cvar(serie, tf=self.config['returns_timeframe'])
            backtests[model.name]['mean_hhi'] = compute_mean_hhi(portfolios)
            backtests[model.name]['mean_rotation'] = compute_mean_rotation(portfolios)
        
        return backtests

    def _instanciate_generators(self, model_names):
        generators = []
        if 'historical' in model_names:
            historical_generator = HistoricalGenerator(asset_returns=self.asset_returns, features=self.features)
            generators.append(historical_generator)
        if 'CTGAN' in model_names:
            ctgan_generator = CTGANGenerator(asset_returns=self.asset_returns, features=self.features)
            generators.append(ctgan_generator)

        return generators