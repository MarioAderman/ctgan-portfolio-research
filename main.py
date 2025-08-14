# Standard library imports
import json

# Local application imports
from src.backtester import Backtester
from src.utils import load_data

config = json.load(open("./config.json"))
asset_prices, asset_returns, features, rebalance_dates = load_data(config)

backtester = Backtester(
                        asset_prices=asset_prices,
                        asset_returns=asset_returns,
                        config=config,
                        rebalance_dates=rebalance_dates,
                        features=features)

backtests = backtester.run_backtests()

# Print the results
for model_name, results in backtests.items():
    print(f"--- Results for {model_name} ---")
    print("Performance Metrics:")
    print(f"  Annualized Return: {results['annualized_return']:.2%}")
    print(f"  CVaR (Ex-post): {results['cvar_expost']:.2%}")
    print(f"  Mean HHI: {results['mean_hhi']:.4f}")
    print(f"  Mean Rotation: {results['mean_rotation']:.4f}")
    print("\nAsset Allocations (Portfolios):")
    print(results['portfolios'])
    print("\n")
