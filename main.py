# Standard library imports
import json
import warnings

# Suppress specific deprecation warnings from external libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="rdt")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Local application imports
from src.backtester import Backtester
from src.utils import load_data
from src.visualization import PortfolioVisualizer
from src.progress_display import HackerProgressDisplay

# Initialize progress display and show header
progress = HackerProgressDisplay()
progress.print_header()

config = json.load(open("./config.json"))
asset_prices, asset_returns, features, rebalance_dates = load_data(config)

backtester = Backtester(
                        asset_prices=asset_prices,
                        asset_returns=asset_returns,
                        config=config,
                        rebalance_dates=rebalance_dates,
                        features=features)

backtests = backtester.run_backtests()

# Print formatted results using progress display
progress.print_results_header()

for model_name, results in backtests.items():
    metrics = {
        'return': f"{results['annualized_return']:.2f}%",
        'cvar': f"{results['cvar_expost']:.2f}%", 
        'hhi': f"{results['mean_hhi']:.4f}",
        'rotation': f"{results['mean_rotation']:.4f}"
    }
    progress.print_model_results(model_name, metrics)

# Create visualizations if enabled
if config.get('create_visualizations', False):
    asset_names = asset_prices.columns.tolist()
    visualizer = PortfolioVisualizer(backtests, asset_names)

    # Create comprehensive dashboard
    visualizer.create_summary_dashboard(save_path="./charts")
    progress.print_visualization_status("./charts")

# Print completion footer
progress.print_footer()
