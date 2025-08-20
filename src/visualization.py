import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

class PortfolioVisualizer:
    """
    Creates visualizations for portfolio optimization results
    """
    
    def __init__(self, backtests, asset_names):
        self.backtests = backtests
        self.asset_names = asset_names
        
    def plot_portfolio_allocations(self, model_name, save_path=None):
        """
        Create a stacked area chart showing portfolio allocations over time
        """
        portfolios = self.backtests[model_name]['portfolios']
        
        plt.figure(figsize=(14, 8))
        
        # Create stacked area plot
        plt.stackplot(portfolios.index, 
                     *[portfolios[col]/100 for col in portfolios.columns], 
                     labels=portfolios.columns,
                     alpha=0.7)
        
        plt.title(f'Portfolio Allocations Over Time - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Weight', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/portfolio_allocations_{model_name}.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_cumulative_returns(self, save_path=None):
        """
        Plot cumulative returns for all models
        """
        plt.figure(figsize=(12, 8))
        
        for model_name in self.backtests.keys():
            returns_series = self.backtests[model_name]['total_return_serie']
            plt.plot(returns_series.index, returns_series.values, 
                    linewidth=2, label=f'{model_name} Portfolio')
        
        plt.title('Cumulative Portfolio Returns Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/cumulative_returns.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_risk_metrics_comparison(self, save_path=None):
        """
        Create bar charts comparing risk metrics across models
        """
        models = list(self.backtests.keys())
        
        # Extract metrics
        returns = [self.backtests[model]['annualized_return'] for model in models]
        cvars = [self.backtests[model]['cvar_expost'] for model in models]
        hhis = [self.backtests[model]['mean_hhi'] for model in models]
        rotations = [self.backtests[model]['mean_rotation'] for model in models]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Annualized Returns
        bars1 = ax1.bar(models, returns, color='skyblue', alpha=0.7)
        ax1.set_title('Annualized Returns (%)', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{returns[i]:.1f}%', ha='center', va='bottom')
        
        # CVaR
        bars2 = ax2.bar(models, cvars, color='salmon', alpha=0.7)
        ax2.set_title('CVaR Ex-post (%)', fontweight='bold')
        ax2.set_ylabel('CVaR (%)')
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{cvars[i]:.1f}%', ha='center', va='bottom')
        
        # HHI (Diversification)
        bars3 = ax3.bar(models, hhis, color='lightgreen', alpha=0.7)
        ax3.set_title('Mean HHI (Diversification)', fontweight='bold')
        ax3.set_ylabel('HHI (0=concentrated, 1=diversified)')
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{hhis[i]:.3f}', ha='center', va='bottom')
        
        # Portfolio Rotation
        bars4 = ax4.bar(models, rotations, color='gold', alpha=0.7)
        ax4.set_title('Mean Portfolio Rotation', fontweight='bold')
        ax4.set_ylabel('Rotation')
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{rotations[i]:.1f}', ha='center', va='bottom')
        
        plt.suptitle('Portfolio Performance Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/risk_metrics_comparison.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_allocation_heatmap(self, model_name, save_path=None):
        """
        Create a heatmap showing asset allocations over time using matplotlib
        """
        portfolios = self.backtests[model_name]['portfolios']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap using matplotlib imshow
        data = portfolios.T.values
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Allocation (%)', rotation=270, labelpad=15)
        
        # Set ticks and labels
        ax.set_xticks(range(len(portfolios.index)))
        ax.set_xticklabels([d.strftime('%Y-%m') for d in portfolios.index], rotation=45)
        ax.set_yticks(range(len(portfolios.columns)))
        ax.set_yticklabels(portfolios.columns)
        
        # Add text annotations
        for i in range(len(portfolios.columns)):
            for j in range(len(portfolios.index)):
                text = ax.text(j, i, f'{data[i, j]:.1f}', 
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.title(f'Asset Allocation Heatmap - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Rebalance Date', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/allocation_heatmap_{model_name}.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_equal_weight_comparison(self, save_path=None):
        """
        Compare actual portfolio vs equal weight benchmark
        """
        plt.figure(figsize=(12, 8))
        
        for model_name in self.backtests.keys():
            returns_series = self.backtests[model_name]['total_return_serie']
            plt.plot(returns_series.index, returns_series.values, 
                    linewidth=2, label=f'{model_name} Portfolio')
        
        # Add equal weight benchmark (if we calculate it)
        # For now, just add a horizontal line at equal weight level
        equal_weight = 100 / len(self.asset_names)
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, 
                   label=f'Equal Weight Baseline (10% each)')
        
        plt.title('Portfolio Performance vs Equal Weight Baseline', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/equal_weight_comparison.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def create_summary_dashboard(self, save_path=None):
        """
        Create a comprehensive dashboard with all key visualizations
        """
        print("Creating Portfolio Analysis Dashboard...")
        print("=" * 50)
        
        # Create plots directory if specified
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
        
        # 1. Risk metrics comparison
        print("📊 Risk Metrics Comparison")
        self.plot_risk_metrics_comparison(save_path)
        
        # 2. Cumulative returns
        print("📈 Cumulative Returns")
        self.plot_cumulative_returns(save_path)
        
        # 3. Portfolio allocations for each model
        for model_name in self.backtests.keys():
            print(f"🎯 Portfolio Allocations - {model_name}")
            self.plot_portfolio_allocations(model_name, save_path)
            
            print(f"🔥 Allocation Heatmap - {model_name}")
            self.plot_allocation_heatmap(model_name, save_path)
        
        # 4. Equal weight comparison
        print("⚖️ Equal Weight Comparison")
        self.plot_equal_weight_comparison(save_path)
        
        print("✅ Dashboard creation complete!")
        if save_path:
            print(f"📁 Charts saved to: {save_path}")