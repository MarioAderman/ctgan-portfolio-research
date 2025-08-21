# Standard library imports
import warnings

# Third party imports
import hdbscan
import pandas as pd
import torch
from sdv.tabular import CTGAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Local application imports
from src.generators.normalizer import Normalizer

warnings.filterwarnings("ignore")

def _select_optimal_cuda_setting(sample_size, epochs, features_count):
    """
    Intelligently select CUDA vs CPU based on hardware and dataset characteristics.
    
    Args:
        sample_size: Number of samples to generate
        epochs: Number of training epochs
        features_count: Number of features in dataset
        
    Returns:
        bool: True to use CUDA, False to use CPU
    """
    if not torch.cuda.is_available():
        return False
    
    # Get GPU information
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    # Calculate computational complexity score
    compute_score = sample_size * epochs * features_count
    
    # Hardware-specific thresholds based on empirical testing
    if 'gtx 1050' in gpu_name or 'gtx 1060' in gpu_name:
        # Lower-end GPUs: CPU often faster for small datasets
        threshold = 50000  # Conservative threshold
    elif 'rtx' in gpu_name or 'gtx 1080' in gpu_name or 'tesla' in gpu_name:
        # Higher-end GPUs: More aggressive CUDA usage
        threshold = 10000
    else:
        # Unknown GPU: Conservative approach
        threshold = 30000
    
    use_cuda = compute_score > threshold
    
    # Log the decision for transparency
    print(f"🔧 Hardware: {torch.cuda.get_device_name(0)}")
    print(f"📊 Compute score: {compute_score} (threshold: {threshold})")
    print(f"⚡ Selected: {'CUDA' if use_cuda else 'CPU'} for optimal performance")
    
    return use_cuda

class CTGANGenerator():

    def __init__(self, asset_returns, params=None, features=None):
        self.asset_returns = asset_returns
        self.features = features
        self.name = 'CTGAN'
        
        # Default parameters with intelligent CUDA selection
        default_params = {
            'embedding_dim': 32,
            'generator_dim': (64, 64),
            'discriminator_dim': (64, 64),
            'epochs': 5,
            'generator_lr': 1e-4,
            'discriminator_lr': 1e-4,
            'verbose': False
        }
        
        if params:
            self.params = params
        else:
            # Intelligent CUDA selection will be done when we know dataset size
            self.params = default_params


    def generate_sample(self, sample_size, start_date, end_date):
        # Intelligent CUDA selection if not explicitly set
        if 'cuda' not in self.params:
            features_count = len(self.asset_returns.columns)
            if self.features is not None:
                features_count += len(self.features.columns)
            
            self.params['cuda'] = _select_optimal_cuda_setting(
                sample_size=sample_size,
                epochs=self.params['epochs'],
                features_count=features_count
            )

        model = CTGAN(**self.params)
        returns_interval = self.asset_returns.loc[
            (self.asset_returns.index <= end_date) & (self.asset_returns.index >= start_date)]
        fit_cols = list(self.asset_returns.columns) + ['cluster']
        normalizer = None
        
        if self.features is not None:
            returns_interval = returns_interval.join(self.features, how='left').ffill()
            normalizer = Normalizer()
            returns_interval = normalizer.normalize(returns_interval)
            fit_cols = list(self.asset_returns.columns) + list(self.features.columns) + ['cluster']


        # Applies PCA
        pca, returns_interval = self._construct_pca(returns_interval)
        fit_cols = [f"C_{i}" for i in range(pca.n_components_)] + ['cluster']


        # Dimensionality reduction
        returns_interval, X_embedded = self._reduce_dim(returns_interval)


        # Clusters definition
        returns_interval = self._define_clusters(returns_interval, X_embedded)

        # Fits CTGAN using categorical variable of state       
        model.fit(returns_interval[fit_cols])

        sample = model.sample(sample_size)[fit_cols[:-1]]
        sample_val = sample.values

       # Reconstruct assets
        sample_val = pca.inverse_transform(sample_val)
        
        # De-normalizes
        if self.features is not None:
            sample_val = normalizer.denormalize(sample_val)


        return sample_val
    
    def _construct_pca(self, returns_interval):
        pca = PCA(n_components=returns_interval.shape[1])
        pca.fit(returns_interval)
        asset_returns_interval_trans = pca.transform(returns_interval)
        pca_cols = [f"C_{i}" for i in range(pca.n_components_)]
        return pca, pd.DataFrame(asset_returns_interval_trans,
                                        index=returns_interval.index,
                                        columns=pca_cols)

    def _reduce_dim(self, returns_interval, dims=2):
        X_embedded = TSNE(n_components=dims, learning_rate='auto', init='pca').fit_transform(returns_interval)
        returns_interval['x'] = X_embedded[:, 0]
        returns_interval['y'] = X_embedded[:, 1]
        return returns_interval, X_embedded

    def _define_clusters(self, returns_interval, X_embedded):
        cluster_dim = max(10, int(len(returns_interval) * 0.005))
        clusterer = hdbscan.HDBSCAN(min_samples=cluster_dim, min_cluster_size=cluster_dim)
        clusterer.fit(X_embedded)
        returns_interval['cluster'] = ['c_' + str(c) for c in clusterer.labels_]

        return returns_interval