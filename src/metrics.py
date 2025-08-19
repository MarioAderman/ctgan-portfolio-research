import numpy as np
import pandas as pd


def compute_annualized_return(serie):
    '''
    Calculates the annualized return of a serie.
    '''
    total_return = (serie.iloc[-1]/serie.iloc[0]) - 1  # Total return over period
    # Calculate actual years elapsed (not just number of observations)
    start_date = serie.index[0]
    end_date = serie.index[-1]
    years_elapsed = (end_date - start_date).days / 365.25  # Account for leap years
    annualized_return = ((1 + total_return)**(1/years_elapsed) - 1) * 100
    return annualized_return

def compute_cvar(serie, alpha=0.95, tf=365):
    '''
    Calculates the ex post CVAR of a serie.
    '''
    cvar = serie.pct_change(tf).dropna().sort_values()
    var = np.percentile(cvar, 100 - 100*alpha)
    cvar = cvar[cvar<=var]
    cvar = cvar.mean() 
    cvar = -100 * cvar
    return cvar
        
def compute_mean_hhi(portfolios):
    '''
    Calculates the diversification measure (HHI) for a set of historical portfolios.
    '''
    mean_hhi = pd.Series(dtype=np.float)
    for t, portfolio in portfolios.iterrows():
            m = len(portfolio)
            w = portfolio[portfolio>0.0]
            w /= 100
            hhi = w**2
            hhi = hhi.sum()
            hhi = (1-hhi) / (1-(1/m))
            mean_hhi[t] = hhi
    mean_hhi = mean_hhi.mean()
    return mean_hhi

def compute_mean_rotation(portfolios):
    '''
    Calculates the mean absolute rotation of the historical portfolios
    '''
    rotation = portfolios.diff()
    rotation = rotation.dropna()
    rotation = rotation.abs()
    rotation = rotation.sum(axis=1)
    rotation /= 2
    mean_rotation = rotation.mean()
    return mean_rotation