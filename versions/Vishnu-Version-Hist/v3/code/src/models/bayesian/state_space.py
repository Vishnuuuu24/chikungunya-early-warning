"""
Bayesian State-Space Model for Chikungunya Early Warning (Stabilized)

Phase 4.2: Stabilization with increased MCMC budget and adapt_delta.

Changes from v2_proto:
- Default warmup: 500 -> 1000
- Default samples: 500 -> 1000
- Default chains: 4
- New parameter: adapt_delta (default 0.95)

Hierarchical Negative Binomial state-space model with:
- Latent log-transmission risk Z_{d,t}
- AR(1) dynamics with climate forcing
- Partial pooling across districts

Reference: Phase 4.2 stabilization specification
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import warnings

try:
    from cmdstanpy import CmdStanModel
    CMDSTAN_AVAILABLE = True
except ImportError:
    CMDSTAN_AVAILABLE = False
    warnings.warn("CmdStanPy not available. Install with: pip install cmdstanpy")

from ..base import BaseModel


class BayesianStateSpace(BaseModel):
    """
    Hierarchical Bayesian state-space model for outbreak prediction.
    
    Uses Stan for MCMC inference via CmdStanPy.
    
    Phase 4.2 stabilization:
    - Increased default MCMC budget
    - Added adapt_delta control parameter
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="bayesian_state_space", config=config)
        
        if not CMDSTAN_AVAILABLE:
            raise RuntimeError("CmdStanPy required but not available")
        
        # Model configuration - STABILIZATION: increased defaults
        self.n_warmup = config.get('n_warmup', 1000) if config else 1000
        self.n_samples = config.get('n_samples', 1000) if config else 1000
        self.n_chains = config.get('n_chains', 4) if config else 4
        self.adapt_delta = config.get('adapt_delta', 0.95) if config else 0.95  # NEW
        self.seed = config.get('seed', 42) if config else 42

        # Config-driven outbreak percentile for converting posterior predictive
        # counts into outbreak probabilities.
        self.outbreak_percentile = config.get('outbreak_percentile') if config else None
        
        # Stan model path
        self.stan_file = config.get('stan_file', None) if config else None
        
        # Fitted objects
        self.model_ = None
        self.fit_ = None
        self.data_ = None
        self.district_map_ = None
        
    def _get_stan_file(self) -> Path:
        """Get path to Stan model file."""
        if self.stan_file:
            return Path(self.stan_file)
        
        # Default location relative to this file
        # Path: versions/Vishnu-Version-Hist/v3/code/src/models/bayesian/state_space.py
        # Stan: versions/Vishnu-Version-Hist/v3/stan_models/hierarchical_ews_v01.stan
        module_dir = Path(__file__).parent
        # Go up: bayesian -> models -> src -> code -> v3
        v3_root = module_dir.parent.parent.parent.parent
        stan_path = v3_root / "stan_models" / "hierarchical_ews_v01.stan"
        
        if stan_path.exists():
            return stan_path
        
        raise FileNotFoundError(f"Stan model not found at {stan_path}")
    
    def _prepare_stan_data(
        self, 
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'cases'
    ) -> Dict[str, Any]:
        """
        Prepare data dictionary for Stan model.
        
        Args:
            df: DataFrame with features, target, and metadata
            feature_cols: Feature column names (used for temp anomaly)
            target_col: Column with case counts
            
        Returns:
            Dictionary formatted for Stan
        """
        # Sort by district and time.
        # IMPORTANT: If the caller added a unique row id to break ties for
        # duplicate (state, district, year, week), include it in the sort.
        sort_cols = ['state', 'district', 'year', 'week']
        if '_unique_row_id' in df.columns:
            sort_cols.append('_unique_row_id')
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        # Create district IDs
        df['district_id'] = pd.factorize(df['state'] + '_' + df['district'])[0] + 1
        self.district_map_ = df[['state', 'district', 'district_id']].drop_duplicates()
        
        # Create time index (week within the dataset)
        df['time_idx'] = pd.factorize(
            df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)
        )[0] + 1
        
        N = len(df)
        D = df['district_id'].max()
        T_max = df['time_idx'].max()
        
        # Get cases (convert to integer counts)
        # If we have incidence per 100k, convert back to approximate counts
        if target_col == 'cases':
            y = df['cases'].fillna(0).astype(int).values
        else:
            y = df[target_col].fillna(0).astype(int).values
        
        # Get temperature anomaly (use feat_temp_anomaly if available)
        if 'feat_temp_anomaly' in df.columns:
            temp_anomaly = df['feat_temp_anomaly'].fillna(0).values
        elif 'temp_celsius' in df.columns:
            # Compute simple anomaly: deviation from mean
            temp_anomaly = (df['temp_celsius'] - df['temp_celsius'].mean()).fillna(0).values
        else:
            temp_anomaly = np.zeros(N)
        
        stan_data = {
            'N': N,
            'D': D,
            'T_max': T_max,
            'district': df['district_id'].values,
            'time': df['time_idx'].values,
            'y': y,
            'temp_anomaly': temp_anomaly
        }
        
        self.data_ = stan_data
        return stan_data
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[list] = None
    ) -> 'BayesianStateSpace':
        """
        Fit the Bayesian state-space model via MCMC.
        
        Args:
            X: Feature matrix (for API compatibility)
            y: Target vector (for API compatibility)
            df: Full DataFrame with metadata (preferred)
            feature_cols: Feature column names
            
        Returns:
            self
        """
        if df is None:
            raise ValueError("DataFrame with metadata required for Bayesian model")
        
        # Compile Stan model
        stan_file = self._get_stan_file()
        print(f"Compiling Stan model from {stan_file}...")
        self.model_ = CmdStanModel(stan_file=str(stan_file))
        
        # Prepare data
        print("Preparing data for Stan...")
        stan_data = self._prepare_stan_data(df, feature_cols or [])
        
        print(f"Data summary: N={stan_data['N']}, D={stan_data['D']}, T_max={stan_data['T_max']}")
        
        # Run MCMC - STABILIZATION: added adapt_delta
        print(f"Running MCMC: {self.n_chains} chains, {self.n_warmup} warmup, {self.n_samples} samples")
        print(f"  adapt_delta={self.adapt_delta}")
        self.fit_ = self.model_.sample(
            data=stan_data,
            chains=self.n_chains,
            iter_warmup=self.n_warmup,
            iter_sampling=self.n_samples,
            seed=self.seed,
            adapt_delta=self.adapt_delta,  # STABILIZATION: new parameter
            show_progress=True
        )
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Get posterior predictive outbreak probabilities.
        
        For now, returns probability that predicted cases exceed threshold.
        
        Args:
            X: Feature matrix (for API compatibility)
            df: DataFrame with metadata
            
        Returns:
            Array of outbreak probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get posterior predictive samples
        y_rep = self.fit_.stan_variable('y_rep')  # Shape: (n_samples * n_chains, N)
        
        # Compute probability of exceeding config-driven percentile of training data.
        # Use all training cases (including zeros) and enforce minimum threshold 1.0
        # to match the outbreak-threshold logic used in lead-time evaluation.
        y_train = self.data_['y']
        percentile = self.outbreak_percentile if self.outbreak_percentile is not None else 75
        threshold = float(np.percentile(y_train, percentile)) if len(y_train) else 1.0
        threshold = max(threshold, 1.0)
        
        # P(outbreak) = fraction of posterior samples exceeding threshold
        prob_outbreak = (y_rep > threshold).mean(axis=0)
        
        return prob_outbreak

    def get_latent_risk_samples_per_observation(self) -> np.ndarray:
        """Return posterior samples of latent risk aligned to each observation row.

        The Stan model defines a latent state Z[d, t]. Each observation i maps to
        (district[i], time[i]) provided in the Stan data.

        Returns:
            Array of shape (n_draws, N) with Z for each observation.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if self.data_ is None:
            raise ValueError("Stan data not prepared.")

        Z = self.get_latent_states()  # (n_draws, D, T_max)
        district_idx = np.asarray(self.data_['district'], dtype=int) - 1
        time_idx = np.asarray(self.data_['time'], dtype=int) - 1
        if district_idx.ndim != 1 or time_idx.ndim != 1:
            raise ValueError("Invalid district/time index arrays in Stan data")

        return Z[:, district_idx, time_idx]

    def get_latent_risk_summary_per_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, sd) of latent risk Z for each observation row."""
        z_samples = self.get_latent_risk_samples_per_observation()
        return z_samples.mean(axis=0), z_samples.std(axis=0)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get MCMC diagnostics.
        
        Returns:
            Dictionary with R-hat, ESS, divergences, etc.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        # Get summary statistics
        summary = self.fit_.summary()
        
        # Key parameters to check
        key_params = ['mu_alpha', 'sigma_alpha', 'rho', 'beta_temp', 'sigma', 'phi']
        
        # Get divergences from diagnostic output
        try:
            diag_output = self.fit_.diagnose()
            n_divergences = diag_output.count('divergent') if diag_output else 0
        except:
            n_divergences = "N/A"
        
        diagnostics = {
            'n_divergences': n_divergences,
            'max_rhat': summary['R_hat'].max(),
            'min_ess_bulk': summary['ESS_bulk'].min(),
            'min_ess_tail': summary['ESS_tail'].min(),
            'parameter_summary': {}
        }
        
        # Get summary for key parameters
        for param in key_params:
            if param in summary.index:
                row = summary.loc[param]
                diagnostics['parameter_summary'][param] = {
                    'mean': row['Mean'],
                    'std': row['StdDev'],
                    'rhat': row['R_hat'],
                    'ess_bulk': row['ESS_bulk']
                }
        
        return diagnostics
    
    def get_latent_states(self) -> np.ndarray:
        """
        Get posterior samples of latent states Z.
        
        Returns:
            Array of shape (n_samples, D, T_max)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return self.fit_.stan_variable('Z')
    
    def get_posterior_predictive(self) -> np.ndarray:
        """
        Get posterior predictive samples.
        
        Returns:
            Array of shape (n_samples, N)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return self.fit_.stan_variable('y_rep')
    
    def print_diagnostics(self) -> None:
        """Print formatted diagnostics summary."""
        diag = self.get_diagnostics()
        
        print("\n" + "=" * 50)
        print("MCMC DIAGNOSTICS")
        print("=" * 50)
        
        print(f"\nDivergences: {diag['n_divergences']}")
        print(f"Max R-hat: {diag['max_rhat']:.4f}")
        print(f"Min ESS (bulk): {diag['min_ess_bulk']:.0f}")
        print(f"Min ESS (tail): {diag['min_ess_tail']:.0f}")
        
        print("\nParameter Estimates:")
        print("-" * 50)
        print(f"{'Parameter':<15} {'Mean':>10} {'Std':>10} {'R-hat':>8} {'ESS':>8}")
        print("-" * 50)
        
        for param, vals in diag['parameter_summary'].items():
            print(f"{param:<15} {vals['mean']:>10.3f} {vals['std']:>10.3f} "
                  f"{vals['rhat']:>8.3f} {vals['ess_bulk']:>8.0f}")
        
        # Diagnostic flags
        print("\n" + "-" * 50)
        if diag['n_divergences'] != 0 and diag['n_divergences'] != "N/A":
            print("⚠️  WARNING: Divergences detected!")
        if diag['max_rhat'] > 1.05:
            print("⚠️  WARNING: R-hat > 1.05 (chains may not have converged)")
        if diag['min_ess_bulk'] < 100:
            print("⚠️  WARNING: Low ESS (< 100)")
        
        no_divergences = diag['n_divergences'] == 0 or diag['n_divergences'] == "N/A"
        if no_divergences and diag['max_rhat'] <= 1.05 and diag['min_ess_bulk'] >= 100:
            print("✓ All diagnostics passed")
