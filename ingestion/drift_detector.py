import numpy as np
import pandas as pd

class SimpleDriftDetector:
    """
    Task 3.3 implementation for Drift Detection.
    Uses an absolute Z-score of the rolling batch means against an established baseline distribution.
    Optimized to compute baseline parameters once and safely skip missing/new features.
    """
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.baseline_stats = {}  # In-memory storage cache for baseline statistical parameters

    def set_baseline(self, baseline_df):
        """
        Calculates and caches the baseline mean and standard deviation per feature column.
        Ensures we are not wastefully re-computing historical statistics inside our ingestion loop.
        """
        if baseline_df is None or baseline_df.empty:
            return
        
        for col in baseline_df.columns:
            if np.issubdtype(baseline_df[col].dtype, np.number):
                self.baseline_stats[col] = {
                    "mean": baseline_df[col].mean(),
                    "std": baseline_df[col].std()
                }

    def get_drifted_columns(self, current_df):
        """
        Task 3.3 Functional Requirements:
        - Evaluates columns that overlap in BOTH baseline and runtime data frames.
        - Skips vanished features automatically.
        - Ignores new features until a proper baseline is computed.
        - Evaluates all features fully without exiting early on the first error flag.
        """
        if current_df.empty or not self.baseline_stats:
            return []
        
        drifted_features = []
        # Explicit column intersection check (Only processes overlapping keys)
        target_cols = set(self.baseline_stats.keys()).intersection(current_df.columns)
        
        for col in target_cols:
            if not np.issubdtype(current_df[col].dtype, np.number):
                continue
                
            base_mean = self.baseline_stats[col]["mean"]
            base_std = self.baseline_stats[col]["std"]
            curr_mean = current_df[col].mean()
            
            # Guard clause against zero-variance features to prevent division by zero math errors
            if base_std == 0 or pd.isna(base_std):
                continue
                
            # Perform basic parametric Z-score drift testing calculation
            z_score = abs(curr_mean - base_mean) / base_std
            if z_score > self.threshold:
                drifted_features.append(col)
        
        if drifted_features:
            print(f"Drift detected in columns: {drifted_features}")
            
        return drifted_features