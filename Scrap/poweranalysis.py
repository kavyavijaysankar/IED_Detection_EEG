import numpy as np
from statsmodels.stats.power import TTestIndPower

def calculate_rank_sum_sample_size(effect_size_d, alpha=0.05, power=0.80):
    """
    Calculates required sample size per group for a Wilcoxon Rank-Sum test
    using the Asymptotic Relative Efficiency (ARE) adjustment over a t-test.
    
    effect_size_d: Cohen's d (Mean_diff / Standard_Deviation)
    """
    analysis = TTestIndPower()
    # Calculate sample size for a standard independent t-test
    n_ttest = analysis.solve_power(
        effect_size=effect_size_d, 
        alpha=alpha, 
        power=power, 
        ratio=1.0, 
        alternative='two-sided'
    )
    
    # Apply non-parametric ARE correction factor (~15% boost)
    n_ranksum = np.ceil(n_ttest * 1.157)
    return int(n_ranksum)

# Example Calculations across different potential effect sizes
print("--- Sample Size Requirements per Group (Power = 80%, alpha = 0.05) ---")
for d, label in [(1.2, "Very Large"), (0.8, "Large"), (0.5, "Moderate"), (0.3, "Small-Moderate")]:
    n_per_group = calculate_rank_sum_sample_size(d)
    print(f"Effect Size d = {d:.1f} ({label}): {n_per_group} EEGs per group (Total N = {n_per_group * 2})")