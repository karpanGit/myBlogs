import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Tkagg')
%matplotlib
import matplotlib.pyplot as plt

random.seed(1)

def permute_values(pooled_data: pd.Series, n1: int, n2: int) -> float:
    n = n1 + n2
    idx1 = random.sample(range(n), n1)
    idx2 = list(set(range(n)) - set(idx1))
    return pooled_data.loc[idx1].mean() - pooled_data.loc[idx2].mean()


def compute_stat_significance(n1_success, n1_total, n2_success, n2_total, a=0.05, n_permutations=1000):
    '''
    Computes whether the count difference is statistically significant using a permutation test
    :param n1_success: number of times the treatment is successful using the new drug
    :param n1_total: number of animals treated with the new drug
    :param n2_success: number of times the treatment is successful using the traditional treatment
    :param n2_total: number of animals treated with the traditional treatment
    :param a: significance level (default 0.05)
    :param n_permutations: number of permutations (default 1000)
    :return: True if the difference is statistically significant and False otherwise
    '''
    n1_fail = n1_total - n1_success
    n1_mean = n1_success/n1_total
    n2_fail = n2_total - n2_success
    n2_mean = n2_success / n2_total

    pooled_data = pd.Series([1] * n1_success + [0] * n1_fail + [1] * n2_success + [0] * n2_fail)
    differences = []
    for _ in range(n_permutations):
        differences.append(permute_values(pooled_data, n1=n1_total, n2=n2_total))
    differences = pd.Series(differences)
    p_value = (differences>(n1_mean-n2_mean)).mean()
    print(f'contingency matrix: {n1_success}, {n1_total}, {n2_success}, {n2_total}, p-value is {p_value:.4f}')
    return p_value < a


power_results = []
for effect_size in [0.015, 0.02]:

    # hypothetical data for the traditional treatment
    n_hypothetical_data = 10_000
    traditional_treatment_effectiveness = 0.80336
    hypothetical_data2 = [1]*round(traditional_treatment_effectiveness*n_hypothetical_data)
    hypothetical_data2.extend([0]*(n_hypothetical_data-len(hypothetical_data2)))

    # hypothetical data for the new drug treatment
    new_drug_treatment_effectiveness = 0.80336 + effect_size
    hypothetical_data1 = [1]*round(new_drug_treatment_effectiveness*n_hypothetical_data)
    hypothetical_data1.extend([0]*(n_hypothetical_data-len(hypothetical_data1)))

    n_bootstrap_trials = 200

    for n_sample in tqdm(range(3_000, 8_000, 1_000)):
        # use bootstrapping to create samples
        difference_significances = []
        for _ in tqdm(range(n_bootstrap_trials)):
            sample1 = np.random.choice(hypothetical_data1, replace=True, size=n_sample)
            sample2 = np.random.choice(hypothetical_data2, replace=True, size=n_sample)
            difference_significances.append(compute_stat_significance(n1_success=sum(sample1), n1_total=n_sample,
                                                                      n2_success=sum(sample2), n2_total=n_sample,
                                                                      a=0.05, n_permutations=500)
                                    )
        power_results.append({'effect size': effect_size, 'sample size': n_sample, 'power': sum(difference_significances)/n_bootstrap_trials})
        print(power_results)

power_results = pd.DataFrame(power_results)

fig = plt.figure()
ax = fig.subplots()
ax.scatter(power_results.loc[power_results['effect size']==0.015, 'sample size'],
           power_results.loc[power_results['effect size']==0.015, 'power'], label='effect size = 0.015')
ax.scatter(power_results.loc[power_results['effect size']==0.02, 'sample size'],
           power_results.loc[power_results['effect size']==0.02, 'power'], label='effect size = 0.02')
ax.legend()
ax.axhline(y=0.8, color='k', ls = '--', lw=1)
ax.text(3000, 0.78, 'target power')
plt.setp(ax, xlabel='sample size')
plt.setp(ax, ylabel='power')
fig.savefig('power.png', dpi=600)


import statsmodels.api as sm
for effective_increase in [0.015, 0.02]:
    effect_size = sm.stats.proportion_effectsize(0.80336 + effective_increase, 0.80336)
    analysis = sm.stats.TTestIndPower()
    result = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='larger')
    print(f'effect size: {effective_increase:.3f}, sample size: {result:.2f}')
