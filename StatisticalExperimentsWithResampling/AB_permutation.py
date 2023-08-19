import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Tkagg')
%matplotlib
import matplotlib.pyplot as plt

# ------------------------------------------
# A/B test

# set the seed for deterministic behaviour
random.seed(1)

def permute_values(pooled_data: pd.Series, n1: int, n2: int) -> float:
    n = n1 + n2
    idx1 = random.sample(range(n), n1)
    idx2 = list(set(range(n)) - set(idx1))
    return pooled_data.loc[idx1].mean() - pooled_data.loc[idx2].mean()

# new drug
n1_success = 2034
n1_fail = 453
n1_total = n1_success + n1_fail
n1_mean = n1_success/n1_total
print(f'new drug: treatment successful: {n1_success}, treatment failed: {n1_fail}, number of animals treated: {n1_total}, success rate: {n1_mean: .2%}')

# traditional treatment
n2_success = 1434
n2_fail = 351
n2_total = n2_success + n2_fail
n2_mean = n2_success/n2_total
print(f'traditional treatment: treatment successful: {n2_success}, treatment failed: {n2_fail}, number of animals treated: {n2_total}, success rate: {n2_mean: .2%}')

# successful treatment denoted by 1 and failed treatment denoted by 0
pooled_data = pd.Series([1]*n1_success + [0]*n1_fail + [1]*n2_success + [0]*n2_fail)

# permutation test
n_permutations = 10_000
differences = []
for _ in tqdm(range(n_permutations)):
    differences.append(permute_values(pooled_data, n1=n1_total, n2=n2_total))
differences = pd.Series(differences)

# plot simulated differences
fig = plt.figure()
ax = fig.subplots()
counts, bins, patches = ax.hist(differences, bins=40, rwidth=0.9, fill=True, ec='k')
for i in range(len(patches)):
    if bins[i] > n1_mean-n2_mean:
        patches[i].set_facecolor('r')
    else:
        patches[i].set_facecolor('w')
ax.axvline(x=n1_mean-n2_mean, color='k', ls = '--', lw=1)
ax.text(1.1*(n1_mean-n2_mean), 0.9*max(counts), 'observed')
plt.setp(ax, xlabel='simulated difference')
plt.setp(ax, ylabel='frequency')
fig.savefig('hist.png', dpi=600)

# obtain the p-value
p_value = (differences>=(n1_mean-n2_mean)).mean()
print(f'p-value is {p_value:.4f}')
