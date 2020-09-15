from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# sample poisson distribution
# lam: no of occurrences
sns.distplot(random.poisson(lam=2, size=1000), kde=False)

# comparing normal and poisson distributions
sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')

# comparing poisson and binomial distributions
sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')

plt.show()