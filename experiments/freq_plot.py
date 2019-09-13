import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

data = open('dump.txt').read()
data = sorted([float(x) for x in data.split('\n') if not x == ''])
sum_scores = np.cumsum(data) / sum(data)
print(data)
print(sum_scores)

density = gaussian_kde(data)
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.title('Distribution of BERT ClaimBuster Scores\nOver a Complete Trump vs. Clinton Presidential Debate\n(Computed Using Gaussian Kernel Density Estimation)')
plt.xlabel('ClaimBuster Score')
plt.ylabel('Density')
plt.show()