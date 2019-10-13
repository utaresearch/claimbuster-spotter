import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_stuff(filename, title, axes):
	data = open(filename).read()
	data = sorted([float(x) for x in data.split('\n') if not x == ''])
	density = gaussian_kde(data)
	xs = np.linspace(0,1,200)
	density.covariance_factor = lambda : .2
	density._compute_covariance()

	axes.plot(xs,density(xs))
	axes.set_title('Distribution of {} ClaimBuster Scores Over a\nComplete Trump vs. Clinton Presidential Debate'.format(title))
	axes.set_xlabel('ClaimBuster Score')
	axes.set_ylabel('Density')


if __name__ == '__main__':
	fig = plt.figure(figsize=(12, 5))
	ax = fig.add_subplot(121)
	ax1 = fig.add_subplot(122)
	plot_stuff('dump.txt', 'BERT', ax)
	plot_stuff('dump2.txt', 'SVM', ax1)
	plt.savefig('comparison.png', dpi=1000)
	plt.show()