# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#

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
	axes.set_title('Distribution of {} ClaimSpotter Scores Over a\nComplete Trump vs. Clinton Presidential Debate'.format(title))
	axes.set_xlabel('ClaimSpotter Score')
	axes.set_ylabel('Density')


if __name__ == '__main__':
	fig = plt.figure(figsize=(12, 5))
	ax = fig.add_subplot(121)
	ax1 = fig.add_subplot(122)
	plot_stuff('../experiments/dump.txt', 'BERT', ax)
	plot_stuff('../experiments/dump2.txt', 'SVM', ax1)
	plt.savefig('../experiments/comparison.png', dpi=1000)
	plt.show()