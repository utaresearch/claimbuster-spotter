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

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from bert_adversarial.core.api.api_wrapper import ClaimSpotterAPI

api = ClaimSpotterAPI()


def get_score(text):
    api_result = api.single_sentence_query(text)

    return api_result[-1]


def compute_kde(x):
    # https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
    x = np.asarray(x).reshape(-1, 1)
    X_plot = np.linspace(0, 1, len(x))[:, np.newaxis]
    fig, ax = plt.subplots()
    colors = ['darkorange']
    kernels = ['gaussian']
    lw = 15

    for color, kernel in zip(colors, kernels):
        kde = KernelDensity(kernel=kernel, bandwidth=0.20).fit(x)
        log_dens = kde.score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw, linestyle='-')

    ax.legend().remove()
    ax.plot(x[:, 0], -0.005 - 0.01 * np.random.random(x.shape[0]), '+k', markersize=lw)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.25)
    plt.show()


cfs_scores = []

with open("../data/two_class/2020_debate.tsv", encoding="utf-8") as test_data_sv:
    test_data = csv.reader(test_data_sv, delimiter="\t", quotechar='"')
    next(test_data)

    for d in test_data:
        # Get label probabilities from trained model
        cfs_score = get_score(d[2])

        # Aggregate CFS scores given by model
        cfs_scores.append(cfs_score[1])

compute_kde(cfs_scores)
