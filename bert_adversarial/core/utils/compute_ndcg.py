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

import math


# ver == 0 or 1
def compute_dcg_term(i, labels, ver=1):
    return labels[i - 1] / math.log2(i + 1) if ver == 0\
        else ((1 << labels[i - 1]) - 1) / math.log2(i + 1)


# Precondition: for each index i, scores[i] corresponds with labels[i]
def compute_ndcg(labels, scores):
    combined = sorted([(scores[i], labels[i]) for i in range(len(scores))], reverse=True)
    labels = [x[1] for x in combined]

    selver = 0

    dcg = sum([compute_dcg_term(i, labels, ver=selver) for i in range(1, len(labels) + 1, 1)])
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum([compute_dcg_term(i, ideal_labels, ver=selver) for i in range(1, len(labels) + 1, 1)])

    return dcg / idcg


if __name__ == "__main__":
    print(compute_ndcg([5, 4, 3, 2, 1, 0], [3, 2, 3, 0, 1, 2]))
    print(compute_ndcg([5, 3, 4, 0, 1, 2], [3, 3, 2, 2, 1, 0]))