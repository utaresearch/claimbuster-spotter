import math

# ver == 0 or 1
def compute_dcg_term(i, labels, ver=1):
    return labels[i - 1] / math.log2(i + 1) if ver == 0\
        else ((1 << labels[i - 1]) - 1) / math.log2(i + 1)


# Precondition: for each index i, scores[i] corresponds with labels[i]
def compute_ndcg(scores, labels):
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