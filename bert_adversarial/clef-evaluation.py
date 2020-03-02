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

import csv
import math
from itertools import chain
from sklearn import metrics as mt
from copy import deepcopy
from numpy import argmax
from bert_adversarial.core.api.api_wrapper import ClaimSpotterAPI

api = ClaimSpotterAPI()


def get_score(text):
    api_result = api.single_sentence_query(text)

    return api_result[-1]


def compute_precisions(labels, scores, cutoff=None):
    # https://github.com/apepa/clef2019-factchecking-task1/blob/master/scorer/task1.py#L81
    combined = sorted([(scores[i], labels[i]) for i in range(len(scores))], reverse=True)
    cutoff = min(cutoff or math.inf, len(combined))
    combined = combined[:cutoff]
    precisions = [0.0] * cutoff

    for i, x in enumerate(combined):
        if x[1] == 1:
            precisions[i] += 1.0

    for i in range(1, cutoff):
        precisions[i] += precisions[i - 1]

    for i in range(1, cutoff):
        precisions[i] /= i+1

    return precisions


def compute_average_precision(labels, scores, cutoff=None):
    # https://github.com/apepa/clef2019-factchecking-task1/blob/master/scorer/task1.py#L52
    combined = sorted([(scores[i], labels[i]) for i in range(len(scores))], reverse=True)
    cutoff = min(cutoff or math.inf, len(combined))
    combined = combined[:cutoff]
    labels = [x[1] for x in combined]
    precisions = []
    num_correct = 0
    num_positive = sum(labels)

    for i, x in enumerate(combined):
        if x[1] == 1:
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    
    if precisions:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0

    return avg_prec


def compute_dcg_term(i, labels, ver=1):
    # Difference between version 0 and 1:
    # https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain
    return labels[i - 1] / math.log2(i + 1) if ver == 0 else ((1 << labels[i - 1]) - 1) / math.log2(i + 1)


def compute_ndcg(labels, scores, cutoff=None, ver=0):
    # Precondition: for each index i, scores[i] corresponds with labels[i]
    combined = sorted([(scores[i], labels[i]) for i in range(len(scores))], reverse=True)
    cutoff = min(cutoff or math.inf, len(combined))
    combined = combined[:cutoff]
    labels = [x[1] for x in combined]

    dcg = sum([compute_dcg_term(i, labels, ver=ver) for i in range(1, len(labels) + 1, 1)])
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum([compute_dcg_term(i, ideal_labels, ver=ver) for i in range(1, len(labels) + 1, 1)])

    try:
        return dcg / idcg
    except ZeroDivisionError:
        return dcg / 0.0000000000001


# We've combined the CLEF test files into one, the code below handles
# assessing the result of the sentences from each file independently.
# The evaluation is executed as it was in the CLEF 2019 competition.

all_ground_truth_labels = []
all_predicted_labels = []
cfs_scores = []

multi_test_doc_ground_truth_labels = []
multi_test_doc_predicted_labels = []
multi_test_doc_cfs_scores = []

p_at_k_thresholds = [10, 20, 50]
prev_sent_id = -1

with open("../data/clef/clef2019_test.tsv", encoding="utf-8") as test_data_sv:
    test_data = csv.reader(test_data_sv, delimiter="\t", quotechar='"')
    next(test_data)

    for d in test_data:
        cur_sent_id = int(d[0])

        if cur_sent_id < prev_sent_id:
            multi_test_doc_ground_truth_labels.append(deepcopy(all_ground_truth_labels))
            multi_test_doc_predicted_labels.append(deepcopy(all_predicted_labels))
            multi_test_doc_cfs_scores.append(deepcopy(cfs_scores))

            all_ground_truth_labels = []
            all_predicted_labels = []
            cfs_scores = []
            prev_sent_id = cur_sent_id
        else:
            prev_sent_id = cur_sent_id
        
        # Get label probabilities from trained model
        cfs_score = get_score(d[2])
        sentence_label = int(d[1])

        # Aggregate ground truth and predicted labels for each sentence
        all_ground_truth_labels.append(sentence_label)
        all_predicted_labels.append(argmax(cfs_score))

        # Aggregate CFS scores given by model
        cfs_scores.append(cfs_score[1])
    
    if len(multi_test_doc_ground_truth_labels) and len(all_ground_truth_labels):
        multi_test_doc_ground_truth_labels.append(deepcopy(all_ground_truth_labels))
        multi_test_doc_predicted_labels.append(deepcopy(all_predicted_labels))
        multi_test_doc_cfs_scores.append(deepcopy(cfs_scores))

num_docs = len(multi_test_doc_ground_truth_labels)
merged_gt_labels = list(chain.from_iterable(multi_test_doc_ground_truth_labels))
merged_pred_labels = list(chain.from_iterable(multi_test_doc_predicted_labels))
classification_report = mt.classification_report(merged_gt_labels, merged_pred_labels, digits=4)
confusion_matrix = mt.confusion_matrix(merged_gt_labels, merged_pred_labels)

balanced_accuracy = sum([mt.balanced_accuracy_score(x, y) for x, y in zip(multi_test_doc_ground_truth_labels, multi_test_doc_predicted_labels)]) / num_docs
average_precision = sum([compute_average_precision(x, y) for x, y in zip(multi_test_doc_ground_truth_labels, multi_test_doc_cfs_scores)]) / num_docs
ndcg = sum([compute_ndcg(x, y) for x, y in zip(multi_test_doc_ground_truth_labels, multi_test_doc_cfs_scores)]) / num_docs

final_p_at_k = [0.0] * len(p_at_k_thresholds)
p_at_k_aggregator = []
precisions_at_k = [compute_precisions(x, y) for x, y in zip(multi_test_doc_ground_truth_labels, multi_test_doc_cfs_scores)]

try:
    for p in precisions_at_k:
        p_at_k_aggregator.append([p[th - 1] for th in p_at_k_thresholds])

    for idx in range(0, len(p_at_k_thresholds)):
        for p_l in p_at_k_aggregator:
            final_p_at_k[idx] += p_l[idx]

    final_p_at_k = [p / num_docs for p in final_p_at_k]
except IndexError:
    final_p_at_k = "Not enough test samples to calculate precisions at defined thresholds."


print("###### Test Set Classification Report #####\n", classification_report)
print("######## Test Set Confusion Matrix ########\n", confusion_matrix, "\n")
print("nDCG Score: ", ndcg)
print("(Mean) Average Precision: ", average_precision)
print("Balanced Accuracy: ", balanced_accuracy, "\n")
print("Average Precision @ k: ", p_at_k_thresholds, final_p_at_k)
