# Computes Cosine Similarity between bit strings
__author__ = "Chirayu Desai"

import math
from collections import Counter
from itertools import combinations


def get_cosine_similarity(bit_string1, bit_string2):
    vector1 = Counter([int(d) for d in bit_string1])
    vector2 = Counter([int(d) for d in bit_string2])
    intersection = set(vector1.keys()) & set(vector2.keys())
    numerator = sum([vector1[x] * vector2[x] for x in intersection])

    sum1 = sum([vector1[x] ** 2 for x in vector1.keys()])
    sum2 = sum([vector2[x] ** 2 for x in vector2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    sly = 0.0
    if denominator:
        sly = float(numerator) / denominator
    # dist = (2 * math.acos(sly)) / math.pi
    return 1 - sly


# cosine = get_cosine_similarity('111111', '000000')
#
# print('Cosine:', cosine)
clusters = dict()
bit_strings = dict()
cluster_dist = dict()
with open("clusters.txt", 'r') as file:
    i = 0
    lines = file.readlines()
    lines = [eval(line.strip()) for line in lines]
    for line in lines:
        clusters[i] = line
        i = i + 1

with open("padded_bit_strings.txt", 'r') as file:
    lines = file.readlines()
    lines = [eval(line.strip()) for line in lines]
    for line in lines:
        bit_strings[line[0]] = line[1]

index = 1
for cluster in clusters:
    print(index)
    index += 1
    words = list()
    [words.append(tup[0]) for tup in clusters[cluster]]
    pairs = (list(combinations(words, 2)))
    if len(pairs) == 0:
        cluster_dist[cluster] = 0.0
    else:
        cluster_dist[cluster] = 0.0
        for pair in pairs:
            sim = get_cosine_similarity(bit_strings[pair[0]], bit_strings[pair[1]])
            cluster_dist[cluster] = cluster_dist[cluster] + sim
        cluster_dist[cluster] = cluster_dist[cluster] / len(pairs)


with open("cluster_wise_avg_cos_dist.txt", 'w') as file:
    for cluster in clusters:
        file.write("Cluster " + " : [" + " ".join(str(v) for v in clusters[cluster]) + "]")
        file.write('\n')
        file.write("Average Cosine Distance between Pairs : " + str(cluster_dist[cluster]))
        file.write('\n')
        file.write('\n')
