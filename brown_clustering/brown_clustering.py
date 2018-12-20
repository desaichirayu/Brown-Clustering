# A Polynomial time implementation of Brown Clustering algorithm
__author__ = "Chirayu Desai"

import os
import re
import math
import json
import string
import operator
from itertools import combinations
from collections import defaultdict, Counter

def get_word_and_tags(word_tag_tokens):
    """
    :param word_tag_tokens: a list of string tokens
    :return: separates each token into word and tag
    """
    word_tokens = list()
    for word_tag in word_tag_tokens:
        word_and_tag = word_tag.rsplit('/', 1)
        x = (word_and_tag[0], word_and_tag[1])
        word_tokens.append(x[0])
    return word_tokens


def get_tokens(line):
    """
    :param line: data line
    :return: string tokens from given lines
    """
    word_tag_tokens = list()
    tokens = line.split()
    word_tag_tokens.extend(tokens)
    return word_tag_tokens


def remove_blank_lines_and_get_tokens(lines):
    """
    :param lines:
    :return:
    """
    all_lines = list()
    for line in lines:
        if line != '\n' and line.split():
            all_lines.append(line.strip().split())
    return all_lines


def get_words_from_tokens(line):
    """
    :param line:
    :return:
    """
    words = list()
    words.append("START")
    for token in line:
        wo = token.rsplit('/', 1)[0]
        wo = wo.lower()
        wo = ''.join(ch for ch in wo if ch is not '.')
        if wo != '':
            words.append(wo)
    words.append("END")
    return words


def process_directory(directory):
    """
    Processes Raw data
    :param directory:
    :return:
    """
    file_names = os.listdir(directory)
    all_data = list()
    for file_name in file_names:
        file = open(directory + file_name, 'r')
        lines = file.readlines()
        lines = remove_blank_lines_and_get_tokens(lines)
        for line in lines:
            new_line = get_words_from_tokens(line)
            all_data.append(new_line)
        file.close()
    return all_data


def process_directory1(directory):
    """
    Processes Raw data
    :param directory:
    :return:
    """
    files = os.listdir(directory)
    file_data = list()
    for file_name in files:
        file = open(directory + file_name, 'r')
        lines = file.readlines()
        all_lines = list()
        lines = filter(lambda x: not x.isspace(), lines)
        for line in lines:
            line = re.sub('\s+', ' ', line).strip()
            line1 = " ".join(get_word_and_tags(get_tokens(line)))
            line1 = ''.join(ch for ch in line1 if ch not in set(string.punctuation))
            line1 = re.sub('\s+', ' ', line1).strip()
            line1 = line1.lower()
            all_lines.append(line1)
        for line2 in all_lines:
            if line2 != "" and line2 != " " and line2 != "\n":
                new_line = ['START'] + line2.split() + ['END']
                file_data.append(new_line)
        file.close()
    return file_data


def get_vocabulary(all_data):
    """
    :param all_data:
    :return: a counter with frequency of vocabulary items
    """
    all_vocabulary = list()
    tokens_to_exclude = ['START', 'END']
    [[all_vocabulary.append(token) for token in sentence if token not in tokens_to_exclude] for sentence in all_data]
    return Counter(all_vocabulary)


def put_unk_in_data(all_data, unk_words):
    """
    :param all_data:
    :param unk_words: words with low frequency
    :return:
    """
    return [['UNK' if wo in unk_words else wo for wo in sentence] for sentence in all_data]


def put_unk(old_vocabulary, all_data):
    """
    Replaces words with low frequency in vocabulary with 'UNK'
    :param old_vocabulary:
    :param all_data:
    :return:
    """
    unk_words, unk_frequency = list(), 0
    vocabulary_new = Counter()
    for wo in old_vocabulary:
        if old_vocabulary[wo] > 10:
            vocabulary_new[wo] = old_vocabulary[wo]
        else:
            unk_words.append(wo)
            unk_frequency = unk_frequency + old_vocabulary[wo]
    vocabulary_new['UNK'] = unk_frequency
    unk_words = set(unk_words)
    data_with_unk_new = put_unk_in_data(all_data, unk_words)
    return vocabulary_new, data_with_unk_new


def get_bi_gram_counts(data_with_unk_loc):
    """
    Computes the bi_gram frequencies for each pair of words in each sentence in the data
    :param data_with_unk_loc:
    :return:
    """
    bi_gram_counts_local = defaultdict(int)
    for sentence in data_with_unk_loc:
        for index in range(0, len(sentence) - 1):
            word1 = sentence[index]
            word2 = sentence[index + 1]
            bi_gram_counts_local[(word1, word2)] = bi_gram_counts_local[(word1, word2)] + 1
    return bi_gram_counts_local


def t_list_to_dict(t_list):
    """
    :param t_list:
    :return:
    """
    ret = {}
    for t in t_list:
        x, y = t
        ret[x] = y
    return ret


data = process_directory('brown_subset/')
original_vocabulary = get_vocabulary(data)
vocabulary_with_unk, data_with_unk = put_unk(original_vocabulary, data)
bi_gram_counts = get_bi_gram_counts(data_with_unk)
vocabulary_with_unk_sorted = sorted(vocabulary_with_unk.items(), key=lambda pair: (-pair[1], pair[0]))

with open('3-1-vocabulary-list.txt', 'w')as temp1:
    temp1.write(json.dumps(t_list_to_dict(vocabulary_with_unk_sorted)))

d = dict(bi_gram_counts)
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

with open('bi_grams-list.txt', 'w')as temp1:
    temp1.write(json.dumps(sorted_d))

N = sum(vocabulary_with_unk.values())


def merge_bg_counts(word1, word2):
    """
    Handles changes in bi-gram counts
    :param word1:
    :param word2:
    """
    vocabulary_with_unk[word1] = vocabulary_with_unk[word1] + vocabulary_with_unk[word2]
    del vocabulary_with_unk[word2]
    values = list(bi_gram_counts.items())
    for bi_gram, frequency in values:
        if word2 not in bi_gram:
            continue
        if bi_gram[0] == word2 and bi_gram[1] == word2:
            bi_gram_counts[(word1, word1)] = bi_gram_counts[(word1, word1)] + frequency
        elif bi_gram[0] == word2:
            bi_gram_counts[(word1, bi_gram[1])] = bi_gram_counts[(word1, bi_gram[1])] + frequency
        elif bi_gram[1] == word2:
            bi_gram_counts[(bi_gram[0], word1)] = bi_gram_counts[(bi_gram[0], word1)] + frequency
        del bi_gram_counts[(bi_gram[0], bi_gram[1])]


def get_ug_count(cluster_local, vocabulary_with_unk_local):
    """
    :param cluster_local:
    :param vocabulary_with_unk_local:
    :return: sum of uni-gram counts of each word in cluster
    """
    return float(sum(vocabulary_with_unk_local[wo] for wo in cluster_local))


def get_bg_count(pair, bi_gram_counts_local):
    """
    :param pair:
    :param bi_gram_counts_local:
    :return: sum of uni-gram counts of each pair of words in cluster
    """
    return float(sum(bi_gram_counts_local[(i, j)] for i in pair[0] for j in pair[1]))


def get_pair_quality(pair):
    """
    :param pair: a pair of clusters
    :return: the quality of the pair
    """
    bgc = get_bg_count(pair, bi_gram_counts) / N
    if not bgc:
        return 0.0
    ugc1 = get_ug_count(pair[0], vocabulary_with_unk)
    ugc2 = get_ug_count(pair[1], vocabulary_with_unk)
    return (bgc / N) * (math.log((bgc * (N * N)) / (ugc1 * ugc2), 10))


def get_pair_edge_weight(pair, pair_edge_weight_cache):
    """
    :param pair: a pair of clusters
    :param pair_edge_weight_cache:
    :return: the weight of edge between two clusters
    Ref: Percy Lang's thesis in reading material
    """
    if pair in pair_edge_weight_cache:
        return pair_edge_weight_cache[pair]
    if pair[0] == pair[1]:
        mi = get_pair_quality(pair)
    else:
        mi = get_pair_quality(pair) + get_pair_quality((pair[1], pair[0]))
    pair_edge_weight_cache[pair] = mi
    return mi


def get_weight_sum(clu, clus, pair_edge_weight_cache):
    """
    :param clu:
    :param clus:
    :param pair_edge_weight_cache:
    :return:
    """
    return sum(get_pair_edge_weight((clu, cl), pair_edge_weight_cache) for cl in clus)


def calculate_l(pair, clusters, pair_edge_weight_cache):
    """
    :param pair:
    :param clusters:
    :param pair_edge_weight_cache:
    :return: the value of L(c,c')
    Ref: Percy Lang's thesis in reading material and https://github.com/percyliang/brown-cluster
    """
    non_merge_clusters = tuple(cl for cl in clusters if cl != pair[0] and cl != pair[1])
    w12 = -get_pair_edge_weight(pair, pair_edge_weight_cache)
    w11 = -get_pair_edge_weight((pair[0], pair[0]), pair_edge_weight_cache)
    w22 = -get_pair_edge_weight((pair[1], pair[1]), pair_edge_weight_cache)
    w1212 = get_pair_edge_weight((pair[0] + pair[1], pair[0] + pair[1]), pair_edge_weight_cache)
    w1nm = -get_weight_sum(pair[0], non_merge_clusters, pair_edge_weight_cache)
    w2nm = -get_weight_sum(pair[1], non_merge_clusters, pair_edge_weight_cache)
    w12nm = get_weight_sum(pair[0] + pair[1], non_merge_clusters, pair_edge_weight_cache)
    delta = w12 + w11 + w22 + w1212 + w1nm + w2nm + w12nm
    return delta


def initialize_l(clusters, pair_edge_weight_cache):
    """
    :param clusters:
    :param pair_edge_weight_cache:
    :return:
    """
    pairs = combinations(clusters, 2)
    l_table = Counter()
    for pair in pairs:
        l_table[pair] = calculate_l(pair, clusters, pair_edge_weight_cache)
    return l_table


def get_new_l(pair1, pair2, clusters, pair_edge_weight_cache):  # improve
    """
    :param pair1:
    :param pair2:
    :param clusters:
    :param pair_edge_weight_cache:
    :return: new value of L
    """
    if pair1 in l_table:
        dt = l_table[pair1]
    elif (pair1[1], pair1[0]) in l_table:
        dt = l_table[pair1]
    else:
        return calculate_l(pair1, clusters, pair_edge_weight_cache)
    sum_old = sum(get_pair_edge_weight((pi, pj), pair_edge_weight_cache) for pi in pair1 for pj in pair2)
    sum_new = sum(get_pair_edge_weight((pi, pair2[0] + pair2[1]), pair_edge_weight_cache) for pi in pair1)
    return dt - sum_old + sum_new


def update_bitstring(clu1, clu2, clus):
    """
    :param clu1:
    :param clu2:
    :param clus:
    """
    for wo in clus[clu1]:
        s = bit_strings[wo[0]]
        bit_strings[wo[0]] = '0' + s
    for wo in clus[clu2]:
        s = bit_strings[wo[0]]
        bit_strings[wo[0]] = '1' + s


def merge_clusters(cluster1, cluster2, clusters, clusters_copy, bit_strings, l_table, pair_edge_weight_cache
                   , words_left):
    """
    Ref: Percy Lang's thesis in reading material and https://github.com/percyliang/brown-cluster
    :param cluster1:
    :param cluster2:
    :param clusters:
    :param clusters_copy:
    :param bit_strings:
    :param l_table:
    :param pair_edge_weight_cache:
    :param words_left:
    :return:
    """
    # update clusters and bit strings
    update_bitstring(cluster1, cluster2, clusters_copy)
    clusters_copy[cluster1] = clusters_copy[cluster1] + clusters_copy[cluster2]
    del clusters_copy[cluster2]
    # merge_back.append((cluster1, cluster2))
    del l_table[(cluster1, cluster2)]
    del l_table[(cluster2, cluster1)]
    next_word, words_left = words_left[0], words_left[1:]
    clusters_copy[next_word] = [next_word]
    non_merge_clusters = tuple([cl for cl in clusters if cl != cluster1 and cl != cluster2])
    clusters = non_merge_clusters + (cluster1, next_word)  # return C
    for i, j in combinations(non_merge_clusters, 2):
        l_table[(i, j)] = get_new_l((i, j), (cluster1, cluster2), clusters, pair_edge_weight_cache)
    merged_node = cluster1
    merge_bg_counts(cluster1[0], cluster2[0])
    for cl in non_merge_clusters:
        del l_table[(cl, cluster1)]
        del l_table[(cl, cluster2)]
        del l_table[(cluster1, cl)]
        del l_table[(cluster2, cl)]
        l_table[(cl, merged_node)] = calculate_l((cl, merged_node), clusters, pair_edge_weight_cache)
        l_table[(cl, next_word)] = calculate_l((cl, next_word), clusters, pair_edge_weight_cache)
    l_table[(merged_node, next_word)] = calculate_l((merged_node, next_word), clusters, pair_edge_weight_cache)
    return clusters, clusters_copy, bit_strings, l_table, pair_edge_weight_cache, words_left


def merge_till_single_cluster(cluster1, cluster2, clusters, clusters_copy, bit_strings, l_table
                              , pair_edge_weight_cache):
    """
    Ref: Percy Lang's thesis in reading material and https://github.com/percyliang/brown-cluster
    :param cluster1:
    :param cluster2:
    :param clusters:
    :param clusters_copy:
    :param bit_strings:
    :param l_table:
    :param pair_edge_weight_cache:
    :return:
    """
    update_bitstring(cluster1, cluster2, clusters_copy)
    clusters_copy[cluster1] = clusters_copy[cluster1] + clusters_copy[cluster2]
    del clusters_copy[cluster2]
    # merge_back.append((cluster1, cluster2))
    del l_table[(cluster1, cluster2)]
    del l_table[(cluster2, cluster1)]
    non_merge_clusters = tuple([cl for cl in clusters if cl != cluster1 and cl != cluster2])
    clusters = non_merge_clusters + (cluster1,)
    for i, j in combinations(non_merge_clusters, 2):
        l_table[(i, j)] = get_new_l((i, j), (cluster1, cluster2), clusters, pair_edge_weight_cache)
    merged_node = cluster1
    merge_bg_counts(cluster1[0], cluster2[0])
    for cl in non_merge_clusters:
        del l_table[(cl, cluster1)]
        del l_table[(cl, cluster2)]
        del l_table[(cluster1, cl)]
        del l_table[(cluster2, cl)]
        l_table[(cl, merged_node)] = calculate_l((cl, merged_node), clusters, pair_edge_weight_cache)
    return clusters, clusters_copy, bit_strings, l_table, pair_edge_weight_cache


# clustering starts here
k = 100
num_merges = 0
bit_strings = Counter()
clusters_copy = dict()
clusters = tuple([(i[0],) for i in vocabulary_with_unk_sorted[:k]])
words_left = [(i[0],) for i in vocabulary_with_unk_sorted[k:]]
pair_edge_weight_cache = dict()
for word in vocabulary_with_unk_sorted:
    bit_strings[word[0]] = ""
for cl in clusters:
    clusters_copy[cl] = [cl]
l_table = initialize_l(clusters, pair_edge_weight_cache)

print("Starting Clustering")

while words_left:
    num_merges = num_merges + 1
    (cluster1, cluster2), _ = l_table.most_common(1)[0]
    print("Merge number : ", num_merges)
    clusters, clusters_copy, bit_strings, l_table, pair_edge_weight_cache, words_left = \
        merge_clusters(cluster1, cluster2, clusters, clusters_copy, bit_strings, l_table, pair_edge_weight_cache
                       , words_left)
    pair_edge_weight_cache.clear()


# Write clusters to a file	
with open("clusters.txt", "w") as file:
    ind = 0
    for cls in clusters_copy.items():
        o, p = cls
        file.write(str(p))
        file.write("\n")
        ind = ind + 1


print("Now starting merge for a single cluster")

while len(clusters) != 1:
    (cluster1, cluster2), _ = l_table.most_common(1)[0]
    clusters, clusters_copy, bit_strings, l_table, pair_edge_weight_cache = \
        merge_till_single_cluster(cluster1, cluster2, clusters, clusters_copy, bit_strings, l_table
                                  , pair_edge_weight_cache)

# Write Bit String Represenatations to a file								  
with open("strings.txt", "w") as file:
    for cls in bit_strings.items():
        file.write(str(cls))
        file.write('\n')

print("Done")
