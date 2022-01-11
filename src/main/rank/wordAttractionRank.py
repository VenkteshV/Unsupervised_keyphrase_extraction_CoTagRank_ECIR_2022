from nltk.corpus import stopwords

import warnings
import re
import networkx as nx
from main.rank.text_process import  filter_text, read_file, rm_tags, stem2word, get_phrases,get_phrases_new
try:
    from nltk.stem.porter import PorterStemmer
    STEM = True
except ImportError:
    print("warning: stem function is off")
    STEM = False

import csv
import itertools
import math
import os
import re

import numpy as np
from gensim import corpora, models, similarities

from main.rank.text_process import filter_text, read_file


def get_edge_freq(text_stemmed, window=2):
    """
    Return a dict, key is edge tuple, value is frequency
    :param text_stemmed: stemmed text of target doc
    :param window: slide window of word graph
    """
    edges = []
    edge_freq = {}
    tokens = text_stemmed.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i+window],2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
    for edge in edges:
        edge_freq[tuple(sorted(edge))] = edges.count(edge)# * 2 / (tokens.count(edge[0]) + tokens.count(edge[1]))
    return edge_freq

def calc_force(freq1, freq2, distance):
    return freq1 * freq2 / (distance * distance)
def dict2list(dict):
    output = []
    for key in dict:
        if isinstance(key, str):
            tmp = [key]
        else:
            tmp = list(key)
        tmp.append(dict[key])
        output.append(tmp)
    return output

def build_graph(edge_weight):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_weight)
    return graph
def calc_dice(freq1, freq2, edge_count):
    return 2 * edge_count / (freq1 + freq2)
class WordAttraction(object):
    """
    Implementation of methods proposed in:
    Corpus-independent Generic Keyphrase Extraction Using Word Embedding Vectors
    POS tag filtering is excluded from process as no POS tagging tool can be applied to all languages,
    and it's time-consuming.
    """
    def __init__(self, embedding_model):
        self.__accuracy, self.__running, self.__text_length = set(), set(), set()
        self.__word_embedding = embedding_model
        self.__MAX_DISTANCE = 100000000
        self.__unigram = None
        self.__bigram = None
        self.__graph = None
        if STEM:
            self.__wnl = PorterStemmer()

    def extract_main(self, text, output_score=True, stem=True,
                     max_words=2, damping=0.85, max_iter=100, converge_threshold=0.01):
        """
        main method for extracting keywords
        :param text: content from which keywords are extracted
        :param output_score: if True, return sorted list of tuples: (key, score). if False, return set of top keywords
        :param stem: if True, keywords returned are stemmed by PorterStemmer in nltk
        :param max_words: max number of keywords expected from text
        :param damping: damping factor for scoring
        :param max_iter: max number of iteration for scoring
        :param converge_threshold: converge condition for scoring
        """
        stemdict = stem2word(text)
        text_candidate = filter_text(text, with_tag=False)
        edge_freq = get_edge_freq(text_candidate, window=2)
        edge_weight = {}
        for edge in edge_freq:
            word1 = edge[0]
            word2 = edge[1]
            try:
                distance = 1 - wvmodel.similarity(stemdict[word1], stemdict[word2])
            except:
                distance = 1
            words = text_candidate.split()
            tf1 = words.count(word1)
            tf2 = words.count(word2)
            cf = edge_freq[edge]
            force = calc_force(tf1, tf2, distance)
            dice = calc_dice(tf1, tf2, cf)
            edge_weight[edge] = force * dice
        edges = dict2list(edge_weight)
        graph = build_graph(edges)
        pr = nx.pagerank(graph, alpha=damping)
        phrases = get_phrases_new(pr, graph, text, ng=3, pl2=0.6, pl3=0.3, with_tag=False)    
        return phrases[:max_words]