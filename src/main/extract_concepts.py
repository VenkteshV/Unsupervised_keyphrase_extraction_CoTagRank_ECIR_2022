import spacy
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from main.evaluation.embedrank_transformers import EmbedRankSentenceBERT,EmbedRankSentenceUSE,CoTagRankUSE

from main.keyword_extraction.helpers import init_nlp
from main.extraction.extractor import PhraseExtractor, PhraseHighlighter
import networkx as nx
import matplotlib.pyplot as plt 
from main.evaluation.embedrank import EmbedRank as ER
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def extract_concepts(text_1):
    nlp = spacy.load('en_core_web_sm')
    # corenlp = StanfordNLPLanguage(stanfordnlp.Pipeline(lang="en"))
    with open(dir_path+'/evaluation/en_kp_list', 'r', encoding='utf-8') as f:
        lists = f.read().split('\n')
        print('load kp_list done.')
    pathData = os.path.join(dir_path, '../data')
    dataset_name = 'Inspec'
    normalization = None
    numOfKeyphrases = 10

    expand = False
    corenlp_grammar = PhraseExtractor(grammar =  "GRAMMAR1",np_method="GRAMMAR",
            np_tags = "NLTK",
            stopwords = "NLTK", nlp = init_nlp({"name":"spacy" , "model_name": "en_core_web_sm"}))
    CoTagRankUSE_object = CoTagRankUSE(numOfKeyphrases, pathData, dataset_name,
                                                            normalization)


    keywords,_ = CoTagRankUSE_object.ExtractKeyphrases(text_1, highlight=True,  expand=expand)

    # CoTagRankUSE_object = ER(numOfKeyphrases, pathData, dataset_name,
    #                                                         normalization)
    # keywords, color_map = CoTagRankUSE_object.ExtractKeyphrases(text_1, expand=True)
    # phrase_selected = [(phrase[0].lstrip(),phrase[1],phrase[2]) for phrase in phrase_lists]
    # del color_map[-1]

    for keyword in keywords:
        print("\t", keyword)
    with open('results-extramarks.html', 'w') as file:
        file.write(PhraseHighlighter.to_html(text_1, keywords))
    return keywords


def expand_concepts(text_1):
    nlp = spacy.load('en_core_web_sm')
    # corenlp = StanfordNLPLanguage(stanfordnlp.Pipeline(lang="en"))
    with open(dir_path+'/evaluation/en_kp_list', 'r', encoding='utf-8') as f:
        lists = f.read().split('\n')
        print('load kp_list done.')
    corenlp_grammar = PhraseExtractor(grammar =  "GRAMMAR1",np_method="GRAMMAR",
         np_tags = "NLTK",
         stopwords = "NLTK", nlp = init_nlp({"name":"spacy" , "model_name": "en_core_web_sm"}))
    numOfKeyphrases = 10

    pathData = os.path.join(dir_path, '../data')
    dataset_name = 'Inspec'
    normalization = None
    expand = True
    CoTagRankUSE_object = CoTagRankUSE(numOfKeyphrases, pathData, dataset_name,
                                                            normalization)
    keywords,_,color_map = CoTagRankUSE_object.ExtractKeyphrases(text_1, highlight=True,  expand=expand)

    # CoTagRankUSE_object = ER(numOfKeyphrases, pathData, dataset_name,
    #                                                         normalization)
    # keywords, color_map = CoTagRankUSE_object.ExtractKeyphrases(text_1, expand=True)
    # phrase_selected = [(phrase[0].lstrip(),phrase[1],phrase[2]) for phrase in phrase_lists]
    # del color_map[-1]

    for keyword in keywords:
        print("\t", keyword)
    with open('results-extramarks.html', 'w') as file:
        file.write(PhraseHighlighter.to_html(text_1, keywords))
    return keywords