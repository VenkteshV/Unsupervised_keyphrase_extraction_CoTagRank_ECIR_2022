import os
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import pickle
import pandas as pd
import gzip
import numpy as np
import spacy
from main.keyword_extraction.helpers import init_nlp
import re
from pke.base import LoadFile
import joblib
from main.extraction.extractor import PhraseExtractor, PhraseHighlighter
from keep.utility import getlanguage, CreateKeywordsFolder, LoadFiles, Convert2TrecEval
from sklearn.model_selection import train_test_split

spacy_nlp = spacy.load('en_core_web_sm')


dir_path = os.path.dirname(os.path.realpath(__file__))

pathData = (dir_path + '/data')

phrase_extractor = PhraseExtractor(grammar =  "GRAMMAR1",np_method="GRAMMAR",
        np_tags = "NLTK",
        stopwords = "NLTK", nlp = init_nlp({"name":"spacy" , "model_name": "en_core_web_sm"}))
lan = 'en-US'

def clean_sentence(text):
    text = text.replace("\\\\.","")
    text = text.replace("\\","")
    text = text.replace("&nbsp","")
    text = text.replace("&thinsp","")
    text = text.replace("&times","")
    text = text.replace("&lambda","")
    text = text.replace("&rsquo","")
    text = text.replace("•", "")
    text = text.lstrip("\n")
    return text
def get_phrases(text):
    keyphrase_candidates = []
    phrases = phrase_extractor.run(text)
    keyphrase_candidates = [phrase[0] for phrase in phrases if len(phrase[0])>2]
    # print(keyphrase_candidates)
    return keyphrase_candidates
def group_docs_by_topics(data_df):
    topic_doc_dict = dict()
    topics = list(set(data_df["labels"].values))
    topic_doc_dict = {topic: [] for topic in topics}
    for index, row in data_df.iterrows():
        #     print("row**",row)
            sentences = []
            doc = spacy_nlp(row[0])
            for i, sent in enumerate(doc.sents):
                if sent.text.strip()!="" and len(sent.text.strip().split())>2:
                    # print("sent.text.strip()",sent.text.strip())

                    sentences.append(sent.text)
            if sentences!=[]:
                topic_doc_dict[row[1]].append(sentences)
    return topic_doc_dict
#     print(topic_doc_dict)
def group_phrases_by_topics(data):
    topic_doc_dict = dict()
    topics = list(set(data["labels"].values))
    topic_doc_dict = {topic: [] for topic in topics}
    for index, row in data.iterrows():
            topic_doc_dict[row[1]].append(row[0])
#     print(topic_doc_dict)
    joblib.dump(topic_doc_dict,"topic_grouped_phrases_500")


if __name__ == '__main__':
    data_docs = pd.read_csv("cbse_science_500.csv")
    train, val = train_test_split(data_docs, test_size=0.30)

    train["documents"] = train["documents"].apply(lambda x: clean_sentence(x))
    train.to_csv("cbse_science_500_sentences_data_train.csv",index=False)
    docs_grouped = group_docs_by_topics(train)
    joblib.dump(docs_grouped,"topic_grouped_docs_500_train")


    # train["documents"] = train["documents"].apply(lambda x: get_phrases(x))
#     print(data_docs)
    # train.to_csv("cbse_science_500_phrases_data_train.csv",index=False)
    val["documents"] = val["documents"].apply(lambda x: clean_sentence(x))
    val.to_csv("cbse_science_500_sentences_data_val.csv",index=False)
    docs_grouped_val = group_docs_by_topics(val)
    joblib.dump(docs_grouped_val,"topic_grouped_docs_500_val")


    # val["documents"] = val["documents"].apply(lambda x: get_phrases(x))
#     print(data_docs)
    # val.to_csv("cbse_science_500_phrases_data_val.csv",index=False)
    # data = pd.read_csv("cbse_science_500_phrases_data.csv")
    # data_grouped = group_phrases_by_topics(data)
    # data_docs = pd.read_csv("cbse_science_500.csv")
    # data_docs["documents"] = data_docs["documents"].apply(lambda x: clean_sentence(x))
