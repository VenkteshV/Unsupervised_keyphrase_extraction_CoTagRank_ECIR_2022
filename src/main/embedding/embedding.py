""" This file contains our core representation method mentioned in the paper
USE + LDA and plain USEEMbeddings methods"""
import re
from typing import Tuple
import torch 
from torch import nn,optim
import numpy as np
import unidecode
import tensorflow_hub as hub
import tensorflow as tf
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import sent2vec
from operator import add
from statistics import mean
from sentence_transformers import SentenceTransformer
import joblib
import time
import os
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))



# module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/6" 
# embed = hub.Module(module_url)




class PerturbMethods:
    REMOVE = 'remove'
    REPLACE = 'replace'


class Pooling:
    MEAN = 'mean'
    MAX = 'max'
    MIN = 'min'


class Embedding:
    def __init__(self, encoder):
        self.encoder = encoder

    def run(self, text, phrases):
        pass

class UseSentenceEmbedding(Embedding):
    def __init__(self, encoder):
        super().__init__(encoder)
        # g = tf.Graph()
        with tf.device('/CPU:0'):
        # We will be feeding 1D tensors of text into the graph.
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            
            #kindly replace the location in hub.module with the url commented out below

            # "https://tfhub.dev/google/universal-sentence-encoder-large/3"
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
            self.embedded_text = embed(self.text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        # g.finalize()

        self.session = tf.Session(config=tf.ConfigProto( allow_soft_placement=True))
        self.session.run(init_op)
        print("init _____")



    def get_tokenized_sents_embeddings_USE(self, sents,expand=False):
        # for sent in sents:
            # if '\n' in sent:
            #     raise RuntimeError('New line is not allowed inside a sentence')
           
        vectors_USE =  self.session.run(self.embedded_text, feed_dict={self.text_input: sents})
        # with self.session.as_default():
        #     vectors_USE = vectors_USE.eval()
        # if expand:
        #     None
        # else:
        #     self.session.close()
        return vectors_USE
    def run(self,doc_text, text, phrases, lda_model, dictionary, expand=False):
            joint_corpus = [doc_text for doc_text, _, _ in [(doc_text, 0, -1)] + phrases]
            # doc = nlp(text)
            # for sentence in doc.sents:


            embeddings = self.get_tokenized_sents_embeddings_USE(joint_corpus, expand)
            text_emb = np.array(embeddings[0])
            # text_emb = text_emb.reshape(1,-1)
            phrase_embs = np.array(embeddings[1:])

            return text_emb, phrase_embs

class TopicSentenceLDA(Embedding):
    def __init__(self, encoder):
        super().__init__(encoder)
        print("Now returning topical sentence + LDA embeddings")
        self.encoder =  SentenceTransformer(dir_path + '/../../data/bi_encoder_sentence_triplets')
    def fetch_word_vector_rep(self,phrases, lemmatizer, dictionary, K, distributions):
        try:
            phrase_vectors = []
            result = []
            for phrase in phrases:   
                word_vectors = [] 
                for word in phrase.split(' '):
                    if (word) in dictionary:
                        word_vectors.append([distributions[k][dictionary.index((word))] for k in range(K)])
                    else:
                        word_vectors.append([0]*500)
                if word_vectors:
                    phrase_vectors.append([sum(word_list) for word_list in zip(*word_vectors)])

            return np.vstack(phrase_vectors)
        except:
            return []

    def run(self,doc_text, text, phrases, lda_model, dictionary, expand=False):
        # print("text", text, phrases)
        joint_corpus = [text for text, _, _ in [(doc_text, 0, -1)] + phrases]
        embeddings = self.encoder.encode(joint_corpus)
        stoplist = stopwords.words('english')
        tf_vectorizer = CountVectorizer(stop_words=stoplist,
                                        vocabulary=dictionary)

        tf = tf_vectorizer.fit_transform(text)

        # compute the topic distribution over the document
        distribution_topic_document = lda_model.transform(tf)[0]

        # compute the word distributions over topics
        distributions = lda_model.components_ / lda_model.components_.sum(axis=1)[:,
                                            np.newaxis]

        text_emb = np.array(embeddings[0])
        text_emb = text_emb.reshape(1,-1)
        lemmatizer =  WordNetLemmatizer()
        K = len(distribution_topic_document)
        distribution_topic_document = distribution_topic_document.reshape(1,-1)
        word_vectors = self.fetch_word_vector_rep(joint_corpus[1:], lemmatizer, dictionary, K, distributions)   

        # print("word_vectors", word_vectors, word_vectors.shape)
        # print("joint_corpus",joint_corpus[1:])
        vectors_tpbert_LDA = np.c_[distribution_topic_document ,  text_emb.reshape(1,-1)]
        phrase_embs = np.array(embeddings[1:])
        # term_embeddings = np.c_[word_vectors ,  phrase_embs]
        # print("word_vectors",word_vectors, phrase_embs.shape)
        term_embeddings = np.hstack((word_vectors ,  phrase_embs))
        vectors_tpbert_LDA = vectors_tpbert_LDA.squeeze()

        # print("text_emb",vectors_lda_USE.shape)
        # print("phrase_embs", term_embeddings.shape)
        return vectors_tpbert_LDA, term_embeddings  

     
# this class implements our novel topically guided contrastively learnt sentence embeddings
class TopicalSentenceEmbedding(Embedding):
    def __init__(self, encoder):
        super().__init__(encoder)
        print("Now returning topical sentence embeddings")
        self.encoder =  SentenceTransformer(dir_path + '/../../data/bi_encoder_sentence_triplets')
    def run(self, text, phrases, method=None):
        # print("text", text, phrases)
        embeddings = self.encoder.encode([text for text, _, _ in [(text, 0, -1)] + phrases])


        text_emb = np.array(embeddings[0])
        phrase_embs = np.array(embeddings[1:])
        # print("text_emb",text_emb.shape)
        # print("phrase_embs", phrase_embs.shape)
        return text_emb, phrase_embs        
# this class implements our novel embedddingmethod LDA + USE
class UseEmbedding(Embedding):
    def __init__(self, encoder):
        super().__init__(encoder)
        # g = tf.Graph()
        with tf.device('/CPU:0'):
        # We will be feeding 1D tensors of text into the graph.
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            
            #kindly replace the location in hub.module with the url commented out below

            # "https://tfhub.dev/google/universal-sentence-encoder-large/3"
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
            self.embedded_text = embed(self.text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        # g.finalize()

        self.session = tf.Session(config=tf.ConfigProto( allow_soft_placement=True))
        self.session.run(init_op)
        print("init _____")



    def get_tokenized_sents_embeddings_USE(self, sents,expand=False):
        # for sent in sents:
            # if '\n' in sent:
            #     raise RuntimeError('New line is not allowed inside a sentence')
           
        vectors_USE =  self.session.run(self.embedded_text, feed_dict={self.text_input: sents})
        # with self.session.as_default():
        #     vectors_USE = vectors_USE.eval()
        # if expand:
        #     None
        # else:
        #     self.session.close()
        return vectors_USE

    def fetch_word_vector_rep(self,phrases, lemmatizer, dictionary, K, distributions):
        try:
            phrase_vectors = []
            result = []
            for phrase in phrases:   
                word_vectors = [] 
                for word in phrase.split(' '):
                    if (word) in dictionary:
                        word_vectors.append([distributions[k][dictionary.index((word))] for k in range(K)])
                    else:
                        word_vectors.append([0]*500)
                if word_vectors:
                    phrase_vectors.append([sum(word_list) for word_list in zip(*word_vectors)])

            return np.vstack(phrase_vectors)
        except:
            return []

    def run(self,doc_text, text, phrases, lda_model, dictionary, expand=False):
        joint_corpus = [doc_text for doc_text, _, _ in [(doc_text, 0, -1)] + phrases]
        # doc = nlp(text)
        # for sentence in doc.sents:
        stoplist = stopwords.words('english')
        tf_vectorizer = CountVectorizer(stop_words=stoplist,
                                        vocabulary=dictionary)

        tf = tf_vectorizer.fit_transform(text)

        # compute the topic distribution over the document
        distribution_topic_document = lda_model.transform(tf)[0]

        # compute the word distributions over topics
        distributions = lda_model.components_ / lda_model.components_.sum(axis=1)[:,
                                            np.newaxis]


        embeddings = self.get_tokenized_sents_embeddings_USE(joint_corpus, expand)
        text_emb = np.array(embeddings[0])
        importance_lda = 8

        text_emb = text_emb.reshape(1,-1)
        lemmatizer =  WordNetLemmatizer()
        K = len(distribution_topic_document)
        distribution_topic_document = distribution_topic_document.reshape(1,-1)
        word_vectors = self.fetch_word_vector_rep(joint_corpus[1:], lemmatizer, dictionary, K, distributions)   

        # print("word_vectors", word_vectors, word_vectors.shape)
        # print("joint_corpus",joint_corpus[1:])
        vectors_lda_USE = np.c_[distribution_topic_document ,  text_emb.reshape(1,-1)]
        phrase_embs = np.array(embeddings[1:])
        # term_embeddings = np.c_[word_vectors ,  phrase_embs]
        term_embeddings = np.hstack((word_vectors ,  phrase_embs))
        vectors_lda_USE = vectors_lda_USE.squeeze()

        # print("text_emb",vectors_lda_USE.shape)
        # print("phrase_embs", term_embeddings.shape)
        return vectors_lda_USE, term_embeddings

    def phrase_embeddings_expansion(self, phrases):
        embeddings =  self.get_tokenized_sents_embeddings_USE(phrases)
        return embeddings
        
class NaiveEmbedding(Embedding):
    def __init__(self, encoder):
        super().__init__(encoder)

    def run(self, text, phrases, method=None):
        # print("text", text, phrases)
        embeddings = self.encoder.encode([text for text, _, _ in [(text, 0, -1)] + phrases])
        text_emb = np.array(embeddings[0])
        phrase_embs = np.array(embeddings[1:])
        # print("text_emb",text_emb.shape)
        # print("phrase_embs", phrase_embs.shape)
        return text_emb, phrase_embs

class Sent2Vec(Embedding):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(dir_path+'/torontobooks_unigrams.bin')


    def run(self,text, phrases, method=None):
        if method == "EmbedRank":
            embeddings = self.model.embed_sentences([text for text in [(text)] + phrases])
        else:
            embeddings = self.model.embed_sentences([text for text,_,_ in [(text,0,-1)] + phrases])
        text_emb = np.array(embeddings[0])
        phrase_embs = np.array(embeddings[1:])
        # print("text_emb",text_emb.shape)
        # print("phrase_embs", phrase_embs.shape)
        return text_emb, phrase_embs