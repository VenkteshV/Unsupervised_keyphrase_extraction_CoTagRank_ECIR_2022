import os
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import pickle
import pandas as pd
import gzip
import numpy as np
import re
from pke.base import LoadFile
from keep.utility import getlanguage, CreateKeywordsFolder, LoadFiles, Convert2TrecEval

dir_path = os.path.dirname(os.path.realpath(__file__))

pathData = (dir_path + '/data')


lan = 'en-US'


def display_topics(model, feature_names, no_top_words,topic_index):
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx == topic_index:
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
                            
def load_lda_model(lda_model):
        model = LatentDirichletAllocation()
        with gzip.open(lda_model, 'rb') as f:
            (dictionary,
                model.components_,
                model.exp_dirichlet_component_,
                model.doc_topic_prior_) = pickle.load(f)
        return model, dictionary
def LoadDatasetFiles(dataset_name):
    # Gets all files within the dataset fold
    listFile = LoadFiles(pathToDatasetName + '/docsutf8/*')
    print(f"\ndatasetID = {dataset_name}; Number of Files = "
            f"{len(listFile)}; Language of the Dataset = {lan}")
    return listFile

def get_topic_label(text,dictionary,lda_model):
    stoplist = stopwords.words('english')
    tf_vectorizer = CountVectorizer(stop_words=stoplist,
                                    vocabulary=dictionary)

    tf = tf_vectorizer.fit_transform(text)

    # compute the topic distribution over the document
    distribution_topic_document = lda_model.transform(tf)[0]
    display_topics(lda_model, tf_vectorizer.get_feature_names(),10,np.argmax(distribution_topic_document))
    # print(distribution_topic_document,distribution_topic_document[np.argmax(distribution_topic_document    )] ,np.argmax(distribution_topic_document    ))
    return np.argmax(distribution_topic_document)

if __name__ == '__main__':
    documents = []
    print("Enter dataset name")
    dataset = input()

    lda_model = pathData + "/Models/Unsupervised/lda/" + dataset + '_lda_500.gz'

    model,dictionary = load_lda_model(lda_model)
    pathToDatasetName = pathData + "/Datasets/" + dataset
    list_of_docs = LoadDatasetFiles(dataset)
    topic_labels = []

    for i, doc in enumerate(list_of_docs):
        try:
            with open(doc, 'r') as doc_reader:
                print(doc)
                doc_text = doc_reader.read()
                doc_text = re.sub(r'\d:\d\d', r'', doc_text)
                doc_text = re.sub(r'\d\d:\d\d', r'', doc_text)

                print(doc,doc_text)
                document = LoadFile()
                document.load_document(input=doc_text,
                    language='en',
                    normalization='stemming')
                texts = []
                text = []

                # # loop through sentences
                for sentence in document.sentences:
                    # get the tokens (stems) from the sentence if they are not
                    # punctuation marks 
                    text.extend([sentence.stems[i] for i in range(sentence.length)
                                if sentence.pos[i] != 'PUNCT' and
                                sentence.pos[i].isalpha()])

                # add the document to the texts container
                texts.append(' '.join(text))
                topic_label = get_topic_label(texts, dictionary,model)
                documents.append(doc_text)
                topic_labels.append(topic_label)
                dict_data = {'documents':documents, 'labels':topic_labels}
                dataframe = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_data.items() ]))
                dataframe['labels'].apply(pd.to_numeric)
                print(dataframe)
                dataframe.to_csv('{}_500.csv'.format(dataset),index=False)
        except:
            print("exception")
            continue
            