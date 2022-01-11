import os
import sys
import csv
import math
import glob
import pickle
import gzip
import json
from multiprocessing import Pool,Process,Manager

from sklearn.model_selection import GridSearchCV


from pke.base import LoadFile
from pke.base import ISO_to_language

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# def read_files(file):
#     print("input_file", file)

#             # initialize load file object to load the text files
#     doc = LoadFile()

#     # read the input file using utils from pke repository
#     doc.load_document(input=file,
#                     language="en",
#                     normalization="stemming",
#                     max_length=10**6)

#     # current document placeholder
#     text = []

#     # loop through sentences
#     for sentence in doc.sentences:
#         # get the tokens (stems) from the sentence if they are not
#         # punctuation marks 
#         text.extend([sentence.stems[i] for i in range(sentence.length)
#                     if sentence.pos[i] != 'PUNCT' and
#                     sentence.pos[i].isalpha()])

#     # add the document to the texts container
#     return ' '.join(text)
"""methd to train lda on any dataset modified from
implementation in https://github.com/boudinfl/pke/blob/06b8017bb6ef2c247ba19e65cb23ff39a4df7a53/pke/utils.py
We use this to train LDA model on Khan academy data collected by us"""

def compute_lda_model(input_dir,
                      output_file,
                      n_topics=500,
                      extension="txt",
                      language="en",
                      normalization="stemming",
                      max_length=10**6):
    """Compute a LDA model from a collection of documents. Latent Dirichlet
    Allocation is computed using sklearn module.
    Args:
        input_dir (str): the input directory.
        output_file (str): the output file.
        n_topics (int): number of topics for the LDA model, defaults to 500.
        extension (str): file extension for input documents, defaults to xml.
        language (str): language of the input documents, used for stop_words
            in sklearn CountVectorizer, defaults to 'en'.
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
    """

    # texts container
    texts = []
    file_list = []
    pool =Pool(20)

    for input_file in glob.iglob(input_dir + '/*.' + extension):
        file_list.append(input_file)
    # texts.append(pool.map(read_files, file_list))

    # loop throught the documents
    for input_file in file_list:
        print("input_file", input_file)

        # initialize load file object to load the text files
        doc = LoadFile()

        # read the input file using utils from pke repository
        doc.load_document(input=input_file,
                          language=language,
                          normalization=normalization,
                          max_length=max_length)

        # current document placeholder
        text = []

        # loop through sentences
        for sentence in doc.sentences:
            # get the tokens (stems) from the sentence if they are not
            # punctuation marks 
            text.extend([sentence.stems[i] for i in range(sentence.length)
                         if sentence.pos[i] != 'PUNCT' and
                         sentence.pos[i].isalpha()])

        # add the document to the texts container
        texts.append(' '.join(text))

    tf_vectorizer = CountVectorizer(
        stop_words=stopwords.words(ISO_to_language[language]))
    tf = tf_vectorizer.fit_transform(texts)

    vocabulary = tf_vectorizer.get_feature_names()
    print("n_topics",n_topics)
    #  LDA model and training loop
    lda_model = LatentDirichletAllocation(n_components=n_topics,
                                          random_state=0,
                                          learning_method='batch')
    lda_model.fit(tf)


    # Options to try with our LDA
    # Beware it will try *all* of the combinations, so it'll take ages
    # search_params = {
    # 'n_components': [500, 100, 1500, 1000, 2000, 300],
    # }

    # # Set up LDA with the options we'll keep static
    # model = LatentDirichletAllocation(learning_method='batch')

    # # Try all of the options
    # gridsearch = GridSearchCV(model, param_grid=search_params, n_jobs=-1, verbose=1)
    # gridsearch.fit(tf)

    # # What did we find?
    # print("Best Model's Params: ", gridsearch.best_params_)
    # print("Best Log Likelihood Score: ", gridsearch.best_score_)
    # model save
    saved_model = (vocabulary,
                   lda_model.components_,
                   lda_model.exp_dirichlet_component_,
                   lda_model.doc_topic_prior_)


    # # create directories from path if not exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # dump the LDA model
    with gzip.open(output_file, 'wb') as fp:
        pickle.dump(saved_model, fp)


if __name__ == "__main__":
    input_path = "src/data/Datasets/keyphrase_expansion/docsutf8"
    output_path = "keyphrase_expansion_lda.gz"
    compute_lda_model(
        input_path,
        output_file = output_path
    )