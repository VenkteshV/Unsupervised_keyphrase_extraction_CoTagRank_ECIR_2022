import os
import json
import requests
from keep.utility import getlanguage, CreateKeywordsFolder, LoadFiles, Convert2TrecEval
from helpers import read_json

from main.keyword_extraction.helpers import init_keyword_extractor, init_nlp

dir_path = os.path.dirname(os.path.realpath(__file__))


class EmbedRank(object):
    def __init__(self, numOfKeywords, pathData, dataset_name, normalization):
        self.__lan = getlanguage(pathData + "/Datasets/" + dataset_name)
        self.__numOfKeywords = numOfKeywords
        self.__dataset_name = dataset_name
        self.__normalization = normalization
        self.__pathData = pathData
        self.__pathToDFFile = self.__pathData + "/Models/Unsupervised/dfs/" + self.__dataset_name + '_dfs.gz'
        self.__pathToDatasetName = self.__pathData + "/Datasets/" + self.__dataset_name
        self.__keywordsPath = self.__pathData + '/Keywords/EmbedRank/' + self.__dataset_name
        self.__outputPath = self.__pathData + "/conversor/output/"
        self.__algorithmName = "EmbedRank"
        
        self.model = init_keyword_extractor(read_json(dir_path+'/config/embedRank.json'))

    def LoadDatasetFiles(self):
        # Gets all files within the dataset fold
        listFile = LoadFiles(self.__pathToDatasetName + '/docsutf8/*')
        print(f"\ndatasetID = {self.__dataset_name}; Number of Files = "
              f"{len(listFile)}; Language of the Dataset = {self.__lan}")
        return listFile

    def CreateKeywordsOutputFolder(self):
        # Set the folder where keywords are going to be saved
        CreateKeywordsFolder(self.__keywordsPath)

    def runSingleDoc(self, doc, text = None, expand=False):
        try:
            # read raw document
            # print("doc",doc)
            if text:
                doc_text = text
                doc=text
            else:
                with open(doc, 'r') as doc_reader:
                    doc_text = doc_reader.read()
            if self.__dataset_name == "SemEval2010":
                if len(doc_text.split("INTRODUCTION")) > 1:
                    doc_text_abstract = doc_text.split("INTRODUCTION")[0]

                    doc_text_intro_partial = " ".join(doc_text.split("INTRODUCTION")[1].split(" ")[:150])
                else:
                    doc_text_abstract = " ".join(doc_text.split(" ")[:400])
                    doc_text_intro_partial = " "
                doc_text = doc_text_abstract+" "+doc_text_intro_partial
            if self.__dataset_name == "NLM500":
                doc_text_abstract_intro = " ".join(doc_text.split(" ")[:400])
                doc_text = doc_text_abstract_intro
            # extract keywords
            if expand:
                keywords, relevance, color_map = self.model.run(doc_text,expand=expand, method = "EmbedRank")
            else:
                keywords, relevance = self.model.run(doc_text,expand=expand, method = "EmbedRank")
            keywords = [(keyword, score) for (keyword), score in zip(keywords, relevance) if keyword]
        except e:
            print(e)

            keywords = []
        if expand:
            return keywords, color_map
        return keywords

    def runMultipleDocs(self, listOfDocs, expand=False):
        self.CreateKeywordsOutputFolder()

        for j, doc in enumerate(listOfDocs):
            # docID keeps the name of the file (without the extension)
            docID = '.'.join(os.path.basename(doc).split('.')[0:-1])

            keywords = self.runSingleDoc(doc, expand)

            # Save the keywords; score (on Algorithms/NameOfAlg/Keywords/NameOfDataset
            with open(os.path.join(self.__keywordsPath, docID), 'w', encoding="utf-8") as out:
                for (key, score) in keywords:
                    out.write(f'{key} {score}\n')

            # Track the status of the task
            print(f"\rFile: {j + 1}/{len(listOfDocs)}", end='')

        print(f"\n100% of the Extraction Concluded")

    def ExtractKeyphrases(self,text=None,  expand=False):


        print(f"\n\n-----------------Extract Keyphrases--------------------------")
        listOfDocs = self.LoadDatasetFiles()
        if text:
            return self.runSingleDoc(None, text, expand)
        self.runMultipleDocs(listOfDocs, expand)

    def Convert2Trec_Eval(self, EvaluationStemming=False):
        Convert2TrecEval(self.__pathToDatasetName, EvaluationStemming, self.__outputPath, self.__keywordsPath,
                         self.__dataset_name, self.__algorithmName)
