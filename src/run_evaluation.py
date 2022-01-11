import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from IPython.display import display
from keep import KEA
from keep import KPMiner
from keep import MultiPartiteRank
from keep import PositionRank
from keep import Rake
from keep import SIGTREC_Eval
from keep import SingleRank
from keep import TFIDF
from keep import TextRank
from keep import TopicRank
from keep import TopicalPageRank
from keep import YAKE

from main.evaluation.embedrank_transformers import TopicCSRankLDA,CoTagRankSentenceUSE,TopicCSRank,EmbedRankSentenceBERT,CoTagRankPositional,EmbedRankSentenceUSE,CoTagRankUSE, CoTagRankWindow, CoTagRanks2v
from main.evaluation.embedrank import EmbedRank as ER
# from main.evaluation.wordattraction import WordAttractionRank as WordAttractionRank

import argparse
import warnings

#CotagRankUSE is the main algorithm CoTagRank mentioned in the paper
def keyword_extraction(expand=False):
    for algorithm in ListOfAlgorithms:
        print("\n")
        print("----------------------------------------------------------------------------------------")
        # print(f"Preparing Evaluation for \033[1m{algorithm}\033[0m algorithm")

        for i in range(len(ListOfDatasets)):
            dataset_name = ListOfDatasets[i]
            print("\n----------------------------------")
            # print(f" dataset_name = {dataset_name}")
            print("----------------------------------")

            if algorithm == 'MultiPartiteRank':
                MultiPartiteRank_object = MultiPartiteRank(numOfKeyphrases, pathData, dataset_name)
                MultiPartiteRank_object.ExtractKeyphrases()
                MultiPartiteRank_object.Convert2Trec_Eval(EvaluationStemming)
            if algorithm == 'YAKE':
                YAKE_object = YAKE(numOfKeyphrases, pathData, dataset_name)
                YAKE_object.ExtractKeyphrases()
                YAKE_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TopicalPageRank':
                TopicalPageRank_object = TopicalPageRank(numOfKeyphrases, pathData, dataset_name, normalization)
                TopicalPageRank_object.ExtractKeyphrases()
                TopicalPageRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TopicRank':
                TopicRank_object = TopicRank(numOfKeyphrases, pathData, dataset_name)
                TopicRank_object.ExtractKeyphrases()
                TopicRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'PositionRank':
                PositionRank_object = PositionRank(numOfKeyphrases, pathData, dataset_name, normalization)
                PositionRank_object.ExtractKeyphrases()
                PositionRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'SingleRank':
                SingleRank_object = SingleRank(numOfKeyphrases, pathData, dataset_name, normalization)
                SingleRank_object.ExtractKeyphrases()
                SingleRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TextRank':
                TextRank_object = TextRank(numOfKeyphrases, pathData, dataset_name, normalization)
                TextRank_object.ExtractKeyphrases()
                TextRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'EmbedRankSentenceBERT':
                EmbedRankSentenceBERT_object = EmbedRankSentenceBERT(numOfKeyphrases, pathData, dataset_name,
                                                                     normalization)
                EmbedRankSentenceBERT_object.ExtractKeyphrases()
                EmbedRankSentenceBERT_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'EmbedRank':
                ER_object = ER(numOfKeyphrases, pathData, dataset_name, normalization)
                ER_object.ExtractKeyphrases()
                ER_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'CoTagRanks2v':
                CoTagRanks2v_object = CoTagRanks2v(numOfKeyphrases, pathData, dataset_name, normalization)
                CoTagRanks2v_object.ExtractKeyphrases(expand=expand)
                CoTagRanks2v_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'EmbedRankSentenceUSE':
                EmbedRankSentenceUSE_object = EmbedRankSentenceUSE(numOfKeyphrases, pathData, dataset_name,
                                                                     normalization)
                EmbedRankSentenceUSE_object.ExtractKeyphrases()
                EmbedRankSentenceUSE_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'CoTagRankUSE':
                print("pathData", pathData)
                ConceptRankUSE_object = CoTagRankUSE(numOfKeyphrases, pathData, dataset_name,
                                                                        normalization)
                                                        
                ConceptRankUSE_object.ExtractKeyphrases(expand=expand)
                ConceptRankUSE_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TopicCSRank':
                print("pathData", pathData)
                TopicCSRank_object = TopicCSRank(numOfKeyphrases, pathData, dataset_name,
                                                                        normalization)
                                                        
                TopicCSRank_object.ExtractKeyphrases(expand=expand)
                TopicCSRank_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'TopicCSRankLDA':
                print("pathData", pathData)
                TopicCSRankLDA_object = TopicCSRankLDA(numOfKeyphrases, pathData, dataset_name,
                                                                        normalization)
                                                        
                TopicCSRankLDA_object.ExtractKeyphrases(expand=expand)
                TopicCSRankLDA_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'CoTagRankWindow':
                print("pathData", pathData)
                ConceptRankWindow_object = CoTagRankWindow(numOfKeyphrases, pathData, dataset_name,
                                                                        normalization)
                                                        
                ConceptRankWindow_object.ExtractKeyphrases(expand=expand)
                ConceptRankWindow_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == 'CoTagRankPositional':
                print("pathData", pathData)
                CoTagRankPositional_object = CoTagRankPositional(numOfKeyphrases, pathData, dataset_name,
                                                                        normalization)
                                                        
                CoTagRankPositional_object.ExtractKeyphrases(expand=expand)
                CoTagRankPositional_object.Convert2Trec_Eval(EvaluationStemming)
            elif algorithm == "CoTagRankSentenceUSE":
                print("pathData", pathData)
                CoTagRankSentenceUSE_object = CoTagRankSentenceUSE(numOfKeyphrases, pathData, dataset_name,
                                                                        normalization)
                                                        
                CoTagRankSentenceUSE_object.ExtractKeyphrases(expand=expand)
                CoTagRankSentenceUSE_object.Convert2Trec_Eval(EvaluationStemming)

            elif algorithm == 'WordAttractionRank':
                print("pathData", pathData)
                WordAttractionRank_object = WordAttractionRank(numOfKeyphrases, pathData, dataset_name,
                                                                        normalization)
                                                        
                WordAttractionRank_object.ExtractKeyphrases(expand=expand)
                WordAttractionRank_object.Convert2Trec_Eval(EvaluationStemming)

def evaluation():
    for dataset in ListOfDatasets:
        print("\n")
        print("----------------------------------------------------------------------------------------")
        # print(f"Running Evaluation for \033[1m{dataset}\033[0m dataset")

        path2qrel_file = pathOutput+dataset+".qrel"
        datasetid = os.path.basename(path2qrel_file)

        resultsFiles = []
        for alg in ListOfAlgorithms:
            resultsFiles.append(pathOutput + dataset+"_"+alg + ".out")
        print("resultsFiles",resultsFiles)
        sig = SIGTREC_Eval()
        results = sig.Evaluate(path2qrel_file, datasetid, resultsFiles, measures, statistical_test, formatOutput)
        for res in results:
            if formatOutput == "df":
                display(res)
                res.to_csv("results-latest-"+dataset+".csv")
            else:
                print(res)


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Some algorithms have a normalization parameter which may be defined with None, stemming or lemmatization
    normalization = None  # Other options: "stemming" (porter) and "lemmatization"
    warnings.filterwarnings("ignore")

    # Num of Keyphrases do Retrieve
    numOfKeyphrases = 10
    expand= False

    # ListOfDatasets = ['110-PT-BN-KP', '500N-KPCrowd-v1.1', 'citeulike180',
    #                   'fao30', 'fao780', 'Inspec', 'kdd', 'Krapivin2009',
    #                   'Nguyen2007', 'pak2018', 'PubMed', 'Schutz2008', 'SemEval2010',
    #                   'SemEval2017', 'theses100', 'wiki20', 'www', 'cacic', 'wicc', 'WikiNews']

    # datasets used in paper -'Inspec', 'SemEval2017', 'SemEval2010', KhanAcad
    #warning SemEval 2010 can take a long time as they are lengthy 
    #scientific articles. KhanAcad mostly has shorter text but some long transcripts too
    # to evaluate quality of concepts extracted and might take time to run for all 2429 files
    # CoTagRankUSE corresponds to proposed algorithm CoTagRank
    ListOfDatasets = ['SemEval2010']
    parser = argparse.ArgumentParser()
    parser.add_argument("--expand", help="expand extracted concepts (works only with cotagrank)")
    args = parser.parse_args()

    if args.expand:
        expand = True
    else:
        expand = False

# 'MultiPartiteRank','WordAttractionRank','SingleRank', 'TopicalPageRank','TextRank','EmbedRankSentenceUSE',
#   'EmbedRankSentenceBERT',
# 'EmbedRank','CoTagRankWindow','CoTagRanks2v','CoTagRankPositional','CoTagRankUSE'
# 'SingleRank', 'TopicalPageRank','TextRank',EmbedRankSentenceUSE,
#  'CoTagRankUSE', 'EmbedRankSentenceBERT','EmbedRank'
    ListOfAlgorithms =  ['EmbedRankSentenceBERT','CoTagRankPositional'
]
# ,'MultiPartiteRank','CoTagRankWindow','CoTagRanks2v','CoTagRankUSE','CoTagRankPositional'
# 'SingleRank', 'TopicalPageRank','TextRank','EmbedRankSentenceUSE',
#  'CoTagRankUSE', 'EmbedRankSentenceBERT','MultiPartiteRank','CoTagRankWindow','CoTagRanks2v'
# ['EmbedRank', 'WordAttractionRank',
# 'SingleRank', 'TopicalPageRank','TextRank','EmbedRankSentenceUSE',
#  'CoTagRankUSE', 'EmbedRankSentenceBERT','MultiPartiteRank','CoTagRankWindow','EmbedRankGraph']


# 'SingleRank', 'TopicalPageRank','TextRank','EmbedRankSentenceUSE',
#  'CoTagRankUSE', 'EmbedRankSentenceBERT','MultiPartiteRank', 'YAKE']
#       'EmbedRank',
# 'SingleRank', 'TopicalPageRank','TextRank','EmbedRankSentenceUSE',
#  'CoTagRankUSE', 'EmbedRankSentenceBERT','MultiPartiteRank']
# #     'EmbedRank',
# 'SingleRank', 'TopicalPageRank','TextRank','EmbedRankSentenceUSE',
#  'CoTagRankUSE', 'EmbedRankSentenceBERT','MultiPartiteRank']

#     'EmbedRank'
# 'SingleRank', 'TopicalPageRank','TextRank','EmbedRankSentenceUSE',
#  'CoTagRankUSE', 'EmbedRankSentenceBERT','MultiPartiteRank']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataPath = (dir_path + '/data')
    pathData = dataPath
    pathOutput = pathData + "/conversor/output/"
    pathDataset = pathData + "/Datasets/"
    pathKeyphrases = pathData + "/Keywords/"

    EvaluationStemming = True  # performs stemming when comparing the results obtained from the algorithm with the ground-truth

    statistical_test = ["student"]  

    measures = [
          'P.10','recall.10','F1.10'    ]

    formatOutput = 'df'  # options: 'csv', 'html', 'json', 'latex', 'sql', 'string', 'df'
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # run keyword extraction
    # ------------------------------------------------------------------------------------------------------------------
    keyword_extraction(expand)

    # ------------------------------------------------------------------------------------------------------------------
    # run evaluation
    # ------------------------------------------------------------------------------------------------------------------
    evaluation()