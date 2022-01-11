import pandas as pd
import glob
from statistics import mean
from sklearn.metrics import cohen_kappa_score

if __name__ == "__main__":
    # data = pd.read_excel("Evaluation.xlsx")
    # annotator1 = []
    # annotator2 = []
    # data["Keyword"] = data["Keyword"].str.replace('\d+','')
    # data["Keyword"] = data["Keyword"].str.replace(".",'')
    # for index, row in data.iterrows():
    #     annotator1.append(row["Relevance"])
    #     annotator2.append(row["annotator2"])
    # print(len(data["Relevance"].values),len(annotator2), len(annotator1))

    # # for input_file in glob.iglob("annotations/annotations/*"):
    # #     print(input_file)

    # #     annotator_2 = pd.read_csv(input_file)
    # #     for index, row in annotator_2.iterrows():
    # #         if row["relevance"] ==1:
    # #             annotator2.append(row["keywords"])
    # # print(len(annotator2))
    # cohen_kappa =[]
    # i=0
    # # while i +18 <= (int(len(annotator1))):
    # #     print(i,i+18)
    # #     print(annotator1[i:i+18])
    # #     cohen_kappa.append(cohen_kappa_score(annotator1[i:i+18],annotator2[i:i+18]))
    # #     print(mean(cohen_kappa ), cohen_kappa_score(annotator1,annotator2))
    # #     i = i+18
    # print("cohen's kappa score",cohen_kappa_score(annotator1,annotator2))
    data = pd.read_csv("Concept_expansion_evaluation_annotated.csv")
    annotator1 = []
    annotator2 = []
    data = data[data["label1"].notnull()]
    for index, row in data.iterrows():
        annotator1.append(row["label1"])
        annotator2.append(row["label2"])
    print(len(data["label1"].values),len(annotator2), len(annotator1))

    # for input_file in glob.iglob("annotations/annotations/*"):
    #     print(input_file)

    #     annotator_2 = pd.read_csv(input_file)
    #     for index, row in annotator_2.iterrows():
    #         if row["relevance"] ==1:
    #             annotator2.append(row["keywords"])
    # print(len(annotator2))
    cohen_kappa =[]
    i=0
    # while i +18 <= (int(len(annotator1))):
    #     print(i,i+18)
    #     print(annotator1[i:i+18])
    #     cohen_kappa.append(cohen_kappa_score(annotator1[i:i+18],annotator2[i:i+18]))
    #     print(mean(cohen_kappa ), cohen_kappa_score(annotator1,annotator2))
    #     i = i+18
    print("cohen's kappa score",cohen_kappa_score(annotator1,annotator2))