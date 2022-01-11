import spacy
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
from main.evaluation.embedrank_transformers import EmbedRankSentenceBERT,EmbedRankSentenceUSE,CoTagRankUSE, CoTagRankWindow, CoTagRankSentenceUSE, TopicCSRank

from main.keyword_extraction.helpers import init_nlp
from main.extraction.extractor import PhraseExtractor, PhraseHighlighter
import networkx as nx
import matplotlib.pyplot as plt 
import os
from main.evaluation.embedrank import EmbedRank as ER



if __name__ == '__main__':
    text_1 =                    """Four students performed an experiment to calculate the density of a stone. 
    While measuring the mass of the stone with the help of spring balance, the first student immersed the stone in water, 
    the second student immersed it in sulphuric acid, the third student immersed it in kerosene and 
    the fourth student allowed it to hang freely in air. The correct value of mass of the stone will be obtained by   
     In this experiment, we need true value of mass that can be measured by suspending the stone freely in air.
      When we immerse the stone in some liquid, the liquid will exert an upward buoyant force on the stone in upward direction. 
      This changes the reading of the spring balance and we will not get the correct value. 
"""


    
#        """Four students performed an experiment to calculate the density of a stone. 
#     While measuring the mass of the stone with the help of spring balance, the first student immersed the stone in water, 
#     the second student immersed it in sulphuric acid, the third student immersed it in kerosene and 
#     the fourth student allowed it to hang freely in air. The correct value of mass of the stone will be obtained by   
#      In this experiment, we need true value of mass that can be measured by suspending the stone freely in air.
#       When we immerse the stone in some liquid, the liquid will exert an upward buoyant force on the stone in upward direction. 
#       This changes the reading of the spring balance and we will not get the correct value. 
# """

    
#      """Four students performed an experiment to calculate the density of a stone. 
#     While measuring the mass of the stone with the help of spring balance, the first student immersed the stone in water, 
#     the second student immersed it in sulphuric acid, the third student immersed it in kerosene and 
#     the fourth student allowed it to hang freely in air. The correct value of mass of the stone will be obtained by   
#      In this experiment, we need true value of mass that can be measured by suspending the stone freely in air.
#       When we immerse the stone in some liquid, the liquid will exert an upward buoyant force on the stone in upward direction. 
#       This changes the reading of the spring balance and we will not get the correct value. 
# """
          
    #"""When a body is immersed fully or partially in a fluid, it experiences an upward buoyant force that is equal to the weight of the fluid displaced by it."""
    
      
    

    
     # Magnitude of the magnetic field produced is proportional to the amount of current and inversely proportional to the distance. Direction of the field lines is in accordance with the right hand thumb rule.
 
    
    
    #  """Passage of current through a straight conductor display some fixed pattern. Pick out the odd one.
    #             Field lines are unaffected by the quantity of current flowing.
    #             """
    
    
    # Magnitude of the magnetic field produced is proportional to the amount of current and inversely proportional to the distance. Direction of the field lines is in accordance with the right hand thumb rule.
    #  """The crack band approach for producing mesh 
    #             independent load–displacement curves for fracture in plain concrete 
    #             is based on the idea that the crack opening is transformed into inelastic 
    #             strain by distributing it over an element length dependent zone [5]. 
    #             This approach will only produce mesh independent load–displacement curves, 
    #             if the inelastic strain profiles in the finite element analysis are mesh size dependent. 
    #             This requirement is an important difference to the nonlocal model which is designed to produce both 
    #             mesh size independent load–displacement curves and strain profiles. In CDPM2, the crack band approach is 
    #             applied only to the tensile part of the damage algorithm by replacing the stress–inelastic strain law shown in Fig. 2(b) 
    #             by a stress–inelastic displacement law of the form(13)σ=ftexp(−ϵinhwft)if(ϵin>0)Here, wft is a crack opening threshold used 
    #             to control the slope of the softening curve and h is the width of the crack-band, which in the present study is equal to 
    #             the maximum dimension of the element along the principal direction of the strain tensor corresponding to the maximum tensile 
    #             principal strain at the onset of damage. 
    #             For the compressive part, a stress–inelastic strain law was used to determine
    #             the compressive damage parameter, since it was reported in [14] for columns 
    #             subjected to eccentric compression that inelastic strain profiles in compression do 
    #             not exhibit a mesh dependence which would satisfy the assumptions of the crack-band approach. 
    #             This approach of applying the crack-band approach only to the tensile part has already been successfully used in Grassl et al. [16]."""
    

    
    
    
    # """  Four students performed an experiment to calculate the density of a stone. 
    #         While measuring the mass of the stone with the help of spring balance, the first student immersed the stone in water, 
    #         the second student immersed it in sulphuric acid, the third student immersed it in kerosene and 
    #         the fourth student allowed it to hang freely in air. The correct value of mass of the stone will be obtained by   
    #         In this experiment, we need true value of mass that can be measured by suspending the stone freely in air.
    #         When we immerse the stone in some liquid, the liquid will exert an upward force on the stone in upward direction. 
    #         This changes the reading of the spring balance and we will not get the correct value. """
            
    
    
    #  """The crack band approach for producing mesh 
    #             independent load–displacement curves for fracture in plain concrete 
    #             is based on the idea that the crack opening is transformed into inelastic 
    #             strain by distributing it over an element length dependent zone [5]. 
    #             This approach will only produce mesh independent load–displacement curves, 
    #             if the inelastic strain profiles in the finite element analysis are mesh size dependent. 
    #             This requirement is an important difference to the nonlocal model which is designed to produce both 
    #             mesh size independent load–displacement curves and strain profiles. In CDPM2, the crack band approach is 
    #             applied only to the tensile part of the damage algorithm by replacing the stress–inelastic strain law shown in Fig. 2(b) 
    #             by a stress–inelastic displacement law of the form(13)σ=ftexp(−ϵinhwft)if(ϵin>0)Here, wft is a crack opening threshold used 
    #             to control the slope of the softening curve and h is the width of the crack-band, which in the present study is equal to 
    #             the maximum dimension of the element along the principal direction of the strain tensor corresponding to the maximum tensile 
    #             principal strain at the onset of damage. 
    #             For the compressive part, a stress–inelastic strain law was used to determine
    #             the compressive damage parameter, since it was reported in [14] for columns 
    #             subjected to eccentric compression that inelastic strain profiles in compression do 
    #             not exhibit a mesh dependence which would satisfy the assumptions of the crack-band approach. 
    #             This approach of applying the crack-band approach only to the tensile part has already been successfully used in Grassl et al. [16]."""
    

    #  """Voiceover: There are several different types of lung cancer,
    #             and to determine the type a patient has,
    #             cancer cells need to be taken
    #             from either fluid around the lungs,
    #             or from a lung tissue sample known as a biopsy
    #             or from a sputum sample.
    #             Then this sample is taken back to the lab,
    #             where the cells are looked at under a microscope,
    #             and a diagnosis is made based
    #             on some characteristics of the cell.
    #             There are two main categories of lung cancer,
    #             one being small cell lung cancer,
    #             and the other non-small cell lung cancer.
    #             Maybe you can already tell
    #             that these two main categories have to do
    #             with the actual size of the cell.
    #             For small cell lung cancer, this is a tiny cell,
    #             so I like to think of it as a baby cell.
    #             A baby doesn't have much distance
    #             from its head to i's toes, right?
    #             Well, a small cell, then, doesn't have much distance
    #             from one side to the other.
    #             That means its nucleus and cell wall
    #             are close to each other.
    #             Also like a baby, this particular type of cell
    #             is not fully developed.
    #             Small cell lung cancer typically occurs in females.
    #             Let me draw her here with a pink bow,
    #             and give her a cigarette, because this occurs
    #             in females with a long history of smoking.
    #             A thing to keep in mind about this type of lung cancer,
    #             is that it divides quickly,
    #             and spreads rapidly throughout the body."""
    
    
    
    
#      """  Four students performed an experiment to calculate the density of a stone. 
#     While measuring the mass of the stone with the help of spring balance, the first student immersed the stone in water, 
#     the second student immersed it in sulphuric acid, the third student immersed it in kerosene and 
#     the fourth student allowed it to hang freely in air. The correct value of mass of the stone will be obtained by   
#      In this experiment, we need true value of mass that can be measured by suspending the stone freely in air.
#       When we immerse the stone in some liquid, the liquid will exert an upward force on the stone in upward direction. 
#       This changes the reading of the spring balance and we will not get the correct value. 
# """
    

#     """The crack band approach for producing mesh 
#     independent load–displacement curves for fracture in plain concrete 
#     is based on the idea that the crack opening is transformed into inelastic 
#     strain by distributing it over an element length dependent zone [5]. 
#     This approach will only produce mesh independent load–displacement curves, 
#     if the inelastic strain profiles in the finite element analysis are mesh size dependent. 
#     This requirement is an important difference to the nonlocal model which is designed to produce both 
#     mesh size independent load–displacement curves and strain profiles. In CDPM2, the crack band approach is 
#     applied only to the tensile part of the damage algorithm by replacing the stress–inelastic strain law shown in Fig. 2(b) 
#     by a stress–inelastic displacement law of the form(13)σ=ftexp(−ϵinhwft)if(ϵin>0)Here, wft is a crack opening threshold used 
#     to control the slope of the softening curve and h is the width of the crack-band, which in the present study is equal to 
#     the maximum dimension of the element along the principal direction of the strain tensor corresponding to the maximum tensile 
#     principal strain at the onset of damage. 
#     For the compressive part, a stress–inelastic strain law was used to determine
#      the compressive damage parameter, since it was reported in [14] for columns 
#      subjected to eccentric compression that inelastic strain profiles in compression do 
#      not exhibit a mesh dependence which would satisfy the assumptions of the crack-band approach. 
#      This approach of applying the crack-band approach only to the tensile part has already been successfully used in Grassl et al. [16].
# """
#     """Voiceover: There are several different types of lung cancer,
# and to determine the type a patient has,
# cancer cells need to be taken
# from either fluid around the lungs,
# or from a lung tissue sample known as a biopsy
# or from a sputum sample.
# Then this sample is taken back to the lab,
# where the cells are looked at under a microscope,
# and a diagnosis is made based
# on some characteristics of the cell.
# There are two main categories of lung cancer,
# one being small cell lung cancer,
# and the other non-small cell lung cancer.
# Maybe you can already tell
# that these two main categories have to do
# with the actual size of the cell.
# For small cell lung cancer, this is a tiny cell,
# so I like to think of it as a baby cell.
# A baby doesn't have much distance
# from its head to i's toes, right?
# Well, a small cell, then, doesn't have much distance
# from one side to the other.
# That means its nucleus and cell wall
# are close to each other.
# Also like a baby, this particular type of cell
# is not fully developed.
# Small cell lung cancer typically occurs in females.
# Let me draw her here with a pink bow,
# and give her a cigarette, because this occurs
# in females with a long history of smoking.
# A thing to keep in mind about this type of lung cancer,
# is that it divides quickly,
# and spreads rapidly throughout the body."""
    
    
    
#      """  Petiole in a plant is the part of a     
#  The main parts of a leaf are petiole and lamina. 
#  The petiole is a small stalk by which a leaf is attached to a stem. 
#  The broad green part of the leaf is lamina.     """
    
#      """Voiceover: There are several different types of lung cancer,
# and to determine the type a patient has,
# cancer cells need to be taken
# from either fluid around the lungs,
# or from a lung tissue sample known as a biopsy
# or from a sputum sample.
# Then this sample is taken back to the lab,
# where the cells are looked at under a microscope,
# and a diagnosis is made based
# on some characteristics of the cell.
# There are two main categories of lung cancer,
# one being small cell lung cancer,
# and the other non-small cell lung cancer.
# Maybe you can already tell
# that these two main categories have to do
# with the actual size of the cell.
# For small cell lung cancer, this is a tiny cell,
# so I like to think of it as a baby cell.
# A baby doesn't have much distance
# from its head to i's toes, right?
# Well, a small cell, then, doesn't have much distance
# from one side to the other.
# That means its nucleus and cell wall
# are close to each other.
# Also like a baby, this particular type of cell
# is not fully developed.
# Small cell lung cancer typically occurs in females.
# Let me draw her here with a pink bow,
# and give her a cigarette, because this occurs
# in females with a long history of smoking.
# A thing to keep in mind about this type of lung cancer,
# is that it divides quickly,
# and spreads rapidly throughout the body."""

    # stanfordnlp.download('en')
    nlp = spacy.load('en_core_web_sm')
    # corenlp = StanfordNLPLanguage(stanfordnlp.Pipeline(lang="en"))
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(dir_path+'/main/evaluation/en_kp_list', 'r', encoding='utf-8') as f:
        lists = f.read().split('\n')
        print('load kp_list done.')
    corenlp_grammar = PhraseExtractor(grammar =  "GRAMMAR1",np_method="GRAMMAR",
         np_tags = "NLTK",
         stopwords = "NLTK", nlp = init_nlp({"name":"spacy" , "model_name": "en_core_web_sm"}))
    numOfKeyphrases = 15

    pathData = dir_path+'/data'
    dataset_name = 'SemEval2017'
    normalization = None
    expand = True
    CoTagRankUSE_object = TopicCSRank(numOfKeyphrases, pathData, dataset_name,
                                                            normalization)

    keywords,_,color_map = CoTagRankUSE_object.ExtractKeyphrases(text_1, highlight=True,  expand=expand)

    color_map = color_map[:len(keywords)]

    # CoTagRankUSE_object = ER(numOfKeyphrases, pathData, dataset_name,
    #                                                         normalization)
    # keywords, color_map = CoTagRankUSE_object.ExtractKeyphrases(text_1, expand=True)
    # phrase_selected = [(phrase[0].lstrip(),phrase[1],phrase[2]) for phrase in phrase_lists]
    # del color_map[-1]

    for keyword in keywords:
        print("\t", keyword)
    with open('results-expanded.html', 'w') as file:
        file.write(PhraseHighlighter.to_html(text_1, keywords))
    graph = nx.Graph()
    graph.add_edges_from(
        [(v[0], u[0]) for v in keywords for u in keywords if u != v])
    if expand:
        nx.draw_networkx(graph, with_labels=True, node_color = color_map,  edgelist =[])
    else:
        nx.draw_networkx(graph, with_labels=True,  edgelist =[])
    plt.show()  
