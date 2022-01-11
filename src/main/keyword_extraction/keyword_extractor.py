from main.rank.concept_expansion import ConceptExpansion
import  numpy as np
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
class KeywordExtractor:
    """Selects candidate phrases from input text, calculates their embeddings and applies Ranking algorithm
    to extract relevant keywords from text"""

    def __init__(self, phrase_extractor, embed, rank):
        self.phrase_extractor = phrase_extractor
        self.embed = embed
        self.rank = rank

    def run(self, doc_text, text=None, lda_model=None, dictionary=None, lists=None, method=None, highlight = False, expand=False):
        if method == 'CoTagRank' or  method == 'CoTagRankWindow' or method == "TopicCSRankLDA":
            phrases = self.phrase_extractor.run(doc_text, lists)
            # print("phrases", phrases)
            text_embedding, phrase_embeddings = self.embed.run(doc_text, text, phrases, lda_model, dictionary,False) 
            if len(phrases) >0:
                ranked_phrases, phrases_with_positions = self.rank.run(doc_text, phrases, text_embedding,
            phrase_embeddings, highlight)
            else:
                ranked_phrases = [(1.0, phrase[0]) for phrase in phrases]
                phrases_with_positions = phrases
            # print("ranked_phrases",ranked_phrases)
            if expand:
                color_map = []
                concept_expansion = ConceptExpansion()
                concepts = [concept for score, concept in ranked_phrases]
                similar_concepts, summaries = concept_expansion.expand_concepts(concepts)
                similar_concepts = [concept for concept in similar_concepts]
                for concept in concepts:
                    color_map.append('green')                    
                concepts.extend(similar_concepts)

                # print("wikipedia expanded concepts", similar_concepts)
                # expanded_phrases_embeddings = np.array(self.embed.phrase_embeddings_expansion(concepts))
                if len(similar_concepts) !=0:
                    concepts = [(concept.lower(), None, None) for  concept in similar_concepts]


                # concepts = [(concept,score) for score, concept in ranked_phrases]
                # similar_concepts = [concept.lower() for concept in similar_concepts]
                # concepts.extend(similar_concepts)
                    text_embedding, expanded_phrases_embeddings = self.embed.run(doc_text, text, concepts, lda_model, dictionary,expand) 


                # text_embedding = np.array(self.embed.phrase_embeddings_expansion([doc_text])[0])
                # phrase_embeddings = np.concatenate((phrase_embeddings, expanded_phrases_embeddings), axis=0)
                    ranked_expanded_phrases, phrases_with_positions = self.rank.run(doc_text, concepts, text_embedding,
                    expanded_phrases_embeddings, highlight)
                    for phrase in ranked_expanded_phrases:
                        color_map.append('blue')
                    # print("ranked_expanded_phrases", ranked_expanded_phrases)
                    ranked_phrases.extend(ranked_expanded_phrases)
                    # print("expanded concepts after reranking", ranked_phrases)
                    final_list_candidates = set()
                    selected_candidates=[]
                    for relevance,phrase in ranked_phrases:
                        if not phrase in final_list_candidates:
                            lemmatized_phrase = ' '.join(wnl.lemmatize(word) for word in phrase.split(" "))
                            final_list_candidates.add(lemmatized_phrase)
                            selected_candidates.append((relevance, lemmatized_phrase))
                return selected_candidates, phrases_with_positions,color_map
            return ranked_phrases, phrases_with_positions
        if lda_model != None:
            phrases = self.phrase_extractor.run(doc_text, lists)
            text_embedding, phrase_embeddings = self.embed.run(doc_text, text, phrases, lda_model, dictionary)
                       
        else:
            phrases = self.phrase_extractor.run(doc_text, lists= lists)
            text_embedding, phrase_embeddings = self.embed.run(doc_text, phrases,method)
        if method == "TopicCSRank" or method == "CoTagRankSentenceUSE":
            ranked_phrases, phrase_relevance = self.rank.run(doc_text, phrases, text_embedding,
                                                                phrase_embeddings)
        else:
            ranked_phrases, phrase_relevance, phrase_aliases = self.rank.run(doc_text, phrases, text_embedding,
                                                                phrase_embeddings)
        # print("ranked_phrases", ranked_phrases)
        if expand:
                color_map = []
                # ranked_phrases = [(keyword, score) for (keyword), score in zip(ranked_phrases, phrase_relevance) if keyword]
                concept_expansion = ConceptExpansion()
                if method=="CoTagRankSentenceUSE":
                    concepts = ranked_phrases
                elif lda_model!=None or method == "EmbedRankSentenceBERT":
                    keywords = [(keyword) for (keyword, _, _), score in zip(ranked_phrases, phrase_relevance) if keyword]
                    # print("ranked_phrases", keywords)
                    concepts = keywords
                else:
                    concepts =  ranked_phrases
                for concept in concepts:
                    color_map.append('green')  
                similar_concepts, summaries = concept_expansion.expand_concepts(concepts)
                # similar_concepts = [concept for concept in similar_concepts]

                if len(similar_concepts) !=0:
                    expanded_concepts = [concept.lower() for  concept in similar_concepts]
                    if lda_model != None:
                        expanded_concepts = [(concept.lower(), None, None) for  concept in similar_concepts]
                        text_embedding, expanded_phrases_embeddings = self.embed.run(doc_text, text, expanded_concepts, lda_model, dictionary)
                    elif method == "EmbedRankSentenceBERT" or method =="CoTagRanks2v" or method == "TopicCSRank":
                        expanded_concepts = [(concept.lower(), None, None) for  concept in similar_concepts]
                        text_embedding, expanded_phrases_embeddings = self.embed.run(doc_text, expanded_concepts)              
 
                    else:
                        text_embedding, expanded_phrases_embeddings = self.embed.run(doc_text, expanded_concepts)
                    if method == "CoTagRankSentenceUSE" or method=="TopicCSRank":
                        ranked_expanded_phrases, phrase_relevance_expanded = self.rank.run(doc_text, expanded_concepts, text_embedding,
                    expanded_phrases_embeddings)
                    else:
                        ranked_expanded_phrases, phrase_relevance_expanded, phrase_aliases = self.rank.run(doc_text, expanded_concepts, text_embedding,
                    expanded_phrases_embeddings)
                    for phrase in ranked_expanded_phrases:
                        color_map.append('blue')
                    if method == "CoTagRankSentenceUSE" or method=="TopicCSRank":
                        ranked_phrases = concepts
                    elif lda_model !=None or method == "EmbedRankSentenceBERT":
                        print("concepts", ranked_expanded_phrases)
                        ranked_phrases = [(ranked,None,None) for ranked in concepts]
                        ranked_expanded_phrases = [(ranked_phrase, None,None) for (ranked_phrase,_,_) in ranked_expanded_phrases]
                    # print("ranked_phrases*************", ranked_phrases)

                    # ranked_expanded_phrases = [(keyword, score) for keyword, score in zip(ranked_expanded_phrases, phrase_relevance) if keyword]
                    ranked_phrases.extend(ranked_expanded_phrases)
                    if method!="CoTagRanks2v" and method!="CoTagRankSentenceUSE" and method!="TopicCSRank":
                        phrase_relevance.extend(phrase_relevance_expanded)
                return ranked_phrases, phrase_relevance, color_map

        return ranked_phrases, phrase_relevance
