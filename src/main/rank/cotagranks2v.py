import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class Rank:
    def __init__(self):
        pass

    def run(self, text, phrases, text_embedding, phrase_embeddings):
        pass


class EmbedMethods:
    NAIVE = 'naive'


class CoTagRanks2v:
    """Implementation of unsupervised `phrase` extraction method using DNN embeddings and pagrank on complete graph. This method tries to
    Find important phrases in text using analysis of their cosine similarity to original text (informativeness) and elated the how related the phrases are to each other.

         phrase: i.e. `noun phrases` from (spacy) which are actually chunks of nouns that represent
         important parts of sentence. This is assumed to be good selection of candidates for phrases.
         DNN: any model which gives good text embeddings optimized for cosine similarity search."""

    def __init__(self, emb_method='NAIVE', mmr_beta=0.55, top_n=10, alias_threshold=0.8):
        """Takes spaCy's language model, dnn encoder model and loss parameter"""
        self.emb_method = getattr(EmbedMethods, emb_method)
        self.min_alpha = 0.001
        self.mmr_beta = mmr_beta
        self.top_n = top_n
        self.alias_threshold = alias_threshold

    def run(self, text, phrases, text_emb, phrase_embs):
        top_phrases, relevance, aliases = self.graphRank(text_emb, phrases, phrase_embs, self.mmr_beta,
                                                   self.top_n, self.alias_threshold)

        return top_phrases, relevance, aliases

    def graphRank(self, text_emb, phrases, phrase_embs, beta=0.55, top_n=15, alias_threshold=0.8):
        """Implementation of graph based ranking extension to embedrank to get top N relevant phrases to text
        """
        # calculate similarities of phrases with text and between phrases
        # print("cotagranks2v phrase_embs",phrases,text_emb.shape)
        text_sims = cosine_similarity(phrase_embs, [text_emb])
        phrase_sims = cosine_similarity(phrase_embs)
        # normalize cosine similarities
        text_sims_norm = self.standardize_normalize_cosine_similarities(text_sims)


        # keep indices of selected and unselected phrases in list
        selected_phrase_indices = []
        unselected_phrase_indices = list(range(len(phrases)))

        document_relevance = (text_sims_norm[unselected_phrase_indices]).squeeze(axis=1).tolist()
        # aliases_phrases = self.get_alias_phrases(phrase_sims[unselected_phrase_indices, :], phrases, alias_threshold)
        top_phrases = [phrases[idx] for idx in unselected_phrase_indices]

        relevance_dict = {}
        for (keyword,_,_), score in zip(top_phrases, document_relevance):
            relevance_dict[keyword] = score
        phrase_to_embedding = {}
        for index, phrase in enumerate(phrases):
            phrase_to_embedding[phrase[0]] = phrase_embs[index]
        # norm = sum(relevance_dict.values())
        # # for word in relevance_dict:
        # #     relevance_dict[word] /= norm
        graph = nx.Graph()
        minx =-1
        maxx =1
        graph.add_weighted_edges_from(
            [(v[0], u[0], ((np.dot(phrase_to_embedding[v[0]], phrase_to_embedding[u[0]])/((np.linalg.norm(phrase_to_embedding[v[0]]) * np.linalg.norm(phrase_to_embedding[u[0]]))+1e-07)) - minx) / (maxx-minx) )for v in top_phrases for u in top_phrases if u != v])
        # print("edges",graph.edges.data(),len(top_phrases), len(graph.edges.data()))
        try:
            pr = nx.pagerank(graph, personalization=relevance_dict,
                        alpha=0.85,
                        tol=0.0001, weight='weight')
        except:
            None
            # pr = nx.pagerank(graph, personalization=relevance_dict,
            #             alpha=1.0,max_iter=400,
            #              weight='weight')

        concepts = sorted([(b, a.lstrip()) for a, b in pr.items()], reverse=True)[:top_n]
        # concepts = [(score, keyword.lstrip()) for (keyword, _, _), score in zip(top_phrases, relevance)]
        
        # if highlight:
        #     phrases_only = [phrase[0].lstrip() for phrase in phrases]
        #     # print("concepts", concepts,'phrases', phrases_only)
        #     phrases_selected = [phrases[phrases_only.index(phrase[1])] for phrase in concepts ]
        #     return concepts, phrases_selected
        print("concepts",concepts)

        return concepts, None, None

    @staticmethod
    def standardize_normalize_cosine_similarities(cosine_similarities):
        """Normalized cosine similarities"""
        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.min(cosine_similarities)) / (
                np.max(cosine_similarities) - np.min(cosine_similarities) + 1e-07)

        # standardize and shift by 0.5
        cosine_sims_norm = 0.5 + (cosine_sims_norm - np.mean(cosine_sims_norm)) / ( np.std(cosine_sims_norm) + 1e-07)

        return cosine_sims_norm

    @staticmethod
    def max_normalize_cosine_similarities_pairwise(cosine_similarities):
        """Normalized cosine similarities of pairs which is 2d matrix of pairwise cosine similarities"""
        cosine_sims_norm = np.copy(cosine_similarities)
        np.fill_diagonal(cosine_sims_norm, np.NaN)

        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.nanmin(cosine_similarities, axis=0)) / (
                np.nanmax(cosine_similarities, axis=0) - np.nanmin(cosine_similarities, axis=0))

        # standardize shift by 0.5
        cosine_sims_norm = \
            0.5 + (cosine_sims_norm - np.nanmean(cosine_sims_norm, axis=0)) / np.nanstd(cosine_sims_norm, axis=0)

        return cosine_sims_norm

    @staticmethod
    def max_normalize_cosine_similarities(cosine_similarities):
        """Normalize cosine similarities using max normalization approach"""
        return 1 / np.max(cosine_similarities) * cosine_similarities.squeeze(axis=1)
