import wikipedia

class ConceptExpansion:
    """Exapnd concepts discovered in given text to represent more diverse cocnepts."""


    def extract_wikipedia_summary(self, concept):
        return wikipedia.summary(concept)

    def wikipedia_search(self, concept):
        return wikipedia.search(concept)  

    def expand_concepts(self, concepts):
        summaries = []
        similar_title_list =[]
        for concept in concepts:
            print("here************")
            try:
                similar_titles = self.wikipedia_search(concept)
                similar_title_list.extend(similar_titles)
                # summary = self.extract_wikipedia_summary(concept)   
                # summaries.append(summary)
                # normalization = None
                # CoTagRankUSE_object = CoTagRankUSE(numOfKeyphrases, pathData, dataset_name,
                #                                                         normalization)
                # _, phrase_lists = CoTagRankUSE_object.ExtractKeyphrases(text_1, highlight=True)
            except:
                continue
        print("similar_title_list", similar_title_list)


        return similar_title_list, summaries
