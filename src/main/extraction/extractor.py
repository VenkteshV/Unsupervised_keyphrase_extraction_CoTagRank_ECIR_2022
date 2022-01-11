import nltk
from nltk.corpus import stopwords
import re
from main.extraction.input_representation import InputTextObj, PosTaggingCoreNLP
from nltk.parse import CoreNLPParser
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

class NPGrammars:
    GRAMMAR1 = """  NP:{<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
    GRAMMAR2 = """  NP:{<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""
    GRAMMAR3 = """  NP:{<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


class NPMethods:
    NOUN_CHUNKS = "noun_chunks"
    GRAMMAR = "grammar"
    REGEX = 'regex'


class NPTags:
    NLTK = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']


class StopWords:
    NLTK = set(stopwords.words('english'))


class PhraseHighlighter:
    """Highlights phrases in text"""
    color = '0,255,255'

    @staticmethod
    def to_html(text, phrases):
        marked_text = ''
        # last_end = 0
        # for phrase, st_idx, end_idx in phrases:
        #     marked_text += text[last_end:st_idx] + PhraseHighlighter._highlight(phrase, 1.0)
        #     last_end = end_idx

        # marked_text += text[last_end:]
        for phrase in phrases:
            text = re.sub(phrase[0].lstrip().lower(),PhraseHighlighter._highlight(phrase[0].lower(), 1.0), text.lower() )

        return text

    @staticmethod
    def _highlight(phrase: str, alpha: float) -> str:
        return f"<b style=\"background-color:rgba({PhraseHighlighter.color},{alpha})\">{phrase}</b>"


class Extractor:
    """Extracts some slices from text and highlights them"""

    def __init__(self):
        pass

    def run(self, text):
        pass


class CoreNLPExtractor(Extractor):

    def __init__(self, nlp):
        self.pos_tagger = PosTaggingCoreNLP("localhost", "9000")
        self.GRAMMAR_EN = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

        self.GRAMMAR_DE = """
        NBAR:
                {<JJ|CARD>*<NN.*>+}  # [Adjective(s) or Article(s) or Posessive pronoun](optional) + Noun(s)
                {<NN>+<PPOSAT><JJ|CARD>*<NN.*>+}

        NP:
        {<NBAR><APPR|APPRART><ART>*<NBAR>}# Above, connected with APPR and APPART (beim vom)
        {<NBAR>+}
        """

        self.GRAMMAR_FR = """  NP:
                {<NN.*|JJ>*<NN.*>+<JJ>*}  # Adjective(s)(optional) + Noun(s) + Adjective(s)(optional)"""

    def get_grammar(self, lang):
        if lang == 'en':
            grammar = self.GRAMMAR_EN
        elif lang == 'de':
            grammar = self.GRAMMAR_DE
        elif lang == 'fr':
            grammar = self.GRAMMAR_FR
        else:
            raise ValueError('Language not handled')
        return grammar


    def extract_candidates(self, text_obj, no_subset=False):
        """
        Based on part of speech return a list of candidate phrases
        :param text_obj: Input text Representation see @InputTextObj
        :param no_subset: if true won't put a candidate which is the subset of an other candidate
        :return: list of candidate phrases (string)
        """

        keyphrase_candidate = set()

        np_parser = nltk.RegexpParser(self.get_grammar(text_obj.lang))  # Noun phrase parser
        trees = np_parser.parse_sents(text_obj.pos_tagged)  # Generator with one tree per sentence

        for tree in trees:
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
                # Concatenate the token with a space
                keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))

        keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}

        if no_subset:
            keyphrase_candidate = unique_ngram_candidates(keyphrase_candidate)
        else:
            keyphrase_candidate = list(keyphrase_candidate)

        return keyphrase_candidate

    def run(self, text, lists = None):
        tagged_text = self.pos_tagger.pos_tag_raw_text(text)
        text_object = InputTextObj(tagged_text, "en")
        candidates = self.extract_candidates(text_object)
        return candidates
        


class PhraseExtractor(Extractor):
    """Extracts candidate phrases from given text using language models"""

    def __init__(self, nlp, grammar='GRAMMAR1', np_method='NOUN_CHUNKS', np_tags='NLTK', stopwords="NLTK"):
        """Takes nlp model (which supports POS tagging, SentTokenizer) and takes text to tokenize"""
        super().__init__()

        self.method = getattr(NPMethods, np_method)
        self.considered_tags = getattr(NPTags, np_tags)
        self.stopwords = getattr(StopWords, stopwords)
        self.grammar = getattr(NPGrammars, grammar)
        self.nlp = nlp

        self._init_np_parser()

    def clean_sentence(self, sentence):
        sentence = re.sub(r'([a-z])([A-Z])', r'\1\. \2', sentence)  # before lower case
        s = sentence.lower()
        # normalization 3: "&gt", "&lt"
        s = re.sub(r'&gt|&lt', ' ', s)
        # normalization 4: letter repetition (if more than 2)
        s = re.sub("\n", ' ',s)
        # normalization 4: letter repetition (if more than 2)
        s = re.sub(r'([a-z])\1{2,}', r'\1', s)
        # normalization 5: non-word repetition (if more than 1)
        s = re.sub(r'([\W+])\1{1,}', r'\1', s)
        s = re.sub(r'\[.*?\]', '. ', s)
        # normalization 9: [.?!] --> [.?!] xxx
        s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    #     # normalization 12: phrase repetition
    #     s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
        s = s.lower()
        return s.strip()
    def run(self, text, lists=None):
        text = self.clean_sentence(text)
        doc = self.nlp(text)

        if self.method == NPMethods.NOUN_CHUNKS:
            phrases = self._extract_candidates_spacy(doc)

        elif self.method == NPMethods.GRAMMAR:
            doc = self._override_stopword_tags(doc)
            tokens = self._extract_tokens(doc)
            phrases = self._extract_candidates_grammar(tokens, lists)

        else:
            phrases = []
        # phrases = [phrase for phrase in phrases if phrase[0].isalpha()]
        return phrases

    def _init_np_parser(self):
        if self.method == NPMethods.GRAMMAR:
            self.np_parser = nltk.RegexpParser(self.grammar)

    def _override_stopword_tags(self, doc):
        if self.stopwords:
            for token in doc:
                if token.text.lower() in self.stopwords:
                    token.tag_ = 'IN'

        return doc

    @staticmethod
    def _extract_tokens(doc):
        return [(token.text.lower(), token.tag_, token.idx, token.idx + len(token)) for token in doc]

    @staticmethod
    def _extract_candidates_spacy(doc):
        phrase_candidates = []

        for chunk in doc.noun_chunks:
            phrase_candidates.append((chunk.text.lower(), chunk.start_char, chunk.end_char))

        return phrase_candidates

    def phrase_in_lists(self, lists, phrase):
        start = 0
        end = len(lists) - 1
        while start < end:
            mid = (start + end) // 2
            if phrase > lists[mid]:
                start = mid + 1
            else:
                end = mid
        if phrase == lists[start]:
            return True
        else:
            return False

    def _extract_candidates_grammar(self, tokens, lists):
        phrase_candidates = []
        np_tree = self.np_parser.parse(tokens)

        for node in np_tree:
            if isinstance(node, nltk.tree.Tree) and node._label  == 'NP':
                tokens = []
                indices = set()
                for node_child in node.leaves():
                    tokens.append(node_child[0])
                    indices.add(node_child[2])
                    indices.add(node_child[3])

                phrase = ' '.join(tokens)

                phrase_start_idx = min(indices)
                phrase_end_idx = max(indices)
                #comment lines 166 to 171 to evaluate results without wikipedia based filtering  step if fine grained cocnepts 
                #are not needed. For example for keyphrase extraction we dont need this as the goal of this piece of code
                #is to consider phrases that fall into wikipedia articles as it conforms to ur definition of a technical concept
                # In keyphrase extarction this may eliminate some phrases that might lead to decrease in accuracy.
                if lists:
                    words = phrase.split(' ')
                    phrase = ''
                    for word in words:
                        if self.phrase_in_lists(lists, word):
                            phrase = phrase +' ' +word
                phrase_candidates.append((phrase, phrase_start_idx, phrase_end_idx))

        sorted_phrase_candidates = self._sort_candidates(phrase_candidates)
        final_list_candidates = set()
        selected_candidates=[]
        # for phrase,position_start,position_end in sorted_phrase_candidates:
        #     if not phrase in final_list_candidates:
        #         lemmatized_phrase = ' '.join(wnl.lemmatize(word) for word in phrase.split(" "))
        #         final_list_candidates.add(lemmatized_phrase)
        #         selected_candidates.append((lemmatized_phrase,position_start,position_end))



        return sorted_phrase_candidates

    @staticmethod
    def _sort_candidates(phrases):
        return sorted(phrases, key=lambda x: x[2])
