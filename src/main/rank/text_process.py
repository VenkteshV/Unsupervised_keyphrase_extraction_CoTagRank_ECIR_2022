import os
from re import match

from nltk import ngrams, pos_tag, word_tokenize
from nltk.stem import SnowballStemmer

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
### preprocess ###
def read_file(path):
    with open(path, encoding='utf-8') as file:
        return file.read()

def get_tagged_tokens(file_text):
    file_splited = file_text.split()
    tagged_tokens = []
    for token in file_splited:
        tagged_tokens.append(tuple(token.split('_')))
    return tagged_tokens

def is_word(token):
    """
    A token is a "word" if it begins with a letter.
    
    This is for filtering out punctuations and numbers.
    """
    return match(r'^[A-Za-z].+', token)

def is_good_token(tagged_token):
    """
    A tagged token is good if it starts with a letter and the POS tag is
    one of ACCEPTED_TAGS in global.ini.
    """
    ACCEPTED_TAGS = set("NN NNS NNP NNPS JJ JJR JJS".split())
    return is_word(tagged_token[0]) and tagged_token[1] in ACCEPTED_TAGS
    
def normalized_token(token):
    """
    Use stemmer to normalize the token.
    """
    return wnl.lemmatize(token.lower())

def filter_text(text, with_tag=True):
    """

    """
    if with_tag:
        tagged_tokens = get_tagged_tokens(text)
    else:
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
    filtered_text = ''
    for tagged_token in tagged_tokens:
        if is_good_token(tagged_token):
            filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
    return filtered_text

### postprocess ###
def rm_tags(file_text):
    """
    remove tags in doc
    """
    file_splited = file_text.split()
    text_notag = ''
    for t in file_splited:
        text_notag = text_notag + ' ' + t[:t.find('_')]
    return text_notag

def get_phrases(pr, graph, text, ng=2, pl2=0.6, pl3=0.3, with_tag=True):
    """
    Return a list as `[('large numbers', 0.233)]`
    """
    tokens = word_tokenize(text.lower())
    edges = graph.edges
    phrases = set()

    for n in range(2, ng+1):
        for ngram in ngrams(tokens, n):

            # For each n-gram, if all tokens are words, and if the normalized
            # head and tail are found in the graph -- i.e. if both are nodes
            # connected by an edge -- this n-gram is a key phrase.
            if all(is_word(token) for token in ngram):
                head, tail = normalized_token(ngram[0]), normalized_token(ngram[-1])
                
                if head in edges and tail in edges[head] and pos_tag([ngram[-1]])[0][1] != 'JJ':
                    # phrase = ' '.join(list(normalized_token(word) for word in ngram))
                    phrase = ' '.join(ngram)
                    phrases.add(phrase)

    if ng == 2:
        phrase2to3 = set()
        for p1 in phrases:
            for p2 in phrases:
                if p1.split()[-1] == p2.split()[0] and p1 != p2:
                    phrase = ' '.join([p1.split()[0]] + p2.split())
                    phrase2to3.add(phrase)
        phrases |= phrase2to3

    phrase_score = {}
    for phrase in phrases:
        score = 0
        for word in phrase.split():
            score += pr.get(normalized_token(word), 0)
        plenth = len(phrase.split())
        if plenth == 1:
            phrase_score[phrase] = score
        elif plenth == 2:
            phrase_score[phrase] = score * pl2 # 
        else:
            phrase_score[phrase] = score * pl3 # 
        # phrase_score[phrase] = score/len(phrase.split())
    sorted_phrases = sorted(phrase_score.items(), key=lambda d: d[1], reverse=True)

    sorted_word = sorted(pr.items(), key=lambda d: d[1], reverse=True)
    out_sorted = sorted(sorted_phrases+sorted_word, key=lambda d: (d[1], d[0]), reverse=True)
    return out_sorted

def stem_doc(text):
    """
    Return stemmed text.
    :param text: text without tags
    """
    words_stem = [normalized_token(w) for w in text.split()]
    return ' '.join(words_stem)

def stem2word(text):
    stem_word = {}
    tokens = [t for t in word_tokenize(text) if is_good_token(t)]
    for t in tokens:
        s = normalized_token(t)
        if s not in stem_word:
            stem_word[s] = t
    return stem_word

def get_phrases_new(pr, graph, text, ng=2, pl2=0.6, pl3=0.3, with_tag=True):
    """
    Return a list as `[('large numbers', 0.233)]`
    """
    tokens = word_tokenize(text.lower())
    phrases = set()
    noun_tags = ('NN', 'NNS', 'NNP', 'NNPS')
    adj_tags = ('JJ', 'JJR', 'JJS')
    for n in range(2, ng+1):
        for ngram in ngrams(tokens, n):
            if pos_tag([ngram[0]])[0][1] in noun_tags+adj_tags and pos_tag([ngram[-1]])[0][1] in noun_tags:
                phrases.add(' '.join(ngram))
    if ng == 2:
        phrase2to3 = set()
        for p1 in phrases:
            for p2 in phrases:
                if p1.split()[-1] == p2.split()[0] and p1 != p2:
                    phrase = ' '.join([p1.split()[0]] + p2.split())
                    phrase2to3.add(phrase)
        phrases |= phrase2to3

    phrase_score = {}
    for phrase in phrases:
        score = 0
        for word in phrase.split():
            score += pr.get(normalized_token(word), 1e-10)
        plenth = len(phrase.split())
        if plenth == 1:
            phrase_score[phrase] = score
        elif plenth == 2:
            phrase_score[phrase] = score * pl2 
        else:
            phrase_score[phrase] = score * pl3 
        # phrase_score[phrase] = score/len(phrase.split())
    sorted_phrases = sorted(phrase_score.items(), key=lambda d: d[1], reverse=True)
    sorted_word = sorted(pr.items(), key=lambda d: d[1], reverse=True)
    out_sorted = sorted(sorted_phrases+sorted_word, key=lambda d: d[1], reverse=True)
    return out_sorted