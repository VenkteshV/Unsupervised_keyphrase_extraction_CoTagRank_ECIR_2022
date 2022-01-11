from os import path
from glob import glob
from string import punctuation
import os
from segtok.tokenizer import web_tokenizer, split_contractions

class Convert(object):
    def __init__(self, pathToDatasetName, EvaluationStemming):
        self.pathToDatasetName = pathToDatasetName
        self.datasetid = self.__get_datasetid__()
        self.lang = self.__get_language__()
        self.EvaluationStemming = self.__get_EvaluationStemming__(EvaluationStemming)
        self.qrels = self.build_qrels()

    #Get keywords ID for each document according to their weight importance. Eg., {'doc1':['uk12', 'uk12']}
    def getKeywordsID(self, keywordsPath):
        listOfKeywordsFile = []
        for file in glob(keywordsPath + '/*'):
            listOfKeywordsFile.append(file.replace(os.sep, '/'))

        toreturn = []
        for resultdoc in sorted(listOfKeywordsFile):
            docid = self.__get_docid__(resultdoc)
            if docid not in self.qrels:
                print('[WARNING] Documento %s not fount in qrels' % docid)
                continue
            gt = self.qrels[docid]
            seen = set()
            result=[]
            keyphrases = self.__readfile__(resultdoc).split('\n')
            if len(keyphrases) == 0:
                idkw = 'uk00'
                gt['--'] = (idkw, False)
            else:
                for weight, kw in self.__sorted_numericList__(keyphrases):
                    kw_key = self.__get_filtered_key__(kw)
                    if kw_key not in gt:
                        idkw = ('uk%d' % len(gt))
                        isrel = False
                        gt[kw] = (idkw, False)
                    else:
                        idkw, isrel = gt[kw_key]
                    if idkw not in seen:
                        seen.add(idkw)
                        result.append( idkw )
            self.qrels[docid] = gt
            toreturn.append( (docid, result) )
        return toreturn

    def CreateOutFile(self, output_path, keywordsPath, dataset_name, algorithm):
        results = self.getKeywordsID(keywordsPath)
        output_file =output_path + f"{dataset_name}_{algorithm}.out"
        print(f"1 - CreateOutFile: {output_file}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(output_file, 'w') as outfile:
            for (docid, result) in results:
                for i, instance in enumerate(result):
                    outfile.write("%s Q0 %s %d %d %s\n" % ( docid, instance, (i+1), (len(result)-i), algorithm ) )

    def CreateQrelFile(self, output_path, dataset_name):
        output_file = output_path + f"{dataset_name}.qrel"
        print(f"2 - CreateQrelFile: {output_file}")
        with open(output_file, 'w') as outfile:
            for docid in self.qrels:
                for (idkw, isrel) in [(idkw, isrel) for (idkw, isrel) in self.qrels[docid].values() if isrel]:
                    outfile.write("%s\t0\t%s\t1\n" % ( docid, idkw ) )

    # Create qrels for dataset - gets (for each doc of the dataset) the list of keywords and its respective id
    def build_qrels(self):
        keysfiles = glob(self.pathToDatasetName + '/keys/*')
        qrels = {}
        j = 0

        for keyfile in keysfiles:
            docid = self.__get_docid__(keyfile)
            gt = {}
            keysunfiltered = self.__readfile__(keyfile).split('\n')
            for goldkey in keysunfiltered:
                gold_key = self.__get_filtered_key__(goldkey)
                if gold_key not in gt:
                    gt[gold_key] = ('k%d' % len(gt), True)
            qrels[docid] = gt

            j += 1

        return qrels
    # UTILS
    def __get_EvaluationStemming__(self, EvaluationStemming):
        filters = []
        if EvaluationStemming:
            if self.lang == 'polish':
                from keep import PolishStemmer
                self.stem = PolishStemmer()
                filters.append(self.__polish_stem__)
            elif self.lang == 'english':
                from nltk.stem import PorterStemmer
                self.stem = PorterStemmer()
                filters.append(self.__nltk_stem__)
            elif self.lang == 'portuguese':
                from nltk.stem import RSLPStemmer
                self.stem = RSLPStemmer()
                filters.append(self.__nltk_stem__)
            else:
                from nltk.stem.snowball import SnowballStemmer
                self.stem = SnowballStemmer(self.lang)
                filters.append(self.__nltk_stem__)
        return filters

    def __get_filtered_key__(self, key):
        key_filtered = self.__simple_filter__(key)
        for termfilter in self.EvaluationStemming:
            key_filtered = termfilter(key_filtered)
        return key_filtered

    def __get_datasetid__(self):
        return path.split(path.realpath(self.pathToDatasetName))[1]

    def __get_docid__(self, dockeypath):
        return path.basename(dockeypath).replace('.txt','').replace('.key','').replace('.out','').replace('.phrases','') 

    def __readfile__(self, filepath):
        with open(filepath, encoding='utf8') as infile:
            content = infile.read()
        return content

    def __get_language__(self):
        return self.__readfile__(self.pathToDatasetName + '/language.txt').replace('\n', '')

    def __get_appname__(self, resultdir):
        return '_'.join([ config for config in path.dirname(resultdir).split(path.sep)[-2:] if config != 'None'])

    # FILTERS
    def __simple_filter__(self, word):
	    term = word.lower()
	    for p in punctuation:
	        term = term.replace(p, ' ')
	    term = ' '.join([ w for w in split_contractions(web_tokenizer(term)) ])
	    return term.strip()
    def __none_filter__(self, word):
	    return word
    def __polish_stem__(self, word):
        return ' '.join(self.stem.stemmer_convert([ w for w in split_contractions(web_tokenizer(word)) ]))
    def __nltk_stem__(self, word):
        return ' '.join([ self.stem.stem(w) for w in split_contractions(web_tokenizer(word)) ])

    # CONVERSORS


    def __sorted_numericList__(self, listofkeys):
        toreturn = []
        for key in listofkeys:
            parts = key.rsplit(' ', 1)
            if len(key) > 0 and len(parts) > 1:
                kw, weight = parts
                try:
                    weight = float(weight)
                except:
                    weight = 0.
                toreturn.append( (weight, kw) )
        return toreturn