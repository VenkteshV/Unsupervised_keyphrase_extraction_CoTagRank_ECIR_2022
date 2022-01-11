import sys, os, subprocess, math, multiprocessing, random
import numpy as np
from numpy import nan
import pandas as pd
from scipy.stats.mstats import ttest_rel
from scipy.stats import ttest_ind, wilcoxon
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import namedtuple
import warnings
#suppress warnings from pandas library
warnings.simplefilter(action='ignore', category=FutureWarning)

Result = namedtuple('Result', ['qrelFileName', 'datasetid', 'resultsFiles'])

def getFileName(qrelFile):
	return os.path.basename(qrelFile)

class SIGTREC_Eval():
	def __init__(self, round_=4, top=sys.maxsize, trec_eval="/home/ayan/concept-extraction-lo-backend/trec_eval"):
		self.nameApp = {}
		self.trec_eval = trec_eval
		self.round = round_
		self.top = top
		self.df = {}

	def Evaluate(self, path2qrel_file, datasetid, resultsFiles, measures, statistical_test, formatOutput):
		resultsFiles = self.ReadTrecEvalFiles(qrelFileName=path2qrel_file, datasetid=datasetid,
											 resultsFiles=resultsFiles)
		res = self.print(resultsFiles, measures, statistical_test, formatOutput)

		return res

	def _build_F1(self, qrelFileName, to_compare, m):
		command = ' '.join([self.trec_eval, qrelFileName, to_compare, '-q ', '-M %d' % self.top, '-m %s.%d'])
		content_P = str(subprocess.Popen(command % ('set_P', self.top), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])[2:-1].split('\\n')
		content_R = str(subprocess.Popen(command % ('recall', self.top), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])[2:-1].split('\\n')
		content_F1 = []
		for i in range(len(content_P)):
			part_P = content_P[i].split('\\t')
			part_R = content_R[i].split('\\t')
			if len(part_P) != len(part_R) or len(part_P) < 3:
				continue
			if part_P[1] != part_R[1]:
				print(part_P[1], part_R[1])
			else:
				Pre = float(part_P[2])
				Rec = float(part_R[2])
				if Pre == 0. or Rec == 0.:
					content_F1.append( 'F1_%d\\t%s\\t0.' % ( self.top, part_P[1] ) )
				else:
					line = 'F1_%d\\t%s\\t%.4f' % (self.top, part_P[1], (2.*Pre*Rec)/(Pre+Rec) )
					content_F1.append( line )
		return content_F1

	def build_df(self, resultsFile, measures):
		raw = []
		ListOfMeasures = self.getListOfMeasures(measures)

		qtd = len(measures)*sum([len(input_result.resultsFiles) for input_result in resultsFile])
		i=0
		for resultFile in resultsFile:
			self.nameApp[resultFile.datasetid] = []
			for m in ListOfMeasures:
				for (idx, to_compare) in enumerate(resultFile.resultsFiles):
					self.nameApp[resultFile.datasetid].append(getFileName(to_compare))
					#print("\r%.2f%%" % (100.*i/qtd),end='')
					i+=1
					mSplit = m.split('.')
					if len(mSplit) > 1:
						self.top = int(mSplit[1])
					if mSplit[0] == "F1":
						content = self._build_F1(resultFile.qrelFileName, to_compare, m)
					else:
					############################################################################################## Tamanho 10 FIXADO ##############################################################################################
						command = ' '.join([self.trec_eval, resultFile.qrelFileName, to_compare, '-q -M %d' % self.top, '-m', m])
						content = str(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])[2:-1].split('\\n')
					for docResult in content[:-1]:
						listOfRes = docResult.split('\\t')
						if (mSplit[0]=="map" or mSplit[0]=="Rprec" or mSplit[0]=="recip_rank" or mSplit[0]=="F1"):
							if len(mSplit) > 1:
								measure = mSplit[0] + "_" + str(self.top)
							else:
								measure = mSplit[0] + "_all"
						else:
							measure = listOfRes[0].strip()

						docid = listOfRes[1]
						result = listOfRes[2]
						raw.extend([(resultFile.datasetid, idx, getFileName(to_compare), measure, docid, result)])
		df_raw = pd.DataFrame(list(filter(lambda x: not x[4]=='all', raw)), columns=['qrel', 'idx_approach', 'approach', 'measure', 'docid', 'result'])
		df_finale = pd.pivot_table(df_raw, index=['qrel', 'docid'], columns=['idx_approach','measure'], values='result', aggfunc='first')
		df_finale.reset_index()
		df_finale[np.array(df_finale.columns)] = df_finale[np.array(df_finale.columns)].astype(np.float64)
		df_finale.replace('None', 0.0, inplace=True)
		df_finale.replace(nan, 0.0, inplace=True)
		#df_finale = df_finale[~df_finale['docid'].isin(['all'])]
		df_finale['fold'] = [0]*len(df_finale)
		return df_finale

	def get_test(self, test, pbase, pcomp, multi_test=False):
		if np.array_equal(pbase.values, pcomp.values):
			pvalue = 1.
		else:
			if test == 'student':
				(tvalue, pvalue) = ttest_rel(pbase, pcomp)
				diff_run1_run2 = [sum(x) for x in zip(pcomp, [-x for x in pbase])]
				effect_size = np.mean(diff_run1_run2)/np.std(diff_run1_run2)
				print("Effect size is:", effect_size)
			elif test == 'wilcoxon':
				(tvalue, pvalue) = wilcoxon(pbase, pcomp)
			elif test == 'welcht':
				(tvalue, pvalue) = ttest_ind(pbase, pcomp, equal_var=False)
		if pvalue < 0.05:
			pbase_mean = pbase.mean()
			pcomp_mean = pcomp.mean()
			if pvalue < 0.01:
				if pbase_mean > pcomp_mean:
					result_test = '▼ '
				else:
					result_test = '▲ '
			else:
				if pbase_mean > pcomp_mean:
					result_test = 'ᐁ '
				else:
					result_test = 'ᐃ '
		else:
			if not multi_test:
				result_test = '  '
			else:
				result_test = '⏺ '
		return result_test

	def build_printable(self, table, significance_tests):
		printable = {}
		for qrel, qrel_group in table.groupby('qrel'):
			raw = []
			base = qrel_group.loc[:,0]
			for idx_app in [idx for idx in qrel_group.columns.levels[0] if type(idx) == int]:
				instance = [ self.nameApp[qrel][idx_app] ]
				for m in qrel_group[idx_app].columns:
					array_results = qrel_group[idx_app][m]
					#print(qrel_group.groupby('fold').mean()[idx_app][m])
					mean_measure_folds = qrel_group.groupby('fold').mean()[idx_app][m].mean()
					test_result=""
					for test in significance_tests:
						if idx_app > 0:
							test_result+=(self.get_test(test, base[m], array_results, len(significance_tests)>1))
						else:
							test_result+=('bl ')
					instance.append('%f %s' % (round(mean_measure_folds,self.round), test_result) )
				raw.append(instance)
			printable[qrel] = pd.DataFrame(raw, columns=['app', *(table.columns.levels[1].values)[:-1]])
		return printable

	def getListOfMeasures(self, measures):
		ListOfMeasures = []
		for measure in measures:
			instances = measure.split('.', 1)
			measureName = instances[0]
			if len(instances) > 1:
				atN = instances[1].split(',')
				for n in atN:
					ListOfMeasures.append(measureName + "." + n.strip())
			else:
				ListOfMeasures.append(measureName)

		return(ListOfMeasures)

	def ReadTrecEvalFiles(self, qrelFileName, datasetid, resultsFiles):
		results = []
		results.append(Result(qrelFileName=qrelFileName, datasetid=datasetid, resultsFiles=resultsFiles))

		return results

	def print(self, resultsFiles, measures, statistical_test, formatOutput):
		self.df = self.build_df(resultsFiles, measures=measures)

		printable = self.build_printable(self.df, statistical_test)

		printResult = []
		for qrel in printable:
			with pd.option_context('display.max_rows', None, 'display.max_columns', 10000000000):
				if formatOutput == "df":
					printResult.append(printable[qrel])
				else:
					printResult.append(getattr(printable[qrel], 'to_' + formatOutput)())

		return printResult
