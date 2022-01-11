import argparse, os
from keep import SIGTREC_Eval
from collections import namedtuple
import pandas as pd
import sys

def getFileName(qrelFile):
    return os.path.basename(qrelFile)

class InputAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(InputAction, self).__init__(*args, **kwargs)
        self.nargs = '+'
    def __call__(self, parser, namespace, values, option_string):
        lst = getattr(namespace, self.dest, []) or []
        if len(values) > 1:
        	lst.append(InputResult(values[0], values[1:]))
        else:
        	with open(values[0].name) as file_input:
        		for line in file_input.readlines():
        			parts = line.strip().split(' ')
        			lst.append(InputResult(parts[0], parts[1:]))
        setattr(namespace, self.dest, lst)

class InputResult(object):
	def __init__(self, qrel, result_to_compare):
		if type(qrel) is str: 
			self.qrel = qrel
			self.result_to_compare = result_to_compare
		else:
			self.qrel = qrel.name
			self.result_to_compare = [ x.name for x in result_to_compare ]
	def __repr__(self):
		return 'InputResult(%r, %r)' % (self.qrel, self.result_to_compare)

def main():
	""" Configuring the argument parser """
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--input', action=InputAction, type=argparse.FileType('rt'), nargs='*', metavar='QREL BASELINE [TO_COMPARE ...]', help='The list of positional argument where the first arg is the qrel file, the second is the baseline result and the third is the optional list of results to compare.')
	parser.add_argument('-m','--measure', type=str, nargs='+', help='Evaluation method.', default=['P.10', 'recall.10'])
	parser.add_argument('-t','--trec_eval', type=str, nargs='?', help='The trec_eval executor path.', metavar='TREC_EVAL_PATH', default="trec_eval")
	parser.add_argument('-s','--statistical_test', type=str, nargs='*', help='Statistical test (Default: None).', default=['None'], choices=['student','wilcoxon','welcht'])
	parser.add_argument('-f','--format', type=str, nargs='?', help='Output format.', default='string', choices=['csv', 'html', 'json', 'latex', 'sql', 'string', 'df'])
	parser.add_argument('-r','--round', type=int, nargs='?', help='Round the result.', default=4)
	parser.add_argument('-M','--top',type=int, nargs='?', help='Max number of docs per topic to use in evaluation (discard rest).', default=sys.maxsize)

	args = parser.parse_args()

	statistical_test = [ st for st in args.statistical_test if st != "None"]

	sig = SIGTREC_Eval(round_=args.round, top=args.top, trec_eval=args.trec_eval)
	qrel = args.input[0].qrel
	result_to_compare = args.input[0].result_to_compare
	results = sig.Evaluate(qrel, getFileName(qrel), result_to_compare, args.measure, statistical_test, args.format)

	for res in results:
		print(res)

if __name__ == '__main__':
	# The entry point for program execution
	main()