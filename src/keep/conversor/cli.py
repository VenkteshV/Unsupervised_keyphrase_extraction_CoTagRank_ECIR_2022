import argparse
from os import path 
from glob import glob
from string import punctuation
import os

from keep import Convert

def main():
    parser = argparse.ArgumentParser()

    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-pathDataset', type=str, nargs=1, help='', required=True)
    required_args.add_argument('-pathKeyphrases', type=str, nargs=1, help='', required=True)
    parser.add_argument('-dataset_name', type=str, nargs=1, help='Dataset name.', required=True)
    parser.add_argument('-pathOutput', type=str, nargs=1, help='Output path.', required=True)
    parser.add_argument('-algorithmName', type=str, nargs=1, help='algorithm.', required=True)
    parser.add_argument('-EvaluationStemming', type=bool, nargs='?', help='Filter method.')

    args = parser.parse_args()

    pathDataset = args.pathDataset[0]
    pathKeyphrases = args.pathKeyphrases[0]
    dataset_name = args.dataset_name[0]
    pathOutput = args.pathOutput[0]
    algorithmName = args.algorithmName[0]
    EvaluationStemming = args.EvaluationStemming

    conv = Convert(pathDataset, EvaluationStemming=EvaluationStemming)
    conv.CreateOutFile(pathOutput, pathKeyphrases, dataset_name, algorithmName)
    conv.CreateQrelFile(pathOutput, dataset_name)

    print()
    print()

if __name__ == '__main__':
	# The entry point for program execution
	main()