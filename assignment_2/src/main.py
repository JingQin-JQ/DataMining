import argparse
import sys
import pandas as pd

from benchmark import Benchmark
import util
import warnings


def get_args():
    """Collect the inputs."""
    parser = argparse.ArgumentParser(
       description='Predict user behavior for Expedia')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='prediction_models',
                        metavar='model.py', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.prediction_models is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')
    
    return args

def main():
    """
    Main function.

    """
    warnings.filterwarnings("ignore")
    # get data from csv file, and catch error reading file
    try:
        df = pd.read_csv('train_sub.csv', sep=',')
    except OSError as e:
        print("ERROR: cannot open or read input file")

    # get command line options
    args = get_args()

    df = util.init_data(df)

    # initial model
    if args.prediction_models == "ml":
        model = ML()
    elif args.prediction_models == "Temporal_algorithm":
        model = Temporal()
    elif args.prediction_models == "benchmark":
        model = Benchmark()             
    else:
        sys.exit("BUG! this should not happen.")

    # call pipline
    predictions, evaluation_scores = model.pipeline(df)

    # print output
    util.output_to_file(predictions, args.output_file)
    # util.output_to_file(evaluation_scores, args.eval_file)
    util.output_to_screen(evaluation_scores)


if __name__ == "__main__":
    main()
