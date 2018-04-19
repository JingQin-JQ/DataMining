import argparse
import sys
import pandas as pd

from benchmark import Benchmark
from ml import ML
from temporal_algorithm import Temporal
import util
import warnings


def get_args():
    """Collect the inputs."""
    parser = argparse.ArgumentParser(
        prog='PROG',
        usage='%(prog)s [options]',
        description='Prideic mood based on smartphone applications',
        epilog='The code was...'
    )
    parser.add_argument('-m', '--prediction_models', dest='prediction_models',required=True,
                        help='Prediction_models: ml, Temporal_algorithm or benchmark',
                        default="benchmark")
    parser.add_argument('-o', '--pred_file', dest='pred_file', required=True,
                        metavar='FILE', default='pred_file',
                        help='The file to store the prediction')
    parser.add_argument('-e', '--eval_file', dest='eval_file',
                        required=True, metavar='FILE', default='eval_file',
                        help='The file to store the score of evaluation')
    
    args = parser.parse_args()

    if args.prediction_models not in ['ml', 'Temporal_algorithm','benchmark']:
        sys.exit('Unknown models ' + args.prediction_models)
    return args


def main():
    """
    Main function.

    """
    warnings.filterwarnings("ignore")
    # get data from csv file, and catch error reading file
    try:
        df = pd.read_csv('data/dataset_mood_smartphone.csv',sep=",")
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
    util.output_to_file(predictions, args.pred_file)
    # util.output_to_file(evaluation_scores, args.eval_file)
    util.output_to_screen(evaluation_scores)


if __name__ == "__main__":
    main()
