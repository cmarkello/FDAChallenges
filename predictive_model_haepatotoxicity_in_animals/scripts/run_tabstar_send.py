#!/usr/bin/env python3

"""
Script Name: run_tabstar_send.py
Description: DOES STUFF
Author: AUTHOR
Date: TODAYS_DATE
"""

import argparse
import sys
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data

r_base = importr('base')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TODO: ADD DESCRIPTION."
    )

    # Input training zip file
    parser.add_argument(
        "-t", "--training_zip",
        help="Input training zip file"
    )

    # Input testing zip file
    parser.add_argument(
        "-u", "--testing_zip",
        help="Input testing zip file"
    )

    # Input training label file
    parser.add_argument(
        "-m", "--training_labels",
        help="Input training label file"
    )

    # Input testing label file
    parser.add_argument(
        "-n", "--testing_labels",
        help="Input testing label file"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )

    return parser.parse_args()

def extract_data(training_zip_file, testing_zip_file, training_data_labels, testing_data_labels):

    return training_dataframe, testing_dataframe

def main():
    """Main script logic."""
    args = parse_args()

    training_dataframe, testing_dataframe = extract_data(args.training_zip_file,
                                                         args.testing_zip_file,
                                                         args.training_data_labels,
                                                         args.testing_data_labels)

    try:
        print(args)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
