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
import subprocess
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tabstar.tabstar_model import TabSTARClassifier


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
    retcode = subprocess.call(['Rscript', 'send_data_extraction.R', '--training_zip_file', training_zip_file,
                                    '--testing_zip_file', testing_zip_file,
                                    '--training_data_labels', training_data_labels,
                                    '--testing_data_labels', testing_data_labels], shell=True)

    if retcode != 0:
        raise Exception("Data extraction failed.")

    training_df = pd.read_csv("training_data.csv")
    testing_df = pd.read_csv("testing_data.csv")

    return training_df, testing_df

#TODO: Implement
#def train_tabstar_model(training_df, testing_df):


def main():
    """Main script logic."""
    args = parse_args()

    training_df, testing_df = extract_data(args.training_zip_file,
                                                         args.testing_zip_file,
                                                         args.training_data_labels,
                                                         args.testing_data_labels)

    #TODO
    #train_tabstar_model(training_df, testing_df)

    try:
        print(args)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
