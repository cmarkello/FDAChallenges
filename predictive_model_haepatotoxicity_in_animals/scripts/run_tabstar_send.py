#!/usr/bin/env python3

"""
Script Name: run_tabstar_send.py
Description: DOES STUFF
Author: AUTHOR
Date: TODAYS_DATE
"""

import argparse
import sys
import os
import pandas as pd
import subprocess
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tabstar.tabstar_model import TabSTARClassifier
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


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
    cwd = os.getcwd()
    print(f"DEBUG CWD: {cwd}")
    retcode = subprocess.check_call(['Rscript', 'send_data_extraction.R',
                                    '--training_zip_file', f'{cwd}/{training_zip_file}',
                                    '--testing_zip_file', f'{cwd}/{testing_zip_file}',
                                    '--training_data_labels', f'{cwd}/{training_data_labels}',
                                    '--testing_data_labels', f'{cwd}/{testing_data_labels}',
                                    '--output_dir', f'{cwd}/test_out'], cwd=cwd)

    if retcode != 0:
        raise Exception(f"Data extraction failed: {retcode}")

    training_df = pd.read_csv(f'{cwd}/test_out/training_data.csv')
    testing_df = pd.read_csv(f'{cwd}/test_out/testing_data.csv')

    return training_df, testing_df

#TODO: Implement
def train_tabstar_model(training_df, testing_df):

    x_train = training_df.iloc[:, 1:]
    y_train = x_train.pop('Target_Organ')
    is_cls = True
    x_test = testing_df.iloc[:, 1:]
    y_test = testing_df.pop('Target_Organ')

    # Sanity checks
    assert isinstance(x_train, DataFrame), "x should be a pandas DataFrame"
    assert isinstance(y_train, Series), "y should be a pandas Series"
    assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

    if x_test is None:
        assert y_test is None, "If x_test is None, y_test must also be None"
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

    assert isinstance(x_test, DataFrame), "x_test should be a pandas DataFrame"
    assert isinstance(y_test, Series), "y_test should be a pandas Series"

    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls()
    tabstar.fit(x_train, y_train)
    tabstar.save("my_model_path.pkl")
    tabstar = TabSTARClassifier.load("my_model_path.pkl")
    tabstar.fit(x_train, y_train)
    y_pred = tabstar.predict(x_test)
    print(f"Test prediction: {y_pred}")
    print(f"Test truth: {y_test}")
    #metric = tabstar.score(X=x_test, y=y_test)
    #print(f"AUC: {metric:.4f}")


def main():
    """Main script logic."""
    args = parse_args()

    #training_df, testing_df = extract_data(args.training_zip,
    #                                                     args.testing_zip,
    #                                                     args.training_labels,
    #                                                     args.testing_labels)

    cwd = os.getcwd()
    training_df = pd.read_csv(f'{cwd}/test_out/training_data.csv')
    testing_df = pd.read_csv(f'{cwd}/test_out/testing_data.csv')
    train_tabstar_model(training_df, testing_df)
    #TODO: Align the testing_data target label type

    try:
        print(args)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
