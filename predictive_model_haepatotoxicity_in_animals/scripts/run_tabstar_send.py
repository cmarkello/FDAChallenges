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
import zipfile
import shutil
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tabstar.tabstar_model import TabSTARClassifier
from pandas import DataFrame, Series


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

def get_directory_names(path):
    """
    Returns a list of directory names within the specified path.
    """
    directory_names = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            directory_names.append(entry)
    return directory_names

#def split_train_test_data(training_zip_file, training_data_labels):
#    cwd = os.getcwd()
#
#    training_data_basename = os.path.splitext(os.path.basename(training_zip_file))[0]
#    df_training_labels = pd.read_csv(f'{cwd}/{training_data_labels}')
#
#    with zipfile.ZipFile(f'{cwd}/{training_zip_file}', 'r') as zip_ref:
#        zip_ref.extractall(f'{cwd}/{training_data_basename}')
#
#    all_sample_ids = get_directory_names(f'{cwd}/{training_data_basename}')
#
#    split
#
#
#    train_sample_ids = get_directory_names(f'{cwd}/output/val')
#    train_sample_df = df_training_labels[df_training_labels['STUDYID'].isin(train_sample_ids)]
#
#    test_sample_ids = get_directory_names(f'{cwd}/output/test')
#    test_sample_df = df_training_labels[df_training_labels['STUDYID'].isin(test_sample_ids)]
#
#
#    shutil.make_archive(f'{cwd}/train_data', "zip", f'{cwd}/output/train')
#    shutil.make_archive(f'{cwd}/test_data', "zip", f'{cwd}/output/val')
#
#    shutil.rmtree(f'{cwd}/output/val')
#    shutil.rmtree(f'{cwd}/output/test')
#
#    train_sample_df.to_csv(f'{cwd}/train_label.csv', index=False)
#    test_sample_df.to_csv(f'{cwd}/test_label.csv', index=False)
#
#    return 'train_data.zip', 'test_data.zip', 'train_label.csv', 'test_label.csv'

def extract_data(training_zip_file, training_data_labels, testing_zip_file = None, testing_data_labels = None):
    cwd = os.getcwd()
    print(f"DEBUG CWD: {cwd}")
    data_extraction_cmd = ['Rscript', f'{cwd}/example_input_app_data/send_data_extraction.R',
                                    '--training_zip_file', f'{cwd}/{training_zip_file}',
                                    '--training_data_labels', f'{cwd}/{training_data_labels}',
                                    '--output_dir', f'{cwd}/test_out']
    if testing_zip_file is not None and testing_data_labels is not None:
        data_extraction_cmd += ['--testing_zip_file', f'{cwd}/{testing_zip_file}', '--testing_data_labels', f'{cwd}/{testing_data_labels}']

    retcode = subprocess.check_call(data_extraction_cmd, cwd=cwd)

    if retcode != 0:
        raise Exception(f"Data extraction failed: {retcode}")

    training_df = pd.read_csv(f'{cwd}/test_out/training_data.csv')
    output_df_list = [training_df]
    print(f"DEBUG x_train: {training_df}")
    if testing_zip_file is not None or testing_data_labels is not None:
        testing_df = pd.read_csv(f'{cwd}/test_out/testing_data.csv')
        output_df_list += [testing_df]

    return output_df_list

#TODO: Implement
def train_tabstar_model(train_test_df_list):

    x_train = train_test_df_list[0].iloc[:, 1:]
    y_train = x_train.pop('Target_Organ')
    is_cls = True

    x_test = None
    y_test = None
    if len(train_test_df_list) == 2:
        x_test = train_test_df_list[1].iloc[:, 1:]
        y_test = x_test.pop('Target_Organ')
        print(f'DEBUG x_test: {x_test}')
        print(f'DEBUG y_test: {y_test}')

    # Sanity checks
    assert isinstance(x_train, DataFrame), "x should be a pandas DataFrame"
    assert isinstance(y_train, Series), "y should be a pandas Series"
    assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

    print("DEBUG flag1")
    if x_test is None:
        assert y_test is None, "If x_test is None, y_test must also be None"
        print("DEBUG flag2")
        print(f'DEBUG splitting test data from training data')
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5)
        print(f"DEBUG x_train: {x_train}")
        print(f"DEBUG y_train: {y_train}")
        print(f"DEBUG x_test: {x_test}")
        print(f"DEBUG y_test: {y_test}")

    print("DEBUG flag3")

    assert isinstance(x_test, DataFrame), "x_test should be a pandas DataFrame"
    assert isinstance(y_test, Series), "y_test should be a pandas Series"

    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls()
    tabstar.val_ratio = 0.5
    tabstar.fit(x_train, y_train)
    tabstar.save("my_model_path.BAYER.xpt.pkl")

    #tabstar = TabSTARClassifier.load("my_model_path.FDA.pkl")
    #tabstar.fit(x_train, y_train)

    y_pred = tabstar.predict(x_test)
    print(f"Test prediction: {y_pred}")
    print(f"Test truth: {y_test}")
    #metric = tabstar.score(X=x_test, y=y_test)
    #print(f"AUC: {metric:.4f}")


def main():
    """Main script logic."""
    args = parse_args()

    train_test_df_list = extract_data(args.training_zip,
                                           args.training_labels,
                                           args.testing_zip,
                                           args.testing_labels)

    #cwd = os.getcwd()
    #train_test_df_list = [pd.read_csv(f'{cwd}/test_out/training_data.csv')]
    train_tabstar_model(train_test_df_list)
    #TODO: Align the testing_data target label type

    try:
        print(args)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
