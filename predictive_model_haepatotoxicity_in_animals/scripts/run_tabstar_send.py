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

    # Input training file format
    parser.add_argument(
        "-f", "--training_data_format",
        type=str,
        default="csv",
        help="Input training data file format. Either 'xpt' or 'csv'. Default 'xpt'."
    )

    # Input tabSTAR model file
    parser.add_argument(
        "-g", "--tabstar_input_model_file",
        help="Input tabstar model file. OPTIONAL"
    )

    # Output tabSTAR model file
    parser.add_argument(
        "-o", "--tabstar_output_model_file",
        default="my_model_path.pkl",
        help="Input tabstar model file to save training to. OPTIONAL. Default 'my_model_path.pkl'."
    )

    # Input tabSTAR validation ratio size relative to total input training sample size
    parser.add_argument(
        "-r", "--val_ratio",
        type=float,
        default=0.1,
        help="Input tabstar validation ratio for train validation sample split. OPTIONAL. Default 0.1."
    )

    # Input tabSTAR validation ratio size relative to total input training sample size
    parser.add_argument(
        "-s", "--test_size",
        type=float,
        default=0.25,
        help="Input tabstar test ratio for train test sample split. OPTIONAL. Default 0.25."
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

def extract_data(training_zip_file, training_data_labels, testing_zip_file = None, testing_data_labels = None, training_data_format = 'csv'):
    cwd = os.getcwd()
    print(f"DEBUG CWD: {cwd}")
    data_extraction_cmd = ['Rscript', f'{cwd}/example_input_app_data/send_data_extraction.R',
                            '--training_zip_file', f'{cwd}/{training_zip_file}',
                            '--training_data_labels', f'{cwd}/{training_data_labels}',
                            '--training_data_format', training_data_format,
                            '--output_dir', f'{cwd}/test_out']
    if testing_zip_file is not None and testing_data_labels is not None:
        data_extraction_cmd += ['--testing_zip_file', f'{cwd}/{testing_zip_file}', '--testing_data_labels', f'{cwd}/{testing_data_labels}']

    print(f'DEBUG data_extraction_cmd: {data_extraction_cmd}')
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
def train_tabstar_model(train_test_df_list, tabstar_input_model_file = None, tabstar_output_model_file = 'my_model_path', val_ratio: float = 0.1, test_size: float = 0.25):
    cwd = os.getcwd()
    x_train = train_test_df_list[0].iloc[:, 1:]
    y_train = x_train.pop('Target_Organ')
    is_cls = True

    x_test = None
    y_test = None
    if len(train_test_df_list) == 2:
        x_test = train_test_df_list[1].iloc[:, 1:]
        y_test = x_test.pop('Target_Organ')

    # Sanity checks
    assert isinstance(x_train, DataFrame), "x should be a pandas DataFrame"
    assert isinstance(y_train, Series), "y should be a pandas Series"
    assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

    if x_test is None:
        assert y_test is None, "If x_test is None, y_test must also be None"
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)

    assert isinstance(x_test, DataFrame), "x_test should be a pandas DataFrame"
    assert isinstance(y_test, Series), "y_test should be a pandas Series"

    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls()
    tabstar.lora_lr = 0.001
    tabstar.lora_r = 32
    tabstar.lora_batch = 16
    tabstar.global_batch = 128
    tabstar.max_epochs = 200
    tabstar.patience = 200
    if tabstar_input_model_file is not None:
        tabstar_input_model_file_basename = tabstar_input_model_file.rsplit('.', 1)[0]
        os.makedirs(f'{cwd}/{tabstar_input_model_file_basename}/', exist_ok=True)
        with zipfile.ZipFile(f'{tabstar_input_model_file}', 'r') as zf:
            zf.extractall(f'{cwd}/{tabstar_input_model_file_basename}/')
        tabstar.load_dir = f'{cwd}/{tabstar_input_model_file_basename}/'

    os.makedirs(f'{cwd}/{tabstar_output_model_file}/', exist_ok=True)
    tabstar.save_dir = f'{cwd}/{tabstar_output_model_file}/'
    tabstar.val_ratio = val_ratio
    tabstar.fit(x_train, y_train)
    tabstar.save(f'{cwd}/{tabstar_output_model_file}/{tabstar_output_model_file}.pkl')

    y_pred = tabstar.predict(x_test)
    print(f"Test prediction: {y_pred}")
    print(f"Test truth: {y_test}")
    #metric = tabstar.score(X=x_test, y=y_test)
    #print(f"AUC: {metric:.4f}")
    print(f'Outputting model to {cwd}/{tabstar_output_model_file}.zip')
    shutil.make_archive(f'{cwd}/{tabstar_output_model_file}', "zip", f'{cwd}/{tabstar_output_model_file}')



def main():
    """Main script logic."""
    args = parse_args()

    training_zip = args.training_zip
    testing_zip = args.testing_zip
    training_labels = args.training_labels
    testing_labels = args.testing_labels
    training_data_format = args.training_data_format
    tabstar_input_model_file = args.tabstar_input_model_file
    tabstar_output_model_file = args.tabstar_output_model_file
    val_ratio = args.val_ratio
    test_size = args.test_size

    # Check if we are just running evaluation or training and evaluating
    if (training_zip is None and testing_zip is not None):
        training_zip = testing_zip
        testing_zip = None

    train_test_df_list = extract_data(training_zip,
                                     training_labels,
                                     testing_zip,
                                     testing_labels,
                                     training_data_format)

    train_tabstar_model(train_test_df_list, tabstar_input_model_file, tabstar_output_model_file, val_ratio, test_size)

    try:
        print(args)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
