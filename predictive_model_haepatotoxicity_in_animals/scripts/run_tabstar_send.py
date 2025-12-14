#!/usr/bin/env python3

"""
Script Name: run_tabstar_send.py
Description: The applicaation for tabstar-based SEND data hepatotoxicity predictive modeling
Author: Charles Markello
Date: 11-13-2025
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
        description="Arugment parser for the tabstar-based SEND data hepatotoxicity predictive model app."
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
        default="xpt",
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
        default="tabstar_model.zip",
        help="Input tabstar model file to save training to. OPTIONAL. Default 'trained_model.zip'."
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

def extract_data_for_training(training_zip_file, training_data_labels, testing_zip_file = None, testing_data_labels = None, training_data_format = 'xpt'):
    cwd = os.getcwd()
    data_extraction_cmd = ['Rscript', f'/usr/bin/send_data_train_extraction.R',
                            '--training_zip_file', f'{cwd}/{training_zip_file}',
                            '--training_data_labels', f'{cwd}/{training_data_labels}',
                            '--training_data_format', training_data_format,
                            '--output_dir', f'{cwd}/test_out']
    if testing_zip_file is not None:
        data_extraction_cmd += ['--testing_zip_file', f'{cwd}/{testing_zip_file}']

    if testing_data_labels is not None:
        data_extraction_cmd += ['--testing_data_labels', f'{cwd}/{testing_data_labels}']

    retcode = subprocess.check_call(data_extraction_cmd, cwd=cwd)

    if retcode != 0:
        raise Exception(f"Data extraction failed: {retcode}")

    training_df = pd.read_csv(f'{cwd}/test_out/training_data.csv')
    output_df_list = [training_df]
    if testing_zip_file is not None or testing_data_labels is not None:
        testing_df = pd.read_csv(f'{cwd}/test_out/testing_data.csv')
        output_df_list += [testing_df]

    return output_df_list

def extract_data_for_testing(testing_zip_file, training_data_format = 'xpt'):
    cwd = os.getcwd()
    data_extraction_cmd = ['Rscript', f'/usr/bin/send_data_test_extraction.R',
                            '--testing_zip_file', f'{cwd}/{testing_zip_file}',
                            '--training_data_format', training_data_format,
                            '--output_dir', f'{cwd}/test_out']

    retcode = subprocess.check_call(data_extraction_cmd, cwd=cwd)

    if retcode != 0:
        raise Exception(f"Data extraction failed: {retcode}")

    testing_df = pd.read_csv(f'{cwd}/test_out/testing_data.csv')

    return testing_df

def train_tabstar_model(train_test_df_list, tabstar_input_model_file = None, tabstar_output_model_file = 'trained_model.zip', val_ratio: float = 0.1, test_size: float = 0.25):
    cwd = os.getcwd()
    x_train = train_test_df_list[0].iloc[:, 1:]
    y_train = x_train.pop('Target_Organ')
    is_cls = True

    x_test = None
    if len(train_test_df_list) == 2:
        x_test = train_test_df_list[1].iloc[:, 1:]

    # Sanity checks
    assert isinstance(x_train, DataFrame), "x should be a pandas DataFrame"
    assert isinstance(y_train, Series), "y should be a pandas Series"
    assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

    if x_test is None:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)

    assert isinstance(x_test, DataFrame), "x_test should be a pandas DataFrame"

    tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
    tabstar = tabstar_cls()
    if tabstar_input_model_file is not None:
        tabstar_input_model_file_basename = tabstar_input_model_file.rsplit('.', 1)[0]
        os.makedirs(f'{cwd}/{tabstar_input_model_file_basename}/', exist_ok=True)
        with zipfile.ZipFile(f'{tabstar_input_model_file}', 'r') as zf:
            zf.extractall(f'{cwd}/{tabstar_input_model_file_basename}/')
        tabstar.load_dir = f'{cwd}/{tabstar_input_model_file_basename}/'
    else:
        # Load the default packaged model
        os.makedirs(f'{cwd}/tabstar_model/', exist_ok=True)
        with zipfile.ZipFile(f'{cwd}/tabstar_model.zip', 'r') as zf:
            zf.extractall(f'{cwd}/tabstar_model/')
        tabstar = TabSTARClassifier.load(f'{cwd}/tabstar_model/tabstar_model.pkl')

    tabstar.lora_lr = 0.001
    tabstar.lora_r = 32
    tabstar.lora_batch = 16
    tabstar.global_batch = 128
    tabstar.max_epochs = 200
    tabstar.patience = 200
    tabstar.val_ratio = val_ratio

    tabstar_output_model_file_basename = tabstar_output_model_file.rsplit('.', 1)[0]
    os.makedirs(f'{cwd}/{tabstar_output_model_file_basename}/', exist_ok=True)
    tabstar.save_dir = f'{cwd}/{tabstar_output_model_file_basename}/'
    tabstar.fit(x_train, y_train)
    tabstar.save(f'{cwd}/{tabstar_output_model_file_basename}/{tabstar_output_model_file_basename}.pkl')

    y_pred = tabstar.predict(x_test)
    series_from_y_pred = pd.Series(y_pred, name='PREDICTION')
    combined_df = pd.concat([train_test_df_list[1]['STUDYID'], series_from_y_pred], axis=1)
    with open(f'{cwd}/test_out/test_prediction.csv', 'w') as csvfile:
        csvfile.write(f'STUDYID,Predicted hepatotoxicity score\n')
        for index, row in combined_df.iterrows():
            prediction_str = f'{row['STUDYID']}, {row['PREDICTION']}'
            csvfile.write(f'{prediction_str}\n')
            print(f'Test prediction: {prediction_str}')

    print(f'Outputting trained model to {cwd}/{tabstar_output_model_file}')
    shutil.make_archive(f'{cwd}/{tabstar_output_model_file_basename}', "zip", f'{cwd}/{tabstar_output_model_file_basename}')
    shutil.rmtree(f'{cwd}/{tabstar_output_model_file_basename}/')

def run_tabstar_model(test_df, tabstar_input_model_file = None, val_ratio: float = 0.1):
    cwd = os.getcwd()
    x_test = test_df.iloc[:, 1:]
    is_cls = True

    # Sanity checks
    assert isinstance(x_test, DataFrame), "x should be a pandas DataFrame"
    assert isinstance(is_cls, bool), "is_cls should be a boolean indicating classification or regression"

    if tabstar_input_model_file is not None:
        tabstar_input_model_file_basename = tabstar_input_model_file.rsplit('.', 1)[0]
        os.makedirs(f'{cwd}/{tabstar_input_model_file_basename}/', exist_ok=True)
        with zipfile.ZipFile(f'{tabstar_input_model_file}', 'r') as zf:
            zf.extractall(f'{cwd}/{tabstar_input_model_file_basename}/')
        tabstar = TabSTARClassifier.load(f'{cwd}/{tabstar_input_model_file_basename}/{tabstar_input_model_file_basename}.pkl')
    else:
        # Load the default packaged model
        os.makedirs(f'{cwd}/tabstar_model/', exist_ok=True)
        with zipfile.ZipFile(f'{cwd}/tabstar_model.zip', 'r') as zf:
            zf.extractall(f'{cwd}/tabstar_model/')
        tabstar = TabSTARClassifier.load(f'{cwd}/tabstar_model/tabstar_model.pkl')

    tabstar.val_ratio = val_ratio

    y_pred = tabstar.predict(x_test)
    series_from_y_pred = pd.Series(y_pred, name='PREDICTION')
    combined_df = pd.concat([test_df['STUDYID'], series_from_y_pred], axis=1)
    with open(f'{cwd}/test_out/test_prediction.csv', 'w') as csvfile:
        csvfile.write(f'STUDYID,Predicted hepatotoxicity score\n')
        for index, row in combined_df.iterrows():
            prediction_str = f'{row['STUDYID']}, {row['PREDICTION']}'
            csvfile.write(f'{prediction_str}\n')
            print(f'Test prediction: {prediction_str}')


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
    if training_zip is None and testing_zip is not None:
        test_df = extract_data_for_testing(testing_zip,
                                            training_data_format)
        run_tabstar_model(test_df, tabstar_input_model_file, val_ratio)
    else:
        train_test_df_list = extract_data_for_training(training_zip,
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
