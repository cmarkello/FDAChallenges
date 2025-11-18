
## Introduction
https://precision.fda.gov/challenges/35/intro
This precisionFDA challenge invites participants to develop predictive toxicology models from animal study data produced in compliance with the Standard for Exchange of Nonclinical Data (SEND).

The purpose of this Challenge is to
- Evaluate how well computational approaches using SEND data can reproduce the hepatotoxicity risk determined by toxicologists
- Create public tools for deriving hepatotoxicity risks from SEND data
- Improve the usability of existing SEND data to advance the roadmap to reducing animal testing in preclinical safety studies

## Acknowledgements
This application was developed by Charles Markello utilizing a number of tools as described below in the Implementation 
section of this README.

## Contact Info
Email: cjmarkello@gmail.com

github: https://github.com/cmarkello

## Running the App
All input data files should be in XPT SEND format. But you can switch to CSV file format support with the
flag `--training_data_format 'csv'`.

### Training and Testing the Model
docker run  --rm -v /path/to/data:/work tab_star_send \
/usr/bin/run_tabstar_send.py \
--training_zip /work/training_data.zip \
--testing_zip /work/testing_data.zip \
--training_labels /work/labels_training_data.csv \
--testing_labels /work/labels_testing_data.csv

docker run  --rm -v /path/to/data:/work tab_star_send \
/usr/bin/run_tabstar_send.py \
--training_zip /work/training_data.zip \
--testing_zip /work/testing_data.zip \
--training_labels /work/labels_training_data.csv

### Training and Testing the Model
docker run  --rm -v /path/to/data:/work tab_star_send \
/usr/bin/run_tabstar_send.py \
--training_zip /work/training_data.zip \
--testing_zip /work/testing_data.zip \
--training_labels /work/labels_training_data.csv \
--testing_labels /work/labels_testing_data.csv


### Testing the Model
docker run  --rm -v $PWD:/work tab_star_send \
/usr/bin/run_tabstar_send.py \
--testing_zip /work/testing_data.zip \
--training_labels /work/labels_training_data.csv 


## Output

The app writes 3 files to the directory `test_out`:
- `training_data.csv` a CSV file containing the data as parsed and preprocessed by the modified SENDQSAR library prior to training the model.
- `testing_data.csv` a CSV file containing the data as parsed and preprocessed by the modified SENDQSAR library prior to testing the model.
- `test_prediction.csv` a CSV file containing the hepatotoxicity prediction for each sample in the `testing_data.csv`. 
The first column `STUDYID` represents the test sample ID and the second column `Predicted hepatotoxicity score` 
represents the liver toxicity prediction where `1` represents hepatotoxicity and `0` represents no hepatotoxicity.

## Resource Requirements

Running tests and training the model can be done on a moderately powerful laptop. 
The laptop I developed this app on was a 16-core 4.5ghz CPU with 16GB of DDR5 RAM and an NVIDIA RTX 4060 Laptop GPU with 8 GB of VRAM.
Given the size of the docker image, it would be recommended that the user run this image on a machine with at least 32 GB of RAM.

## Implementation

Using tabSTAR as the base-model, this app trains, implements and evaluates a tab-based Transformer model incorporating 
the SEND data provided by this challenge. This app uses some of the functions implemented in a modification of the 
SENDQSAR R library as originally developed by Md Aminul Islam Prodhan which were used in scripts for data extraction and 
compilation for initial and iterative ML model training.

Training was done on a first-pass 200 epoch run using a learning rate of 0.001 and a batch size of 16 samples.
Training data used the xpt training and testing data that was given in the example app for predicting 
hepatotoxicity in Animals.

## Resource Links

#### Project github repo
https://github.com/cmarkello/FDAChallenges/tree/main/predictive_model_haepatotoxicity_in_animals

#### Research
https://eilamshapira.com/TabSTAR/
https://github.com/alanarazi7/TabSTAR/tree/master
https://aminuldu07.github.io/SENDQSAR/
https://github.com/aminuldu07/SENDQSAR/tree/main

Paper Reference:
Alan Arazi, Eilam Shapira, & Roi Reichart (2025). 
TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations. 
arXiv preprint arXiv:2505.18125.

## License Disclosure
The submitter of this application, Charles Markello has the right to submit the model/software. All third-party tool 
licenses are MIT licenses and therefore their use in this app are in compliance with their terms.