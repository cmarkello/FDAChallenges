
https://precision.fda.gov/challenges/35/intro
This precisionFDA challenge invites participants to develop predictive toxicology models from animal study data produced in compliance with the Standard for Exchange of Nonclinical Data (SEND).

The purpose of this Challenge is to
- Evaluate how well computational approaches using SEND data can reproduce the hepatotoxicity risk determined by toxicologists
- Create public tools for deriving hepatotoxicity risks from SEND data
- Improve the usability of existing SEND data to advance the roadmap to reducing animal testing in preclinical safety studies


## Implementation

Using tabSTAR as the base-model, this app trains and evaluates and implements a tab-based ML model incorporating the SEND data
provided by this challenge. Uses some of the SENDQSAR functions and scripts for data extraction and compilation for 
initial and iterative ML model training.

Resource Links:
https://eilamshapira.com/TabSTAR/
https://github.com/alanarazi7/TabSTAR/tree/master
https://aminuldu07.github.io/SENDQSAR/
https://github.com/aminuldu07/SENDQSAR/tree/main

Paper Reference:
TabSTAR: A Foundation Tabular Model With Semantically Target-Aware Representations},
Alan Arazi and Eilam Shapira and Roi Reichart,
arXiv preprint arXiv:2505.18125, 2025.