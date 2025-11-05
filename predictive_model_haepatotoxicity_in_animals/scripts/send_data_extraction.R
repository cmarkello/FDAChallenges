#!/usr/bin/env Rscript

# Clear the current R environment
rm(list = ls())

# Load required libraries silently
suppressPackageStartupMessages({
  library(dplyr)
  library(optparse)
  #library(zip)  # Library to handle zip files
  library(archive)
  library(fs)
  library(this.path)
  library(SENDQSAR)
})

# Set File Path
homePath <- dirname(this.path())
setwd(homePath)
print(homePath)

# Define command line arguments
option_list <- list(
  make_option(c("--training_zip_file"), type = "character", default = NULL,
              help = "Path to the training zip file", metavar = "character"),

  make_option(c("--training_data_labels"), type = "character", default = NULL,
              help = "Path to the training CSV file", metavar = "character"),

  make_option(c("--testing_zip_file"), type = "character", default = NULL,
              help = "Path to the testing zip file", metavar = "character"),

  make_option(c("--testing_data_labels"), type = "character", default = NULL,
              help = "Path to the testing CSV file", metavar = "character"),

)
# # Create OptionParser object
 opt_parser <- OptionParser(option_list = option_list)

 # Parse command-line arguments (using "opt_parser" or simulated `args`)
 opt <- parse_args(opt_parser)

# Set the zip file and output file path from command line arguments
training_zip_file <- opt$training_zip_file
testing_zip_file <- opt$testing_zip_file
training_data_labels<- opt$training_data_labels
testing_data_labels <- opt$testing_data_labels

# Debug prints
cat("Training ZIP:", training_zip_file, "\n")
cat("Training CSV:", training_data_labels, "\n")
cat("Testing ZIP:", testing_zip_file, "\n")
cat("Testing CSV:", testing_data_labels, "\n")

# Check that all input files exist
stopifnot(file.exists(training_zip_file))
stopifnot(file.exists(testing_zip_file))
stopifnot(file.exists(training_data_labels))
stopifnot(file.exists(testing_data_labels))

# Create a base directory for all extracted files
temp_dir_base <- tempfile()
dir.create(temp_dir_base)

# Extract training zip
archive_extract(training_zip_file, dir = temp_dir_base)
print("Training zip extracted successfully.")

# Extract testing zip
archive_extract(testing_zip_file, dir = temp_dir_base)
print("Testing zip extracted successfully.")



# List all files and directories recursively
all_extracted_files <- fs::dir_ls(temp_dir_base, recurse = TRUE, all = TRUE)

# Extract unique directory paths
unique_directories <- unique(fs::path_dir(all_extracted_files))

# Assign the base directory as path_db
path_db <- unique_directories[1]

# Assign the rest as studyid_studyids
studyid_studyids <- unique_directories[-1]

cat("Base directory:", path_db, "\n")
cat("Study directories:\n")
print(studyid_studyids)

# Read training/testing CSVs (provided directly, not from ZIP)
training_csv <- read.csv(training_data_labels)
testing_csv <- read.csv(testing_data_labels)

cat("Loaded training CSV:\n")
print(head(training_csv))

cat("Loaded testing CSV:\n")
print(head(testing_csv))

# Combine training and testing CSV
combined_csv <- rbind(training_csv, testing_csv)
combined_csv$STUDYID <- as.character(combined_csv$STUDYID)  # Ensure consistency
print(head(combined_csv))

#----------------------------------------------------------------------------

liver_scores <- get_liver_om_lb_mi_tox_score_list (studyid_or_studyids = studyid_studyids,
                                                   path_db = path_db,
                                                   fake_study = TRUE,
                                                   use_xpt_file =  TRUE,
                                                   output_individual_scores = TRUE,
                                                   output_zscore_by_USUBJID = FALSE)


#-----------column harmonization of "liver_scores"-------------
liver_scores_col_harmonized <- get_col_harmonized_scores_df(liver_score_data_frame=liver_scores,
                                                            Round = TRUE)

# Merge csv and scores data frame
liver_scores_target_organ <- inner_join(combined_csv, liver_scores_col_harmonized , by = "STUDYID")

# get the training data and testing data
training_data <- liver_scores_target_organ[!(liver_scores_target_organ$Target_Organ == "testing_data_Liver" |liver_scores_target_organ$Target_Organ ==  "testing_data_not_Liver"), ]

# Replace only the target_organ column values where they are "liver" with 1
training_data$Target_Organ[training_data$Target_Organ == "Liver"] <- 1

# Replace only the target_organ column values where they are "not_liver" with 0
training_data$Target_Organ[training_data$Target_Organ == "not_Liver"] <- 0

# Convert the target_organ column to a factor with levels 1 and 0
training_data$Target_Organ <- factor(training_data$Target_Organ, levels = c(1, 0))

# creating testing data
testing_data <- liver_scores_target_organ[(liver_scores_target_organ$Target_Organ == "testing_data_Liver" |liver_scores_target_organ$Target_Organ ==  "testing_data_not_Liver"), ]

write.csv(training_data, file = "training_data.csv", row.names = FALSE)
write.csv(testing_data, file = "testing_data.csv", row.names = FALSE)
