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

  make_option(c("--training_data_format"), type = "character", default = "xpt",
              help = "Training data file format. Either 'xpt' or 'csv'. Default: 'xpt'.", metavar = "character"),

  make_option(c("-o", "--output_dir"), type = "character", default = "output_data",
              help = "Path to the output CSV files", metavar = "character")
)
# # Create OptionParser object
opt_parser <- OptionParser(option_list = option_list)

# Parse command-line arguments (using "opt_parser" or simulated `args`)
opt <- parse_args(opt_parser)

# Set the zip file and output file path from command line arguments
training_zip_file <- opt$training_zip_file

training_data_labels<- opt$training_data_labels

testing_zip_file <- opt$testing_zip_file
testing_data_labels <- opt$testing_data_labels
training_data_format <- opt$training_data_format
output_dir <- opt$output_dir

# Debug prints

cat("Training ZIP:", training_zip_file, "\n")
cat("Training CSV:", training_data_labels, "\n")
if (!is.null(testing_zip_file)) {
    cat("Testing ZIP:", testing_zip_file, "\n")
}
if (!is.null(testing_data_labels)) {
    cat("Testing CSV:", testing_data_labels, "\n")
}

# Check that all input files exist
stopifnot(file.exists(training_zip_file))
stopifnot(file.exists(training_data_labels))

# Create a base directory for all extracted files
temp_dir_base <- paste(output_dir, tempfile(), sep="")
dir.create(temp_dir_base, mode = "777", recursive = TRUE)

# Extract training zip
archive_extract(training_zip_file, dir = temp_dir_base)
print("Training zip extracted successfully.")

# Extract testing zip
if (!is.null(testing_zip_file)) {
    temp_dir_base_test <- paste(output_dir, tempfile(), sep="")
    dir.create(temp_dir_base_test, mode = "777", recursive = TRUE)
    archive_extract(testing_zip_file, dir = temp_dir_base_test)
    print("Testing zip extracted successfully.")
}



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

if (!is.null(testing_zip_file)) {
  # List all files and directories recursively
  all_extracted_files_test <- fs::dir_ls(temp_dir_base_test, recurse = TRUE, all = TRUE)

  # Extract unique directory paths
  unique_directories_test <- unique(fs::path_dir(all_extracted_files_test))

  # Assign the base directory as path_db
  path_db_test <- unique_directories_test[1]

  # Assign the rest as studyid_studyids
  studyid_studyids_test <- unique_directories_test[-1]

  cat("Base test directory:", path_db_test, "\n")
  cat("Study test directories:\n")
  print(studyid_studyids_test)
}

replace_colname_lookup <- c(Target_Organ = "Label", Target_Organ = "LABEL") # Rename "Label" to "Target_Organ" for consistency

# Read training/testing CSVs (provided directly, not from ZIP)
training_csv <- read.csv(training_data_labels)
training_csv <- training_csv %>% rename(any_of(replace_colname_lookup))
cat("Loaded training CSV:\n")
print(head(training_csv))

if (!is.null(testing_data_labels)) {
    testing_csv <- read.csv(testing_data_labels)
    testing_csv$STUDYID <- as.character(testing_csv$STUDYID)
    testing_csv <- testing_csv %>% rename(any_of(replace_colname_lookup))
    cat("Loaded testing CSV:\n")
    print(head(testing_csv))
}

# Combine training and testing CSV
combined_csv <- training_csv
combined_csv$STUDYID <- as.character(combined_csv$STUDYID)  # Ensure consistency
combined_csv <- combined_csv %>% rename(any_of(replace_colname_lookup))
print(head(combined_csv))




#----------------------------------------------------------------------------
liver_scores_training <- get_liver_om_lb_mi_tox_score_list(studyid_or_studyids = studyid_studyids,
                                                           path_db = path_db,
                                                           fake_study = TRUE,
                                                           use_xpt_file = (training_data_format == 'xpt'),
                                                           output_individual_scores = TRUE,
                                                           output_zscore_by_USUBJID = FALSE)
write.csv(liver_scores_training, file="/home/cjmarkello/precisionFDAassetts/Predictive_Modeling_of_Hepatotoxicity/debug_output/liver_scores_training.csv")
#-----------column harmonization of "liver_scores"-------------
liver_scores_col_harmonized <- get_col_harmonized_scores_df(liver_score_data_frame=liver_scores_training,
                                                            Round = TRUE)
write.csv(liver_scores_col_harmonized, file="/home/cjmarkello/precisionFDAassetts/Predictive_Modeling_of_Hepatotoxicity/debug_output/liver_scores_col_harmonized_training.csv")

# Merge csv and scores data frame
liver_scores_target_organ <- inner_join(combined_csv, liver_scores_col_harmonized , by = "STUDYID")

# get the training data and testing data
training_data <- liver_scores_target_organ[(toupper(liver_scores_target_organ$Target_Organ) == "LIVER" |toupper(liver_scores_target_organ$Target_Organ) ==  "NOT_LIVER"), ]

# Replace only the target_organ column values where they are "liver" with 1
training_data$Target_Organ[toupper(training_data$Target_Organ) == "LIVER"] <- 1

# Replace only the target_organ column values where they are "not_liver" with 0
training_data$Target_Organ[toupper(training_data$Target_Organ) == "NOT_LIVER"] <- 0

# Convert the target_organ column to a numeric factor with levels 1 and 0
training_data$Target_Organ <- factor(training_data$Target_Organ, levels = c(1, 0))

write.csv(training_data, file = paste(output_dir, "/training_data.csv", sep = ""), row.names = FALSE)

# creating testing data
if (!is.null(testing_zip_file)) {
    liver_scores_testing <- get_liver_om_lb_mi_tox_score_list(studyid_or_studyids = studyid_studyids_test,
                                                           path_db = path_db_test,
                                                           fake_study = TRUE,
                                                           use_xpt_file = (training_data_format == 'xpt'),
                                                           output_individual_scores = TRUE,
                                                           output_zscore_by_USUBJID = FALSE)
    ## DEBUG
    write.csv(liver_scores_testing, file="/home/cjmarkello/precisionFDAassetts/Predictive_Modeling_of_Hepatotoxicity/debug_output/liver_scores_testing.csv")
    #-----------column harmonization of "liver_scores"-------------
    liver_scores_col_harmonized <- get_col_harmonized_scores_df(liver_score_data_frame=liver_scores_testing,
                                                                Round = TRUE)
    write.csv(liver_scores_col_harmonized, file="/home/cjmarkello/precisionFDAassetts/Predictive_Modeling_of_Hepatotoxicity/debug_output/liver_scores_col_harmonized_testing.csv")

    # Merge csv and scores data frame
    #   extract and hold test samples
    if (!is.null(testing_data_labels)) {
      liver_scores_target_organ <- inner_join(testing_csv, liver_scores_col_harmonized , by = "STUDYID")
      #   then extract from liver_scores_target_organ via those sample IDs
      testing_data <- liver_scores_target_organ[(toupper(liver_scores_target_organ$Target_Organ) == "TESTING_DATA_LIVER" |toupper(liver_scores_target_organ$Target_Organ) ==  "TESTING_DATA_NOT_LIVER"), ]
      # Replace only the target_organ column values where they are "liver" with 1
      testing_data$Target_Organ[toupper(testing_data$Target_Organ) == "TESTING_DATA_LIVER"] <- 1

      # Replace only the target_organ column values where they are "not_liver" with 0
      testing_data$Target_Organ[toupper(testing_data$Target_Organ) == "TESTING_DATA_NOT_LIVER"] <- 0

      # Convert the target_organ column to a numeric factor with levels 1 and 0
      testing_data$Target_Organ <- factor(testing_data$Target_Organ, levels = c(1, 0))
    } else {
      testing_data <- liver_scores_col_harmonized
    }

    write.csv(testing_data, file = paste(output_dir, "/testing_data.csv", sep = ""), row.names = FALSE)
}

# Print a message indicating where the files have been saved
cat("Results have been written to", output_dir, "\n")
