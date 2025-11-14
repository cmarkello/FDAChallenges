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
  make_option(c("--testing_zip_file"), type = "character", default = NULL,
              help = "Path to the testing zip file", metavar = "character"),

  make_option(c("--testing_data_format"), type = "character", default = "xpt",
              help = "Testing data file format. Either 'xpt' or 'csv'. Default: 'xpt'.", metavar = "character"),

  make_option(c("-o", "--output_dir"), type = "character", default = "output_data",
              help = "Path to the output CSV files", metavar = "character")
)
# # Create OptionParser object
opt_parser <- OptionParser(option_list = option_list)

# Parse command-line arguments (using "opt_parser" or simulated `args`)
opt <- parse_args(opt_parser)

# Set the zip file and output file path from command line arguments
testing_zip_file <- opt$testing_zip_file
testing_data_format <- opt$testing_data_format
output_dir <- opt$output_dir

# Debug prints

cat("Testing ZIP:", testing_zip_file, "\n")
cat("Testing data format:", testing_data_format, "\n")

# Check that all input files exist
stopifnot(file.exists(testing_zip_file))

# Create a base directory for all extracted files
temp_dir_base <- paste(output_dir, tempfile(), sep="")
dir.create(temp_dir_base, mode = "777", recursive = TRUE)

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

replace_colname_lookup <- c(Target_Organ = "Label", Target_Organ = "LABEL") # Rename "Label" to "Target_Organ" for consistency

# Read training/testing CSVs (provided directly, not from ZIP)
testing_csv <- data.frame("STUDYID" = studyid_studyids, Target_Organ = "NA")
testing_csv <- testing_csv %>% rename(any_of(replace_colname_lookup))
cat("Loaded testing CSV:\n")
print(head(testing_csv))


# Combine training and testing CSV
if (!is.null(testing_zip_file)) {
    combined_csv_test <- testing_csv
    combined_csv_test$STUDYID <- as.character(combined_csv_test$STUDYID)  # Ensure consistency
    combined_csv_test <- combined_csv %>% rename(any_of(replace_colname_lookup))
    print(head(combined_csv_test))
}

#----------------------------------------------------------------------------
liver_scores <- get_liver_om_lb_mi_tox_score_list(studyid_or_studyids = studyid_studyids,
                                                   path_db = path_db,
                                                   fake_study = TRUE,
                                                   use_xpt_file = (training_data_format == 'xpt'),
                                                   output_individual_scores = TRUE,
                                                   output_zscore_by_USUBJID = FALSE)

write.csv(liver_scores, "/home/cjmarkello/precisionFDAassetts/Predictive_Modeling_of_Hepatotoxicity/debug_output/liver_scores.csv")
#-----------column harmonization of "liver_scores"-------------
liver_scores_col_harmonized <- get_col_harmonized_scores_df(liver_score_data_frame=liver_scores,
                                                            Round = TRUE)
# Merge csv and scores data frame
liver_scores_target_organ <- inner_join(combined_csv, liver_scores_col_harmonized , by = "STUDYID")

# creating testing data
#   extract and hold test samples
#   then extract from liver_scores_target_organ via those sample IDs
testing_data <- liver_scores_target_organ

write.csv(testing_data, file = paste(output_dir, "/testing_data.csv", sep = ""), row.names = FALSE)

# Print a message indicating where the files have been saved
cat("Results have been written to", output_dir, "\n")
