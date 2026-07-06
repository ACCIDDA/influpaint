# ---------------------------------------------------------------------------
# validation_split.R
#
# Split the filled RSV dataset into:
#   * RSV_TRAIN.parquet      -> everything the model is allowed to see
#   * RSV_VALIDATION.parquet -> seasons we hold out for later evaluation
#                               (this is also the file to send to Yue)
#
# "Hold out" means: every row whose fluseason is in VALIDATION_SEASONS is
# removed from training, across *all* data sources (NSSP, RSV-Net, NHSN,
# RSV_SMH). That prevents any leakage of the validation truth into training,
# including through synthetic trajectories that were produced for those years.
#
# Pick 1-2 seasons. A few things to keep in mind when choosing:
#   - We hold out the latest season (2025 = the 2025-26 season): train on the
#     past, forecast the most recent season. After this split the training set
#     has 12 surveillance frames (NHSN 2, NSSP 3, RSV-Net 7).
#   - NHSN only starts at season 2023, so holding out 2025 leaves NHSN with two
#     training seasons (2023, 2024).
#   - RSV-Net has many seasons (2018+), so dropping any single one is cheap.
#   - Evaluation needs surveillance data for the held-out season, which lives
#     in RSV_VALIDATION.parquet after this split.
# ---------------------------------------------------------------------------

library(arrow)
library(dplyr)

# ---- Configuration --------------------------------------------------------
VALIDATION_SEASONS <- c(2025) # fluseason start years to hold out (2025-26 season)

INPUT_PATH <- "RSV/data/RSV_FILLED.parquet"
TRAIN_OUT <- "RSV/data/RSV_TRAIN.parquet"
VALID_OUT <- "RSV/data/RSV_VALIDATION.parquet"

# ---- Split ----------------------------------------------------------------
rsv <- read_parquet(INPUT_PATH)

validation <- rsv |> filter(fluseason %in% VALIDATION_SEASONS)
training <- rsv |> filter(!(fluseason %in% VALIDATION_SEASONS))

write_parquet(validation, VALID_OUT)
write_parquet(training, TRAIN_OUT)

# ---- Report ---------------------------------------------------------------
cat("Held-out seasons:", VALIDATION_SEASONS, "\n")
cat("Validation rows :", nrow(validation), "\n")
cat("Training rows   :", nrow(training), "\n\n")

cat("Rows per data source in training set:\n")
print(training |> count(datasetH1))
cat("\nRows per data source in validation set:\n")
print(validation |> count(datasetH1))
