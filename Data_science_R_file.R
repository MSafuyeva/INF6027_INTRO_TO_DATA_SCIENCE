# Load necessary library
install.packages("dplyr")

library(dplyr)
library(xgboost)
library(caret)
library(pROC)
library(Information)

# Read the datasets 
data_2016 <- read.csv("2016_Financial_Data.csv", stringsAsFactors = FALSE)
data_2017 <- read.csv("2017_Financial_Data.csv", stringsAsFactors = FALSE)
data_2018 <- read.csv("2018_Financial_Data.csv", stringsAsFactors = FALSE)


# Check if column names are identical
identical(names(data_2016), names(data_2017)) # TRUE or FALSE
identical(names(data_2017), names(data_2018)) # TRUE or FALSE

colnames(data_2016)[colnames(data_2016) == "X2017.PRICE.VAR...."] <- "PRICE_VAR"
colnames(data_2017)[colnames(data_2017) == "X2018.PRICE.VAR...."] <- "PRICE_VAR"
colnames(data_2018)[colnames(data_2018) == "X2019.PRICE.VAR...."] <- "PRICE_VAR"

# Display the first few rows of the dataset
head(data_2016)
head(data_2017)
head(data_2018)

# Remove first column from each dataset
data_2016 <- data_2016[, -1]
data_2017 <- data_2017[, -1]
data_2018 <- data_2018[, -1]

# Check if the first column was removed
head(data_2016)

# Count missing values for each column in each dataset
colSums(is.na(data_2016)) 

# Check for missing values in data_2016
sum(is.na(data_2016))  # Total number of missing values

# Function to identify column types
get_column_types <- function(df) {
  sapply(df, function(col) {
    if (is.numeric(col)) {
      return("numeric")
    } else if (is.factor(col) || is.character(col)) {
      return("categorical")
    } else {
      return("other")
    }
  })
}

# Apply the function to dataframe
column_types <- get_column_types(data_2016)

# Display the column types
print(column_types)

# Count the number of each type
type_counts <- table(column_types)

# Display the column types and their counts
print(type_counts)

# Convert only categoric column to numeric one
data_2016$Sector <- as.numeric(as.factor(data_2016$Sector))
data_2017$Sector <- as.numeric(as.factor(data_2017$Sector))
data_2018$Sector <- as.numeric(as.factor(data_2018$Sector))

# Function to fill missing values in numeric columns with the median
impute_median <- function(df) {
  for (col in names(df)) {
    if (is.numeric(df[[col]])) {
      # Replace missing values with median
      df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
    }
  }
  return(df)
}

# Apply the function to each dataset
data_2016 <- impute_median(data_2016)
data_2017 <- impute_median(data_2017)
data_2018 <- impute_median(data_2018)

# Check for remaining missing values
colSums(is.na(data_2016))  
colSums(is.na(data_2017))  
colSums(is.na(data_2018)) 

# Function to detect outliers using IQR
detect_outliers <- function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(which(column < lower_bound | column > upper_bound))
}


# Cap outliers to the nearest valid value
cap_outliers <- function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  column[column < lower_bound] <- lower_bound
  column[column > upper_bound] <- upper_bound
  return(column)
}

# Cap outliers for Revenue
data_2016$Revenue <- cap_outliers(data_2016$Revenue)

# Check summary statistics to ensure outliers are capped
summary(data_2016)
summary(data_2017)
summary(data_2018)

# Visualize the distribution of a numeric column (e.g., Revenue) after handling outliers
boxplot(data_2016$Revenue, main = "Boxplot of Revenue (2016 - After Outlier Handling)")
boxplot(data_2017$Revenue, main = "Boxplot of Revenue (2017 - After Outlier Handling)")
boxplot(data_2018$Revenue, main = "Boxplot of Revenue (2018 - After Outlier Handling)")

# Outliers have inherent meaning in this dataset, for this reason after capping it is more suitable
# to keep them as they are

# Combine all datasets into a single dataframe
combined_data <- bind_rows(data_2016, data_2017, data_2018)

# Check structure of the combined dataset
str(combined_data)

# Preview the combined dataset
head(combined_data)

# Check column names
colnames(combined_data)

# Check data types
str(combined_data)

# Count missing values in each column
colSums(is.na(combined_data))


#Feature Engineering for Logistic Regression

# Calculate Information Value (IV) for all variables
iv_summary <- create_infotables(data = combined_data, y = "Class", bins = 10, parallel = FALSE)

# Display IV summary
print(iv_summary$Summary)

# Filter variables with 0.1 ≤ IV ≤ 0.5
selected_vars <- iv_summary$Summary$Variable[iv_summary$Summary$IV >= 0.1 & iv_summary$Summary$IV <= 0.5]

# Include the Class variable in the final dataset
final_model_data <- combined_data[, c(selected_vars, "Class")]

# Display selected variables
cat("Selected Variables Based on IV (0.1 ≤ IV ≤ 0.5):\n")
print(selected_vars)

# Calculate Correlation Matrix for the dataset
correlation_matrix <- cor(final_model_data, use = "complete.obs")

# Identify Highly Correlated Pairs (|correlation| > 0.8)
high_corr <- which(abs(correlation_matrix) > 0.8, arr.ind = TRUE)

# Remove self-correlations
high_corr <- high_corr[high_corr[, 1] != high_corr[, 2], ]

# Create a data frame of correlated pairs
correlated_pairs <- data.frame(
  Variable1 = rownames(correlation_matrix)[high_corr[, 1]],
  Variable2 = colnames(correlation_matrix)[high_corr[, 2]],
  Correlation = correlation_matrix[high_corr]
)

print(correlated_pairs)

# Remove variable with the least Information Value (IV) in each correlated pair
variables_to_remove <- c()

for (i in seq_len(nrow(correlated_pairs))) {
  var1 <- correlated_pairs$Variable1[i]
  var2 <- correlated_pairs$Variable2[i]
  
  iv_var1 <- iv_summary$Summary$IV[iv_summary$Summary$Variable == var1]
  iv_var2 <- iv_summary$Summary$IV[iv_summary$Summary$Variable == var2]
  
  if (iv_var1 < iv_var2) {
    variables_to_remove <- c(variables_to_remove, var1)
  } else {
    variables_to_remove <- c(variables_to_remove, var2)
  }
}

# Remove duplicate entries
variables_to_remove <- unique(variables_to_remove)

# Final dataset after handling correlations
final_cleaned_data <- final_model_data[, !names(final_model_data) %in% variables_to_remove]

# Output results
cat("\nVariables removed due to multicollinearity (least IV):\n")
print(variables_to_remove)

cat("\nFinal dataset summary:\n")
print(summary(final_cleaned_data))


# Modelling
# Step 1: Prepare the Data
set.seed(123)  # For reproducibility

# Ensure Class variable is a factor
final_cleaned_data$Class <- as.factor(final_cleaned_data$Class)

# Split the data into training and testing sets (70% train, 30% test)
train_indices <- sample(seq_len(nrow(final_cleaned_data)), size = 0.7 * nrow(final_cleaned_data))
train_data <- final_cleaned_data[train_indices, ]
test_data <- final_cleaned_data[-train_indices, ]

# Step 2: Build the Logistic Regression Model
logistic_model <- glm(Class ~ ., data = train_data, family = binomial)

# Summary of the model
cat("\nLogistic Regression Model Summary:\n")
print(summary(logistic_model))

# Predictions and Evaluation
predicted_probs <- predict(logistic_model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)  # Threshold = 0.5
predicted_classes <- as.factor(predicted_classes)

# Confusion Matrix
cat("\nConfusion Matrix:\n")
confusion <- confusionMatrix(predicted_classes, test_data$Class)
print(confusion)

# Calculate and Plot AUC-ROC Curve
roc_curve <- roc(test_data$Class, predicted_probs)

cat("\nAUC Value:\n")
print(auc(roc_curve))

# Plot ROC Curve
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)



# Convert the target variable to numeric for xgboost
final_cleaned_data$Class <- as.numeric(as.factor(final_cleaned_data$Class)) - 1

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(final_cleaned_data), 0.7 * nrow(final_cleaned_data))
train_data <- final_cleaned_data[train_indices, ]
test_data <- final_cleaned_data[-train_indices, ]

# Separate features and target variable
train_matrix <- as.matrix(train_data[, !colnames(train_data) %in% "Class"])
train_labels <- train_data$Class
test_matrix <- as.matrix(test_data[, !colnames(test_data) %in% "Class"])
test_labels <- test_data$Class

# Create DMatrix for xgboost
train_dmatrix <- xgb.DMatrix(data = train_matrix, label = train_labels)
test_dmatrix <- xgb.DMatrix(data = test_matrix, label = test_labels)

# Set xgboost parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.3,
  nthread = 2
)

# Train the xgboost model
xgb_model <- xgb.train(
  params = params,
  data = train_dmatrix,
  nrounds = 100,
  watchlist = list(train = train_dmatrix, test = test_dmatrix),
  verbose = 1
)

# Predict probabilities on the test data
predicted_probs <- predict(xgb_model, newdata = test_matrix)

# Convert probabilities to binary predictions
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

# Calculate accuracy
accuracy <- sum(predicted_classes == test_labels) / length(test_labels)
cat("Accuracy of XGBoost Model:", round(accuracy * 100, 2), "%\n")

# Generate the ROC curve
roc_curve <- roc(test_labels, predicted_probs)



# Plot the ROC curve with improved aesthetics
plot(roc_curve, 
     main = "ROC Curve for XGBoost Model", 
     col = "red", 
     lwd = 3,              
     cex.main = 1.5,        
     cex.axis = 1.2,        
     cex.lab = 1.3,         
     font.main = 2)        

# Calculate AUC value
auc_value <- auc(roc_curve)

# Add AUC value as a legend with bold and larger text
legend("bottomright", 
       legend = paste("AUC =", round(auc_value, 2)), 
       col = "red", 
       lwd = 3, 
       cex = 1.3,         
       text.font = 2,   
       bty = "n")     





# Get actual and predicted classes 
actual_classes <- test_data$Class 
predicted_classes <- predicted_classes 

# Create a confusion matrix
conf_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)

# Convert the confusion matrix into a data frame for ggplot
conf_matrix_df <- as.data.frame(as.table(conf_matrix))
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Count")

# Plot the heatmap
ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "pink", high = "blue") +
  geom_text(aes(label = Count), color = "black", size = 4) +
  labs(
    title = "Confusion Matrix Heatmap",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )


# Plot Logistic Regression Feature Importance
ggplot(logistic_top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "darkblue", width = 0.7) +  
  coord_flip() +
  labs(
    title = "Feature Importance (Logistic Regression) - Top Features", 
    x = "Features", 
    y = "Coefficient Magnitude"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 14),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )




# Extract Feature Importance
importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)
xgb_df <- as.data.frame(importance_matrix)

# Select Top 10 Features
xgb_top_features <- xgb_df %>%
  arrange(desc(Gain)) %>%
  head(top_n)

# Plot XGBoost Feature Importance
ggplot(xgb_top_features, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "darkred", width = 0.7) +  
  coord_flip() +
  labs(
    title = "Feature Importance (XGBoost) - Top Features", 
    x = "Features", 
    y = "Importance (Gain)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 14),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )


