# ---
# script: 04-ModelBuilding
# subject: Model building
# date: 2019-07-19
# author: Marcus Di Paula
# github: github.com/marcusdipaula/
# linkedin: linkedin.com/in/marcusdipaula/
# ---



# 4. Building and validating models
#     - Train and test a ML model
#         I plan to use 3 algorithms:
#           randomForest (from caret package)
#           Boosted Tree (from caret package)
#           autoML (from h2o package)
#
#     - Which performance metrics should I relly on?
#       I'll relly on Accuracy


# Loading the caret package
if(!require(caret)) {install.packages("caret"); library(caret)}

# Creating an index to split the dataset into training and test sets
index <- createDataPartition(ds$is_attributed, p = 0.01, list = FALSE)

small_ds <- ds[index,]

index <- createDataPartition(small_ds$is_attributed, p = 0.7, list = FALSE)

# Spliting the dataset
train <- small_ds[index,]
test <- small_ds[-index,]; rm(index, small_ds)

#___________________________________________________________________________

# Random Forest
# Setting parameters that further control how the model will be created
basic_par_rf <- trainControl(
   method = "cv", # The resampling method (cross-validation)
   number = 5, # the number of folds in K-fold cross-validation
   #repeats = 3, # applied only to "repeatedcv" method
   returnData = F, # a logical for saving the used data into a slot called trainingData
   #classProbs = TRUE, # Estimate class probabilities, so would be possible to score models using the area under the ROC curve
   allowParallel = T) # a logical that governs whether train should use parallel processing (if availible)
   #sampling = "smote") # balance the classes prior to model fitting

# Training the random forest model
model_rf <- train(is_attributed ~ .,
                  data = train,
                  method = "rf",
                  ntree = 100,
                  nodesize = 10,
                  metric = "Accuracy",
                  trControl = basic_par_rf); rm(basic_par_rf)

# Saving the model
saveRDS(model_rf, "./models/model_rf.RDS")
# Using the model to predict the test subset
prediction_rf <- tibble(observed = test$is_attributed,
                        predicted = predict.train(model_rf, newdata = test))

# Viewing the confusion matrix
confusionMatrix(prediction_rf$predicted, prediction_rf$observed); rm(prediction_rf)

#___________________________________________________________________________

# Boosted Tree
# Setting parameters that further control how the model will be created
basic_par_bstTree <- trainControl(
    method = "cv", # The resampling method (cross-validation)
    number = 5, # the number of folds in K-fold cross-validation
    #repeats = 3, # applied only to "repeatedcv" method
    search = "random", # To use a random sample of possible tuning parameter combinations
    returnData = F, # a logical for saving the used data into a slot called trainingData
    #classProbs = TRUE, # Estimate class probabilities, so would be possible to score models using the area under the ROC curve
    allowParallel = T) # a logical that governs whether train should use parallel processing (if availible)
    #sampling = "smote") # balance the classes prior to model fitting


# Training the boosted tree model
model_bstTree <- train(is_attributed ~ .,
                       data = train,
                       method = "bstTree",
                       trControl = basic_par_bstTree,
                       tuneLength = 5,
                       metric = "Accuracy"); rm(basic_par_bstTree)

# Saving the model
saveRDS(model_bstTree, "./models/model_bstTree.RDS")
# Using the model to predict the test subset
prediction_bstTree <- tibble(observed = test$is_attributed,
                             predicted = predict.train(model_bstTree, newdata = test))

# Viewing the confusion matrix
confusionMatrix(prediction_bstTree$predicted, prediction_bstTree$observed); rm(prediction_bstTree)


#___________________________________________________________________________

# Auto ML with h2o package
# Mor info at:
# https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/R/automl_binary_classification_product_backorders.Rmd

# Loading the h2o package
if(!require(h2o)) {install.packages("h2o"); library(h2o)}

# Initializing and connecting to a h2o instance
h2o.init(nthreads = -1, #Number of threads: -1 means use all cores on your machine
         max_mem_size = "6G")  #max mem size is the maximum memory to allocate to H2O

# Importing the dataset into a h2o cluster (creates a "H2OFrame" object)
df <- h2o.importFile("ds.csv", col.types = c("int",
                                             "int",
                                             "int",
                                             "int",
                                             "int",
                                             "time",
                                             "time",
                                             "factor"))

# Looking at the description of the dataset
# since we want a classification task, we need the target variable as "factor" or "enum"
h2o.describe(df)

# Specifying the target variable (y) and the prediction ones (x)
y <- "is_attributed"
x <- setdiff(names(df), c(y, "attributed_time"))


# Training the model
auto_ml <- h2o.automl(y = y,
                      x = x,
                      training_frame = df,
                      max_models = 10) # this specifies the max number of models
                                       # does not include the two "ensemble models" that are trained at the end

# Printing the top models
print(auto_ml@leaderboard)

# The winner is the Gradient Boosting altorithm with id GBM_4_AutoML_20190716_203134
# with an AUROC of 0.9991711

# Saving the leader model in a binary format
h2o.saveModel(aml@leader, path = "./auto_ml/")

# Saving the leader model in a format called MOJO, that is optimized for production
h2o.download_mojo(aml@leader, path = "./auto_ml/")







