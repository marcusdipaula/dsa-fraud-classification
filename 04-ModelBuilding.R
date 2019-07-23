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

#_______________________________________Algorithm_01______________________________________#

# Random Forest algorithm
#
# Caret method: 'rf'
#
# Type: Classification, Regression
#
# Tags (types or relevant characteristics according to the caret package guide):
# Bagging, Ensemble Model, Implicit Feature Selection, Random Forest, Supports Class Probabilities
#
# Tuning parameters: mtry (#Randomly Selected Predictors)
#
# Required packages: randomForest
#
# More info: A model-specific variable importance metric is available.
#
# Link to know more: http://topepo.github.io/caret/


# Setting parameters that further control how the model will be created
# More at: http://topepo.github.io/caret/model-training-and-tuning.html#basic-parameter-tuning
basic_par_rf <- trainControl(method = "cv", # The resampling method (cross-validation)

                             number = 5, # the number of folds in K-fold cross-validation, so we have 5-fold
                             # cross-validations

                             #repeats = 3, # applied only to "repeatedcv" method.
                             # Repetitions of the previous k-fold cross-validation, so we'd have
                             # 3 repetitions of 5-fold cross-validations
                             # An illustration: https://www.evernote.com/l/AGKmXIbis1dHSbR_j9dblVk-t3klmWsL_i0/

                             returnData = F) # a logical for saving the used data into a slot called trainingData
                             #classProbs = TRUE, # Estimate class probabilities, so would be possible to score models using the area under the ROC curve

                             #allowParallel = T, # a logical that governs whether train should use parallel processing (if availible)
                          	 # about parallel processing: # http://dept.stat.lsa.umich.edu/~jerrick/courses/stat701/notes/parallel.html

                             #sampling = "smote") # balance the classes prior to model fitting
                             # More at: https://topepo.github.io/caret/subsampling-for-class-imbalances.html

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
prediction_rf <- bind_cols(predict.train(model_rf,
                                          newdata = test,
                                          type = "prob"),
                            tibble(observed = test$is_attributed,
                                   predicted = predict.train(model_rf, newdata = test)
                                   )
                           ) %>% as_tibble()

# Viewing the confusion matrix
confusionMatrix(prediction_rf$predicted, prediction_rf$observed)

# Loading plotROC package (and installint it before, if not installed)
if(!require(plotROC)) {install.packages("plotROC"); library(plotROC)}
# More info about this package at:
# https://cran.r-project.org/web/packages/plotROC/vignettes/examples.html

# Basic ROC plot
basic_ROC_plot_rf <- ggplot(prediction_rf,
                            aes(d = observed,
                                m = yes)) +
                     geom_roc(n.cuts = 0) +
                     style_roc(theme = theme_grey)

# ROC plot with the AUC calculated
basic_ROC_plot_rf +
    annotate("text", x = .75, y = .25,
             label = paste("AUC =", round(calc_auc(basic_ROC_plot_rf)$AUC, 4))
    )


#_______________________________________Algorithm_02______________________________________#

# Boosted Tree
#
# Caret method: 'bstTree'
#
# Type: Classification, Regression
#
# Tags (types or relevant characteristics according to the caret package guide):
# Boosting, Ensemble Model, Tree-Based Model
#
# Tuning parameters: mstop (# Boosting Iterations), maxdepth (Max Tree Depth), nu (Shrinkage)
#
# Required packages: bst, plyr
#
# Link to know more: http://topepo.github.io/caret/


# Setting parameters that further control how the model will be created
# More at: http://topepo.github.io/caret/model-training-and-tuning.html#basic-parameter-tuning
basic_par_bstTree <- trainControl(method = "cv", # The resampling method (cross-validation)

                                  number = 5, # the number of folds in K-fold cross-validation

                                  #repeats = 3, # applied only to "repeatedcv" method

                                  search = "random", # To use a random sample of possible tuning parameter combinations

                                  returnData = F) # a logical for saving the used data into a slot called trainingData

                                  #classProbs = TRUE, # Estimate class probabilities, so would be possible to score models using the area under the ROC curve

                                  #allowParallel = T, # a logical that governs whether train should use parallel processing (if availible)
                              		# about parallel processing in windows: # http://dept.stat.lsa.umich.edu/~jerrick/courses/stat701/notes/parallel.html

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
                             predicted = predict.train(model_bstTree, 
                                                       newdata = test))
                      
# Viewing the confusion matrix
confusionMatrix(prediction_bstTree$predicted, prediction_bstTree$observed)


#_______________________________________Algorithm_03______________________________________#

# Auto ML with h2o package
# Mor info at:
# https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/R/automl_binary_classification_product_backorders.Rmd

# Loading the h2o package (and installing it, if not installed)
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
