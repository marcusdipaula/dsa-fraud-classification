# ---
# script: 02-DataCollect
# subject: Looking for data
# date: 2019-07-11
# author: Marcus Di Paula
# github: github.com/marcusdipaula/
# linkedin: linkedin.com/in/marcusdipaula/
# ---



# 2. Looking for data: 
#     - Identify entities (and its attributes) of the problem 
#       user
#         |_ip (ip address of user)
#         |_device (device type id of user mobile phone)
#         |_os (os version id of user mobile phone)
#
#       event
#         |_click_time (timestamp of click, UTC)
#         |_is_attributed (downloaded an app = 1, didn't download an app = 0)
#         |_attributed_time (if download an app, the time of it)
#
#       ad publisher
#         |_app (app id for marketing)
#         |_channel (channel id of mobile ad publisher)
#       
#     - Collect data that represents entities
#       https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
#       
#     - Which hypotheses could I suppose?
#       There may be a correlation between the quantity of an ip click_time and the is_attributed result.
#       There may be a correlation between the channel and the is_attributed result.
#       
#       
#     - Explore the data (superficially) to understand it 
#       
#       
#     - Could I use an algorithm to address the issue or solve it? Which one?
#       I plan to use 3 algorithms:
#           randomForest (from caret package)
#           Boosted Tree (from caret package)
#           autoML (from h2o package)



# My hardware has a limited amount of RAM memory, so I cannot load the entire train dataset.
# Since the train_sample.csv has an imbalanced target variable (with approximately 0.002 of 
# is_attributed = 1), I assume the train.csv has approximately the same proportion.
# I load the train.csv into a sqlite database so I would be able to query the is_attributed = 1
# from it. Then I'll query a random sample of the same size, but with is_attributed = 0.

# For the first time runing this script, use TRUE, for a second time, use FALSE.
# The first time will create and save a filtered dataset. The second will load the 
# filtered saved datased.
first_time <- FALSE

#___________________________________________________________________________________________________#
#                                                                                                   #
#                                                                                                   #
# REMEMBER: set the working directory and download there all the necessary data from:               #
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data                              #
#                                                                                                   #
# Load the train.csv into a train.sqlite db. I suggest the SQLite Studio:                           #
# https://sqlitestudio.pl/index.rvt                                                                 #
#                                                                                                   #
# or use the the pre filtered dataset (setting the first_time object above to FALSE)                #
# https://github.com/marcusdipaula/dsa-fraud-classification/blob/master/ds.csv                      #
#                                                                                                   #
#                                                                                                   #
#___________________________________________________________________________________________________#


if(first_time){
          
          # Loading the sqldf package (and installing it before, if not installed)
          if(!require(sqldf)) {install.packages("sqldf"); library(sqldf)}
  
          # Loading the data.table package to work faster with heavy datasets
          if(!require(data.table)) {install.packages("data.table"); library(data.table)}
          # more information at:
          # https://rpubs.com/msundar/large_data_analysis
          # https://github.com/Rdatatable/data.table/wiki
          
          # Creating the connection to the sqlite
          con <- dbConnect(SQLite(), dbname = "train.sqlite")
          
          # Getting all the is_attributed = 1
          dt_attributed <- data.table(dbGetQuery(conn = con, 
                                                 statement = "SELECT * 
                                                              FROM train 
                                                              WHERE is_attributed = '1' ") 
                                      )
          
          # Getting a sample of is_attributed = 0 with the same length as the previous object
          # so we could have a dataset with a balanced target variable
          
          # Getting 10x more the number of is_attributed = 1 row (to get a random sample of it)
          dt_not_attributed <- data.table(dbGetQuery(conn = con, 
                                                     statement = paste("SELECT * 
                                                                       FROM train 
                                                                       WHERE is_attributed = '0' 
                                                                       LIMIT ",(10*nrow(dt_attributed)) 
                                                                       )
                                                     ), colClasses = list(numeric=c(1:5,8))
                                          )
          
          # Getting a random sample of the same size of dt_attributed object
          dt_not_attributed <- dt_not_attributed[sample(.N, nrow(dt_attributed))]
          
          # Saving objects as csv
          fwrite(dt_attributed, "dt_attributed.csv")
          fwrite(dt_not_attributed, "dt_not_attributed.csv")
          
          # Combining datasets by row (and removing objects that won't be necessary)
          dt <- rbind(dt_attributed, dt_not_attributed); rm(dt_attributed, dt_not_attributed) 
          
          # Saving the ds object as a csv, removing all objects and disconnecting from the sqlite db
          fwrite(dt, "ds.csv"); dbDisconnect(con); rm(first_time, con, dt)  
          
          # unloading packages not needed anymore
          detach("package:sqldf", unload = TRUE)
          detach("package:data.table", unload = TRUE)
          
          # Loading the tidyverse packages (and installing them before, if not installed)
          if(!require(tidyverse)) {install.packages("tidyverse"); library(tidyverse)}
          
          # Loading the dataset with read_csv (instead of converting to a tibble, so the size
          # of the object will be smaller)
          ds <- read_csv("ds.csv", 
                          col_types = cols(.default = "i",
                                           click_time = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
                                           attributed_time = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
                                           is_attributed = "f")
                         ) 
          } else {
            
            # loading the tidyverse packages
            if(!require(tidyverse)) {install.packages("tidyverse"); library(tidyverse)} 
            
            # if first_time = FALSE, then load the saved dataset
            ds <- read_csv("ds.csv", 
                            col_types = cols(.default = "i",
                                             click_time = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
                                             attributed_time = col_datetime(format = "%Y-%m-%d %H:%M:%S"),
                                             is_attributed = "f") 
                          ); rm(first_time)
                  }

