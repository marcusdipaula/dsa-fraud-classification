# ---
# script: 03-DataWrangling
# subject: Exploring and Feature Engineering
# date: 2019-07-12
# author: Marcus Di Paula
# github: github.com/marcusdipaula/
# linkedin: linkedin.com/in/marcusdipaula/
# ---



# 3. Data preparation and Exploration (Feature Engineering oriented to the 4th phase)
#     - Is my dataset tidy?
#     - Is my dataset clean?
#     - Which correlations exists between all variables and to the target?
#     - There is any NA in my dataset? If so, how should I treat them? Which effects would it have?
#     - Should I narrowing in on observations of interest? Which effects would it have?
#     - Should I reduce my variables? Which effects would it have?
#     - Should I create new variables that are functions of existing ones? Which effects would it have?
#     - Should I binning variables? Which effects would it have?
#     - should I convert variables (categorical = numerical / vv)? Which effects would it have?
#     - Should I dummy coding categorical variables? Which effects would it have?
#     - Should I standardize numerical variables? Which effects would it have?
#     - Can I test my hypotheses?


# if you want to explore use TRUE
want_to_explore <- FALSE

if(want_to_explore) {

          # How many NA's are in the dataset?
          sapply(ds, function(x) sum(is.na(x)))
          
          # Exploring
          str(ds)
          head(ds)
          tail(ds)
          summary(ds)
          cor(x = ds[, sapply(ds, is.integer)] %>% 
                mutate(target = as.integer(ds$is_attributed)) 
              )
          
          # Loading the GGally package (and installing it before, if not installed)
          if(!require(GGally)) {install.packages("GGally"); library(GGally)}
          
          # Ploting a correlation heat map
          ds %>% 
            mutate(day = lubridate::day(ds$click_time),
                   hour = lubridate::hour(ds$click_time),
                   minute = lubridate::minute(ds$click_time),
                   target = as.integer(ds$is_attributed) ) %>% 
            ggcorr() # The non-numeric data will be dropped with a warning
          
          # Boxplot
          ds %>% 
            ggplot(aes(y = lubridate::hour(ds$click_time), 
                       x = is_attributed,
                       fill = is_attributed)) +
            geom_boxplot() +
            labs(x = "Dowloaded (1) or not (0)",
                 y = "Hour of a day",
                 fill = "Color legend")
          
          # Matrix of plots of numerical and categorical variables
          ds[,sapply(ds, is.numeric)] %>% 
                mutate(is_attributed = ds$is_attributed) %>% 
                  na.omit() %>% 
                    ggpairs()
          
          }

# Creating 2 new variables (that expressed strong correlation to the
# target variable), removing some other and dropping NA rows
ds <- ds %>% 
  mutate(day = lubridate::day(ds$click_time),
         hour = lubridate::hour(ds$click_time) ) %>% 
         select(-click_time, -attributed_time) %>% 
         drop_na() ; rm(want_to_explore)

# To avoid problems with valid R variable names (see ?make.names for more info),
# we need to recode the factor classes
ds$is_attributed <- recode_factor(ds$is_attributed, "1" = "yes", "0" = "no")

