# --------------------------------------------------------
# Lab 1 introduction
# --------------------------------------------------------



# --------------------------------------------------------
# Load packages
# --------------------------------------------------------

# For any package that you may not already have installed,
# use the function install.packages("package-name") to
# install it.
library(data.table)
library(glmnet)
library(ggplot2)
library(mlbench)
library(caret)
library(splines)
library(ggeffects)
# --------------------------------------------------------



# --------------------------------------------------------
# High-dimensional data example
# --------------------------------------------------------

# Objective: predict sentiment based on language used in IMDB reviews

# Import (pre-processed) IMDB data
imdb <- fread(file = 'C:/Users/eliza/Desktop/martin/git/771A43/Labs/W1/data/imdb_data.csv')

# Inspect
imdb[1:10,1:10]
dim(imdb) # High-dimensional? Yes.

# Fit ridge regression
# Assess a range of lambda values through (5-fold) cross-validation
X <- imdb[,-c('id','sentiment'),with=F] # Exclude response and id columns.
X <- as.matrix(X) # Make input data into a matrix
cvglmnet <- cv.glmnet(x = X, 
                      y = imdb$sentiment,
                      nfolds = 5,              # Number CV-folds
                      standardize = TRUE,      # Standardize X
                      family='binomial',       # Outcome binary --> logit/binomial
                      alpha=0,                 # alpha=0 --> ridge & alpha = 1 --> lasso
                      type.measure = 'class')  # measure performance in terms of accuracy

# Which lambda gave the best results?
cvglmnet # Accuracy ~ 80%
plot(cvglmnet) 

# From plot:
# i) As we increase the penalty, performance worsens. Why? Increase bias.
# ii) Note log-scale of lambda


# Why does it not "go up" to the left?
# cv.glmnet has efficient heuristic for selecting grid to search over
# let's consider a very small lambda
cvglmnet$lambda # This is the grid we considered
mylambda_grid <- c(10^-10, cvglmnet$lambda.min)
cvglmnet_extra <- cv.glmnet(x = X, 
                            y = imdb$sentiment,
                            nfolds = 10,
                            standardize = TRUE,
                            family='binomial',
                            alpha=0,
                            type.measure = 'class',
                            lambda = mylambda_grid)
plot(cvglmnet_extra)
# Here we can see that, indeed, as we make the penalty "too small" and performance worsens
# This time, because of variance.

# Let's now extract the coefficients that correspond to the lambda which gave the lowest error
best_coefs <- coef(cvglmnet, s = "lambda.min")
best_coefs_dt <- data.table(word=rownames(best_coefs),
                            coef=best_coefs[,1])
best_coefs_dt[order(coef,decreasing = T)]

# The words that are most predictive of positive sentiment are words like:
# "outstanding", "gorgeous"

# For negative, on the other hand, we find: "lame", "disappointment", and "wasted".

# Usefulness? Perhaps used for prediction on newly collected data, as a "measurement machine"?



# --------------------------------------------------------
# Non-linear example
# --------------------------------------------------------

data("Ozone")
Ozone_dt <- as.data.table(Ozone)
names(Ozone_dt) <- c("Month", "Date", "Day", "Ozone", "Press_height",
                  "Wind", "Humid", "Temp_Sand", "Temp_Monte",
                  "Inv_height", "Press_grad", "Inv_temp", "Visib")
Ozone_dt <- Ozone_dt[complete.cases(Ozone_dt)]

# Objective: predict air pollution based on measurements like
#            temperature, wind, etc. (LA, 1976 data)

# Suppose the key relation of interest: 
# pollution ("Ozone") and "Temp_Monte" (Temp Measured at El Monte)
ggplot(Ozone_dt,aes(x=Temp_Monte, y = Ozone)) + geom_point() # Linear/non-linear?

# Use package "caret" to do k-fold cross-validation
tc <- caret::trainControl(method = 'cv', number = 5)

set.seed(12345) # Set to ensure that folds are the same across models
linear_model <- caret::train(Ozone ~ Temp_Monte,
                             data = Ozone_dt,
                             method = "lm",
                             trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model2 <- caret::train(Ozone ~ ns(Temp_Monte,2),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model3 <- caret::train(Ozone ~ ns(Temp_Monte,3),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model4 <- caret::train(Ozone ~ ns(Temp_Monte,4),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model8 <- caret::train(Ozone ~ ns(Temp_Monte,8),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)

summary(caret::resamples(x = list(linear_model,
                                  gam_model2, 
                                  gam_model3,
                                  gam_model4,
                                  gam_model8)))

## ! Estimated -> careful with phrasing
## RMSE LOW GOOD, R^2 HIGH GOOD, MAE LOW GOOD

# ----- For more stability, we can do -----
tc <- caret::trainControl(method = 'repeatedcv', number = 5, repeats = 50)
## -> do this for sure -> also gives standard errors --> find out how to include


set.seed(12345) # Set to ensure that folds are the same across models
linear_model <- caret::train(Ozone ~ Temp_Monte,
                             data = Ozone_dt,
                             method = "lm",
                             trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model2 <- caret::train(Ozone ~ ns(Temp_Monte,2),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model3 <- caret::train(Ozone ~ ns(Temp_Monte,3),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model4 <- caret::train(Ozone ~ ns(Temp_Monte,4),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)
set.seed(12345) # Set to ensure that folds are the same across models
gam_model8 <- caret::train(Ozone ~ ns(Temp_Monte,8),
                           data = Ozone_dt,
                           method = "lm",
                           trControl = tc)

summary(caret::resamples(x = list(linear_model,
                                  gam_model2, 
                                  gam_model3, 
                                  gam_model4,
                                  gam_model8)))


# Suppose we settle for the GAM-spec with 2 dfs
# Let's examine marginal predictions (holding other covars constant)
# Of course, here we only have one X variable. But just for demonstration.
final_model <- lm(Ozone ~ ns(Temp_Monte,2), data=Ozone_dt)
ggpreds <- ggpredict(final_model)
plot(ggpreds)

