# --------------------------------------------------------
# Lab 2 introduction
# --------------------------------------------------------


# --------------------------------------------------------
# Load packages
# --------------------------------------------------------
library(data.table)
library(ggplot2)
library(mlbench)
library(caret)
library(glmnet)
library(randomForest) # New (random forest)
library(rpart.plot)   # New (to plot decision trees)
library(gbm)          # New (boosting)
# --------------------------------------------------------


# --------------------------------------------------------
# Fit a decision tree to the Ozone data
# --------------------------------------------------------

# Load and preprocess as last week
data("Ozone")
Ozone_dt <- as.data.table(Ozone)
names(Ozone_dt) <- c("Month", "Date", "Day", "Ozone", "Press_height",
                     "Wind", "Humid", "Temp_Sand", "Temp_Monte",
                     "Inv_height", "Press_grad", "Inv_temp", "Visib")
Ozone_dt <- Ozone_dt[complete.cases(Ozone_dt)]
Ozone_dt[,c('Month','Date','Day') := NULL] # Let's remove id variables

# Objective: predict air pollution based on measurements like
#            temperature, wind, etc. (LA, 1976 data)

# Use package "caret" to do k-fold cross-validation
tc <- caret::trainControl(method = 'cv', number = 10)

# To fit a decision tree with the caret package, we simply replace the methods argument with "rpart".
# --- Note the addition of the "cp" parameter in tuneGrid. This is the penalty (ISL: alpha)
set.seed(12345)
rpart_model <- caret::train(Ozone ~ .,
                            data = Ozone_dt,
                            tuneGrid=expand.grid(cp=c(0,0.001,0.01,0.05,0.1)),
                            method = "rpart",
                            trControl = tc)

# We inspect the results by printing the object
rpart_model # In this case, pruning (that is cp > 0, did not help improve performance much)

# We can extract the results table (to e.g., enable plotting) as follows
rpart_model$results
ggplot(rpart_model$results,aes(x=cp,y=RMSE)) + geom_point()
ggplot(rpart_model$results,aes(x=cp,y=Rsquared)) + geom_point()

# We can extract the "best model" using $finalModel (the cp parameter having the lowest error)
best_decision_tree <- rpart_model$finalModel

# And we can plot its tree structure using rpart.plot
rpart.plot(best_decision_tree)

# To calculate variable importance, we may use the command:
caret::varImp(best_decision_tree)

# To plot it, we can do as follows:
varimp_object <- caret::varImp(best_decision_tree)
varimp_dt <- data.table(var=rownames(varimp_object),
                        imp=varimp_object$Overall)
varimp_dt <- varimp_dt[order(imp,decreasing = T)]
varimp_dt[,imp := imp / max(imp)] # Making variable importance into a relative measure
ggplot(varimp_dt,aes(y=reorder(var, imp),x=imp)) + geom_point(size=3)


# Note: if we allow our tree to grow deeper (by lowering the minimum number of obs in a leaf,
# then pruning helps!)
set.seed(12345)
rpart_model2 <- caret::train(Ozone ~ .,
                            data = Ozone_dt,
                            tuneGrid=expand.grid(cp=c(0,0.001,0.01,0.05,0.1)),
                            control = rpart.control(minbucket=2),
                            method = "rpart",
                            trControl = tc)
rpart_model2


# --------------------------------------------------------
# Let's now try a random forest on the same data
# --------------------------------------------------------

# To fit a random forest using the caret package, we simply set method="rf"
# --- Note that now we have a parameter "mtry" instead of "cp".
# --- mtry = no. columns to evaluate for each split
mtry_values <- seq(1,ncol(Ozone_dt),2)
set.seed(12345)
rf_model <- caret::train(Ozone ~ .,
                         data = Ozone_dt,
                         tuneGrid=expand.grid(mtry=mtry_values),  
                         method = "rf",
                         trControl = tc)
# Again, inspect
rf_model

# Notes:
# - Substantial improvement compared to standard decision tree (Rsquared: 0.77 > 0.67)
# - Does "decorrelating" the trees help? Yes! RMSE(mtry=1) < RMSE(mtry=9). When mtry=ncol(dt) --> bagging.

# Again, we can extract best model
best_rf_model <- rf_model$finalModel # As we see here, we used 500 trees. We can change the number of trees with the "ntrees" argument

# And again, we can extract variable importance:
caret::varImp(best_rf_model)
varimp_rf_object <- caret::varImp(best_rf_model)
varimp_rf_dt <- data.table(var=rownames(varimp_rf_object),
                           imp=varimp_rf_object$Overall)
varimp_rf_dt <- varimp_rf_dt[order(imp,decreasing = T)]
varimp_rf_dt[,imp := imp / max(imp)]
varimp_rf_dt

# Compare variable importance
# -- Plot them together (rpart, rf)
varimp_dt$model <- 'rpart'
varimp_rf_dt$model <- 'rf'
varimp_combined <- rbind(varimp_dt, varimp_rf_dt)
order_by_rf <- varimp_rf_dt$var
varimp_combined$var <- factor(varimp_combined$var,levels = c(order_by_rf))
ggplot(varimp_combined,aes(y=var, imp),x=imp) + geom_point(size=3) + facet_wrap(~model)

# Example of a meaningful difference: "Press_grad" much more important for the random forest

# Let's produce a partial depenency plot for this variable
randomForest::partialPlot(best_rf_model, pred.data=Ozone_dt, x.var='Press_grad')

# Very non-linear pattern indeed.

# Let's also plot for the most important variable
randomForest::partialPlot(best_rf_model, pred.data=Ozone_dt, x.var='Temp_Monte')


# --------------------------------------------------------
# What about KNN? Again, for the same data, here is how you
# would fit a kNN model, using cross-validation to select 
# the appropriate k
# --------------------------------------------------------
set.seed(12345)
knn_model <- caret::train(Ozone ~ .,
                          data = Ozone_dt,
                          tuneGrid=c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50),
                          method = "knn",
                          trControl = tc,)
knn_model
ggplot(knn_model$results,aes(x=k,y=RMSE)) + geom_line()
ggplot(knn_model$results,aes(x=k,y=Rsquared)) + geom_line()


# --------------------------------------------------------
# We will not use boosting in the assignment, but you
# implement in as above, just replace method="gbm"
# --------------------------------------------------------

# Boosting, as we talked about in the lecture, has a few more parameters that
# should be evaluated
mygrid <- expand.grid(interaction.depth=c(1,2,3,5),          # Depth of each tree (ISL: d)
                      n.trees=c(10,100,500,1000,3000,6000),  # Number of trees
                      shrinkage=c(0.01,0.001),               # How slow we want to learn
                      n.minobsinnode=c(10))                  # Min. number of observations in a node

set.seed(12345)
system.time(boost_model <- caret::train(Ozone ~ .,
                                       data = Ozone_dt,
                                       tuneGrid=mygrid,
                                       method = "gbm",
                                       trControl = tc))
# Again, inspect
boost_model

# Extract best model
best_boost_model <- boost_model$finalModel

# Variable importance
caret::varImp(best_boost_model)
varimp_boost_object <- caret::varImp(best_boost_model)
varimp_boost_dt <- data.table(var=rownames(varimp_boost_object),
                           imp=varimp_boost_object$Overall)
varimp_boost_dt <- varimp_boost_dt[order(imp,decreasing = T)]
varimp_boost_dt[,imp := imp / max(imp)]
varimp_boost_dt

