
#########################################################################################################################################
## Objective: Machine learning, Binary Classification Model on Adult Census Income Dataset displaying K-fold Cross-validation           #
## Data source: titanic data set split into train and test                                                                              #
## Please install "randomForest" package: install.packages("randomForest") for RF                                                       #
## Please install "ggplot2" package: install.packages("ggplot2") for Data Visualization                                                 #
## Please install "caret" package: install.packages("caret") for K-Fold Cross Validation                                                #
## Please install "rpart.plot" package: install.packages("rpart.plot") for Decision Tree Visualization                                  #
#########################################################################################################################################

# Load data and inspect.
# NOTE - Set working directory to bootcamp root folder!
adult.train <- read.csv("C:/Users/muckam/Desktop/DataScienceBootcamp/Datasets/AdultCensusIncome.csv")
str(adult.train)


# Leverage the mighty random forest to explore the data!
#install.packages("randomForest")
library(randomForest)


# Use seed so everyone sees the same thing.
set.seed(39)


# We are predicting income from all other predictors.
rf.1 <- randomForest(income ~ ., data = adult.train, 
                     importance = TRUE)
rf.1


# What features does the RF think are important?
varImpPlot(rf.1)


# Start looking at each of the important features - looking for
# clean separation of the two income classes.
#install.packages("ggplot2")
library(ggplot2)


# The capital.gain feature was the top-ranked feature by the RF.
# The density plot indicates very clean separation of the income labels.
ggplot(adult.train, aes(x = capital.gain, fill = income)) +
  theme_bw() +
  geom_density(alpha = 0.5)


# Next up, the relationship feature.
# For many levels of the relationship variable, there is very clean separation.
ggplot(adult.train, aes(x = relationship, fill = income)) +
  theme_bw() +
  geom_bar()


# And the age feature shows some nice "bump-outs".
ggplot(adult.train, aes(x = age, fill = income)) +
  theme_bw() +
  geom_density(alpha = 0.5)


# The fnlwgt doesn't seem to be as powerful as age in terms of
# "bump-outs"!
ggplot(adult.train, aes(x = fnlwgt, fill = income)) +
  theme_bw() +
  geom_density(alpha = 0.5)


# Many levels of marital.status show very strong separation of
# income levels.
ggplot(adult.train, aes(x = marital.status, fill = income)) +
  theme_bw() +
  geom_bar()


# Many levels of occupation show very strong separation of
# income levels.
ggplot(adult.train, aes(x = occupation, fill = income)) +
  theme_bw() +
  geom_bar()


# Set up caret to perform 10-fold cross validation repeated 3 times.
# Do this for better estimates of generalization error.
#install.packages("caret")
#library(e1071) 
library(caret)
caret.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3)



# Since simple models are more likely to generalize, use just the first 5
# good features we analyzed to build the model.
features <- c("capital.gain", "relationship", "age", "marital.status", "occupation", "income")
adult.train.1 <- adult.train[, features]


# Perform 10-fold CV with rpart trees and explore the effectiveness of various
# values for cp (i.e., the "tuneLength" parameter below). Using rpart here so
# that we can train fast when compared to doing CV with Random Forests. Use a 
# random seed to ensure we all see the same thing.

set.seed(87)
rpart.cv.1 <- train(income ~ ., 
                    data = adult.train.1,
                    method = "rpart",
                    trControl = caret.control,
                    tuneLength = 7)
# Other algorithms to be used as "methods" in caret.

# Display the results of the cross validation run
rpart.cv.1


# What is the standard deviation? 0.4837%! Nice!
cat(paste("\nCross validation standard deviation:",  
          sd(rpart.cv.1$resample$Accuracy), "\n", sep = " "))


# Take a look at the model
#install.packages("rpart.plot")
library(rpart.plot)
prp(rpart.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

