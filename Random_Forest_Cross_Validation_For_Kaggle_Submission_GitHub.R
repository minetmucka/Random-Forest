#########################################################################################################################################
## Objective: Cross Validation using Caret Package on random forest. Save to csv for kaggle submission                                  #
## Data source: titanic data set split into train and test                                                                              #
## Please install "randomForest" package: install.packages("randomForest") for RF                                                       #
## Please install "caret" package: install.packages("caret") for K-Fold Cross Validation                                                #
## Please install "e1071" package: install.packages("e1071")                                                                            #
#########################################################################################################################################

library(caret)
library(randomForest)

# Load up data.
# NOTE - Set your working directory to the correct location for the
#        Kaggle data files.
train <- read.csv("C:/Users/muckam/Desktop/DataScienceBootcamp/Datasets/train.csv", stringsAsFactors = FALSE)
test <- read.csv("C:/Users/muckam/Desktop/DataScienceBootcamp/Datasets/test.csv", stringsAsFactors = FALSE)


# Combine the data to make data cleaning easier.
survived <- train$Survived
data.combined <- rbind(train[, -2], test)


# Transform some variables to factors.
data.combined$Pclass <- as.factor(data.combined$Pclass)
data.combined$Sex <- as.factor(data.combined$Sex)


# Split data back out.
train <- data.combined[1:891,]
train$Survived <- as.factor(survived)

test <- data.combined[892:1309,]


# Subset the features we want to use
features <- c("Survived", "Sex", "Pclass",
              "SibSp", "Parch")


# Set seed to ensure reproducibility between runs
set.seed(12345)


# Set up caret to perform 10-fold cross validation repeated 3 times
caret.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3)


# Use caret to train mighty random forests using 10-fold cross 
# validation repeated 3 times and use 7 values for tuning the
# mtry hyperparameter(i.e., the random number of variables per 
# split). This code returns the best model trained on all the 
# data using the best value of mtry! Mighty!

# NOTE - This code will take a while to run!
rf.cv <- train(Survived ~ ., 
               data = train[, features],
               method = "rf",
               trControl = caret.control,
               tuneLength = 7,
               importance = TRUE,
               ntree = 500)


# Display the results of the cross validation run - Around 78.7% 
# mean accuracy! 
rf.cv


# What is the standard deviation?
cat(paste("\nCross validation standard deviation:",  
          sd(rf.cv$resample$Accuracy), "\n", sep = " "))


# Pull out the the trained model using the best parameters on
# all the data! Mighty!
rf.best <- rf.cv$finalModel


# Look at the model - which variable are important?
varImpPlot(rf.best)


# Create predictions
preds <- predict(rf.cv, test, type = "raw")


# Create dataframe shaped for Kaggle
submission <- data.frame(PassengerId = test$PassengerId,
                         Survived = preds)


# Write out a .CSV suitable for Kaggle submission
write.csv(submission, file = "MySubmission.csv", row.names = FALSE)