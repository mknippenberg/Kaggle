#### Load Train & Val Data ####
data = read.csv("train AND validation.csv")

## First randomly sample 0 or 1 for 32769 observations ##
# 70% train set, 30% validation set
sample = sample(0:1, 32769, replace = TRUE, prob = c(0.3, 0.7))

## Add sample to TRAIN column of data ##
data$TRAIN = sample 

## Randomly split into training and validation data sets ##
## 1 will be training set, 0 will be validation set ##
train = data[data$TRAIN == 1,]
validation = data[data$TRAIN ==0,]

## Strip the TRAIN column as it is constant and throws off some models ##
train = train[,1:10]
validation = validation[,1:10]

write.table(train, file = "train.csv", sep = ",", row.names = F)
write.table(validation, file = "validation.csv", sep = ",", row.names = F)

## Convert ACTION from int to factor to be used in classification ##
train$ACTION = as.factor(train$ACTION)
validation$ACTION = as.factor(validation$ACTION)

#### Generate Model ####

lda = lda(train$ACTION ~., train)
summary(lda)

predictions = predict(lda, train, type="response")

#### Tree ####
# This is no good, there is just 1 endpoint, same as always predicting ACTION =1
library(tree)
tree1 <- tree(ACTION ~ ., data=train)
summary(tree1)

# Use Trees to predict on training data
pred1 <- predict(tree1, train, type="class")
pred1

pred2 <- predict(tree1, validation, type="class")


#### Measure Error ####
# Find Misclassification error rate
error.rate <- 1- (sum(train$ACTION == pred1)/ nrow(train))
error.rate

error.rate <- 1- (sum(validation$ACTION == 1)/ nrow(validation))
error.rate


## Calculate area under ROC curve, need verification package ##
library("verification")
auc = roc.area(train$ACTION, predictions)
auc2 = roc.area(train$ACTION, predictions2)
