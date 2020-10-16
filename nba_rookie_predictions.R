library(tidyverse) #For data manipulation and plotting
library(GGally) #Used for correlation charts
library(glmnet) #Used to create models
library(caret) #Model evaluation and parameter tuning
library(pROC) #ROC measurement
library(e1071) #Used to create the table for confusion matrix

#Reading in data

setwd("C:/Users/rober/OneDrive/MSc/DMML2/Project")

data <- read.csv("student_2431907data.csv",row.names = "X")

######################
#Exploratory Analysis#
######################

#Variance over time

#Understanding changes over time
year_count <- data %>%
  group_by(Year_drafted) %>%
  summarise(records = n()
            ,avg_years = mean(Yrs)
            ,over_5 = sum(Target)
            ,min_yrs = min(Yrs)
            ,max_yrs = max(Yrs)) %>%
  mutate(over_5_pct = over_5 / records)


##Records
b <- ggplot(year_count,aes(Year_drafted,records))
b <- b + geom_bar(stat="identity")
b

l <- ggplot(year_count,aes(Year_drafted,records))
l <- l + geom_line()
l <- l + expand_limits(y = 0)
l <- l + ggtitle("Number of records by year drafted")
l <- l + xlab("Year drafted")
l <- l + ylab("Number of records")
l

##Average years
l <- ggplot(year_count,aes(Year_drafted,avg_years))
l <- l + geom_line()
l <- l + expand_limits(y = 0)
l

b <- ggplot(data,aes(as.factor(Year_drafted),Yrs))
b <- b + geom_boxplot()
b

h <- ggplot(data,aes(Yrs))
h <- h + geom_density()
h

##Max years
l <- ggplot(year_count,aes(Year_drafted,max_yrs))
l <- l + geom_line()
l <- l + expand_limits(y = 0)
l <- l + ggtitle("Maximum career length by year drafted")
l <- l + xlab("Year drafted")
l <- l + ylab("Max career length (years)")
l

##Over 5 years
l <- ggplot(year_count,aes(Year_drafted,over_5_pct))
l <- l + geom_line()
l <- l + expand_limits(y = c(0,1))
l <- l + ggtitle("Proportion of rookies playing over 5 years by year drafted")
l <- l + xlab("Year drafted")
l <- l + ylab("Proportion of rookies playing over 5 years")
l

#Partitioning data
sample_size <- floor(0.7 * nrow(data))
training_index <- sample(seq_len(nrow(data)), size = sample_size)
data_train <- data[training_index, ]
data_test <- data[-training_index, ]

#Checks
data_train %>%
  summarise(Total = n(),Target = sum(Target),Target_pct = sum(Target) / n())

data_test %>%
  summarise(Total = n(),Target = sum(Target),Target_pct = sum(Target) / n())

#Optional: writing out files to have a consistent record if session lost
#write.csv(data_train,"data_train.csv")
#write.csv(data_test,"data_test.csv")

#Optional: reading in features
data_train <- read.csv("data_train.csv",row.names="X")
data_test <- read.csv("data_test.csv",row.names="X")

#Understanding features

summary(data)

data_train_features <- data_train %>%
  select(GP,MIN,PTS,FG_made,FGA,FG_percent,TP_made,TPA,TP_percent,FT_made,FTA,FT_percent
         ,OREB,DREB,REB,AST,STL,BLK,TOV)
  
data_train_features %>%
  gather() %>%
  ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_density() +
    ggtitle("Distribution of candidate features across the training data")

M <- cor(data_train_features);
ggcorr(M,label=TRUE, color = "grey50",label_size = 3,label_round = 2,hjust = 1,layout.exp=1)

#Transforming data
data_train_transformed <- data_train %>%
  mutate(PTS_minute = PTS / MIN
         ,FGA_minute = FGA / MIN
         ,TPA_minute = TPA / MIN
         ,FTA_minute = FTA / MIN
         ,REB_minute = REB / MIN
         ,AST_minute = AST / MIN
         ,STL_minute = STL / MIN
         ,BLK_minute = BLK / MIN
         ,TOV_minute = TOV / MIN
  )

data_test_transformed <- data_test %>%
  mutate(PTS_minute = PTS / MIN
         ,FGA_minute = FGA / MIN
         ,TPA_minute = TPA / MIN
         ,FTA_minute = FTA / MIN
         ,REB_minute = REB / MIN
         ,AST_minute = AST / MIN
         ,STL_minute = STL / MIN
         ,BLK_minute = BLK / MIN
         ,TOV_minute = TOV / MIN
  )


#Selecting and scaling variables
data_train_transformed_selected <- data_train_transformed %>%
  select(GP,MIN,PTS_minute,FGA_minute,FG_percent
         ,TPA_minute,TP_percent,FTA_minute,FT_percent
         ,REB_minute,AST_minute,STL_minute,BLK_minute
         ,TOV_minute,Target)
  
data_test_transformed_selected <- data_test_transformed %>%
  select(GP,MIN,PTS_minute,FGA_minute,FG_percent
         ,TPA_minute,TP_percent,FTA_minute,FT_percent
         ,REB_minute,AST_minute,STL_minute,BLK_minute
         ,TOV_minute,Target)

#Scaling variables
data_train_scaled <- data_train_transformed_selected
data_test_scaled <- data_test_transformed_selected

data_train_scaled[,1:14] <- apply(data_train_scaled[,1:14], 2, scale)
data_test_scaled[,1:14] <- apply(data_test_scaled[,1:14], 2, scale)

##Checking scaling
summary(data_train_scaled)
summary(data_test_scaled)  

###########
#Modelling#
###########

set.seed(42)

#Splitting x and y and creating a model matrix
x <- model.matrix(Target~.,data=data_train_scaled)
y <- data_train_scaled$Target

#Creating model matrix for test
x.test <- model.matrix(Target ~., data_test_scaled)


###
#Ridge regression
###

fit.ridge=glmnet(x,y,family="binomial",alpha=0)

#Checking coefficients over different values of log lambda
plot(fit.ridge,xvar="lambda",label=TRUE)

#Identifying binomial deviance over different values of log lambda
cv.ridge <- cv.glmnet(x,y,family="binomial",alpha=0)
plot(cv.ridge)

#Checking lambda parameters
cv.ridge$lambda.min
cv.ridge$lambda.1se

#Fitting model using lambda at 1se
fit.ridge.lambda.1se <- glmnet(x, y, family="binomial",alpha = 0, lambda=cv.ridge$lambda.1se)

#Looking at coefficients to identify the most important variables
coef(fit.ridge.lambda.1se)

#Creating model matrix for test
x.test.ridge <- model.matrix(Target ~., data_test_scaled)

#Predict on test data
predictions.ridge <- fit.ridge.lambda.1se %>% predict(x.test) %>% as.vector()

#Plot ROC
plot.roc(data_test_scaled$Target,
         predictions.ridge)

#Calculate AUC
auc(data_test_scaled$Target,
    predictions.ridge)

#Classifying predictions
predictions.ridge.class <- ifelse(predictions.ridge > 0.5,1,0) %>% as.integer()

#Confusion matrix
confusionMatrix(table(predictions.ridge.class,data_test_scaled$Target))

###
#Lasso regression
###

#Checking number of variables and coefficients by different values of log lambda
fit.lasso=glmnet(x,y,family="binomial",alpha=1)
plot(fit.lasso,xvar="lambda",label=TRUE)

#Checking coefficients by deviance explained (R2)
plot(fit.lasso,xvar="dev",label=TRUE)

#Identifying binomial deviance over different values of log lambda
cv.lasso <- cv.glmnet(x, y,family="binomial", alpha = 1)
plot(cv.lasso)

#Checking lambda parameters
cv.lasso$lambda.min
cv.lasso$lambda.1se

#Fitting model using lambda at 1se
fit.lasso.lambda.1se <- glmnet(x, y, family="binomial",alpha = 1, lambda=cv.lasso$lambda.1se)

#Looking at coefficients to identify the most important variables
coef(fit.lasso.lambda.1se)

#Predict on test data
predictions.lasso <- fit.lasso.lambda.1se %>% predict(x.test) %>% as.vector()

#Plot ROC
plot.roc(data_test_scaled$Target,
         predictions.lasso)

#Calculate AUC
auc(data_test_scaled$Target,
    predictions.lasso)

#Classifying predictions
predictions.lasso.class <- ifelse(predictions.lasso > 0.5,1,0) %>% as.integer()

#Confusion matrix
confusionMatrix(table(predictions.lasso.class,data_test_scaled$Target))

###
#Elastic net
###

#6 fold cross validation
model.net <- train(
  as.factor(Target) ~., data = data_train_scaled, method = "glmnet", family="binomial"
  ,trControl = trainControl("cv", number = 6),
  tuneLength = 100) #10 different values to try for each parameter

#Best parameter values
model.net$bestTune

#Looking at coefficients to identify the most important variables
coef(model.net$finalModel, model.net$bestTune$lambda)

#Predict on test data
predictions.net <- model.net %>% predict(x.test, type = "prob")
predictions.net <- predictions.net$"1" %>% as.vector()

#Plot ROC
plot.roc(data_test_scaled$Target,
         predictions.net)

#Calculate AUC
auc(data_test_scaled$Target,
    predictions.net)

#Classifying predictions
predictions.net.class <- ifelse(predictions.net > 0.5,1,0) %>% as.integer()

