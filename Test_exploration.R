library(mltools)
library(caTools)
library(DMwR)
library(data.table)
claim <- read.csv("claims.csv", header = TRUE)

#Feature Engineered 
claim_sub <- subset(claim, select = -c(PolicyNumber,RepNumber,WeekOfMonth,DayOfWeek,DayOfWeekClaimed,WeekOfMonthClaimed,
                                       MaritalStatus,DriverRating,Days_Policy_Claim,PoliceReportFiled,WitnessPresent,
                                       NumberOfCars,Month,MonthClaimed,Year))

#Split the dataset
set.seed(123)
split = sample.split(claim_sub$FraudFound_P, SplitRatio = 0.75)
claim1_train = subset(claim_sub, split == TRUE)
claim1_test = subset(claim_sub, split == FALSE)


#OHE for test set
claim1_test <- one_hot(as.data.table(claim1_test))

#Sample the train dataset

claim1_train$FraudFound_P <- as.factor(claim1_train$FraudFound_P)
table(claim1_train$FraudFound_P)
??SMOTE

claim_Smote <- SMOTE(FraudFound_P ~ ., claim1_train, perc.over = 600,perc.under=118)

table(claim_Smote$FraudFound_P)

#One hot encoding
claim_Smote$FraudFound_P <- as.integer(claim_Smote$FraudFound_P)
claim_cat <- one_hot(as.data.table(claim_Smote))
claim_cat <- as.data.frame(claim_cat)
table(claim_cat$FraudFound_P)



# Encoding the target feature as factor only after one-hot encoding in
claim_cat$FraudFound_P <- as.factor(claim_cat$FraudFound_P)
typeof(claim_cat$FraudFound_P)
levels(claim_cat$FraudFound_P) <- c('0','1')
levels(claim_cat$FraudFound_P)

#######################################################MachineShop##########################################################
# Current release from CRAN
install.packages("MachineShop")

library(MachineShop)

RShowDoc("Introduction", package = "MachineShop")

library(doParallel)
registerDoParallel(cores = 4)

## Model formula
fo <- FraudFound_P ~ .


## Gradient boosted mode fit to training set
gbmfit <- fit(fo, data = claim_cat, model = GBMModel)

(vi <- varimp(gbmfit))

plot(vi)

## Test set predicted probabilities
library(dplyr)
## Test set predicted classifications
predict(gbmfit, newdata = claim1_test) %>% head

## Test set performance
obs <- response(fo, data = test)
pred <- predict(gbmfit, newdata = test, type = "prob")
modelmetrics(obs, pred)

## Resample estimation of model performance
(perf <- resample(fo, data = claim_cat, model = GBMModel, control = CVControl))

summary(perf)

plot(perf)

## Tune over a grid of model parameters
gbmtune <- tune(fo, data = iris, model = GBMModel,
                grid = expand.grid(n.trees = c(25, 50, 100),
                                   interaction.depth = 1:3,
                                   n.minobsinnode = c(5, 10)))

plot(gbmtune, type = "line")

## Fit the selected model
gbmtunefit <- fit(fo, data = iris, model = gbmtune)
varimp(gbmtunefit)

## Model comparisons
control <- CVControl(folds = 10, repeats = 5)

gbmperf <- resample(fo, data = claim_cat, model = GBMModel(n.tree = 50), control = control)
rfperf <- resample(fo, data = claim_cat, model = RandomForestModel(ntree = 50), control = control)
nnetperf <- resample(fo, data = claim_cat, model = NNetModel(size = 5), control = control)

perf <- Resamples(GBM = gbmperf, RF = rfperf, NNet = nnetperf)
summary(perf)
