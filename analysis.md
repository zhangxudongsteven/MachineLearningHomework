# Machine Learning Homework
Steven Zhang  
Nov 28, 2016  



## 1 - Synopsis

This analysis aims to predict the activity from device data. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The training data can be download [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv), which has been downloaded to local disk for convience.

## 2 - Data Preprocessing

First, let's load the data and check the dimensions and missing value. 


```r
df.training <- read.csv("data/pml-training.csv")
df.final <- read.csv("data/pml-testing.csv")
dim(df.training)
```

```
## [1] 19622   160
```

```r
dim(df.final)
```

```
## [1]  20 160
```

```r
summary(as.factor(apply(df.final, 2, function(x) sum(is.na(x)))))
```

```
##   0  20 
##  60 100
```

From the analysis above, we can see the testing dataset have 100 complete NA columns, and 60 columns without missing value. In this case, I will remove these column from both dataset. Then, according to the demand in real situations, we should not judge activity type from some variables such as user name and time, even it might be quiet useful. So I will remove the first 7 columns.


```r
temp <- data.frame(id = 1:160, missing = apply(df.final, 2, function(x) sum(is.na(x))))
temp <- filter(temp, missing == 0)
df.training <- select(df.training, temp$id)
df.final <- select(df.final, temp$id)
df.training <- select(df.training, 8:60)
df.final <- select(df.final, 8:60)
```

Then, Let's check the variability and remove the zero-variance covariates.


```r
nsv <- nearZeroVar(df.training, saveMetrics = TRUE)
zeroVarSum <- sum(nsv$zeroVar)
nzvSum <- sum(nsv$nzv)
```

There are **0** zero variance and **0** near-zero-variance. Seems good. 

Next, we need to analysis the multi-collinearity.


```r
cor.train <- cor(df.training[,-53])
kappa.train <- kappa(cor.train, exact = TRUE)
```

The Kappa of original train set is **3888.6681733**, which is too high (>1000). we need to perform PCA. During the PCA process, I will keep **90 cumulative percent** of variance.


```r
preProc <- preProcess(df.training[,-53], method = c("center", "scale", "pca"), thresh = 0.9)
source.data <- predict(preProc, df.training[,-53])
final <- predict(preProc, df.final[,-53])
source.data$classe <- df.training$classe
final$problem_id <- df.final$problem_id
```

Then, the data can be used in the next step. To be attenion, the last column of training set is the class type, but it's the id for final testing set's. Finally, I will split the data into training and testing.


```r
inTrain <- createDataPartition(y=source.data$classe, p = 0.7, list = FALSE)
training <- source.data[inTrain,]
testing <- source.data[-inTrain,]
# detach it to avoid masked
detach("package:dplyr", unload=TRUE)
```

## 3 - Exploratory Analysis

Because we have done a PCA transformation, so the comprehensibility will be influenced. 

## 4 - Classification

In this part, we will use three algorithms to perform the prediction, and use random forest to combine them. It will take a lot of time to train the model, so I have trained them and saved locally. You can use code in the annotation to redo the training.

### 4.1 - SVM

First, I will train a svm model and check it's efficiency. 


```r
library(e1071)
svm.fit <- svm(classe ~ ., data = training)
pred <- predict(svm.fit, newdata = testing)
confusionMatrix(testing$classe, pred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1641   13   14    5    1
##          B   90  966   80    3    0
##          C   10   49  934   23   10
##          D    7    9  116  828    4
##          E    5   10   40   30  997
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9118          
##                  95% CI : (0.9043, 0.9189)
##     No Information Rate : 0.2979          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8883          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9361   0.9226   0.7889   0.9314   0.9852
## Specificity            0.9920   0.9642   0.9804   0.9728   0.9826
## Pos Pred Value         0.9803   0.8481   0.9103   0.8589   0.9214
## Neg Pred Value         0.9734   0.9829   0.9485   0.9876   0.9969
## Prevalence             0.2979   0.1779   0.2012   0.1511   0.1720
## Detection Rate         0.2788   0.1641   0.1587   0.1407   0.1694
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9641   0.9434   0.8846   0.9521   0.9839
```

The overall accuracy is **over 90%**, which is very ideal.

### 4.2 - Random Forest

Then, I will try random forest and check it's efficiency. 


```r
library(randomForest)
rf.fit <- randomForest(classe ~ ., data = training)
pred <- predict(rf.fit, testing, type = "response")
confusionMatrix(testing$classe, pred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1656    6    7    5    0
##          B   25 1099   14    0    1
##          C    2   15 1001    7    1
##          D    1    0   40  922    1
##          E    0    2    7    0 1073
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9772          
##                  95% CI : (0.9731, 0.9809)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9712          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9834   0.9795   0.9364   0.9872   0.9972
## Specificity            0.9957   0.9916   0.9948   0.9915   0.9981
## Pos Pred Value         0.9892   0.9649   0.9756   0.9564   0.9917
## Neg Pred Value         0.9934   0.9952   0.9860   0.9976   0.9994
## Prevalence             0.2862   0.1907   0.1816   0.1587   0.1828
## Detection Rate         0.2814   0.1867   0.1701   0.1567   0.1823
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9895   0.9856   0.9656   0.9893   0.9977
```

The overall accuracy is **over 95%**, which is very ideal.

### 4.3 - Neural Networks

Then, I will try Neural Networks and check it's efficiency.


```r
library(nnet)
nnet.fit <- nnet(classe ~ ., data = training, size = 5)
```

```
## # weights:  130
## initial  value 25755.194006 
## iter  10 value 20193.772304
## iter  20 value 17508.505843
## iter  30 value 16724.903304
## iter  40 value 16297.789855
## iter  50 value 16010.960950
## iter  60 value 15675.447406
## iter  70 value 15441.032523
## iter  80 value 15194.662971
## iter  90 value 15037.170825
## iter 100 value 14906.769375
## final  value 14906.769375 
## stopped after 100 iterations
```

```r
pred <- predict(nnet.fit, testing, type = "class")
confusionMatrix(testing$classe, pred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1121  165  132  240   16
##          B  179  589  145  174   52
##          C  105  367  501   25   28
##          D  126  288   94  365   91
##          E   49  387   70   64  512
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5247          
##                  95% CI : (0.5119, 0.5376)
##     No Information Rate : 0.3052          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3989          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7095   0.3280  0.53185  0.42051   0.7325
## Specificity            0.8715   0.8655  0.89379  0.88061   0.8901
## Pos Pred Value         0.6697   0.5171  0.48830  0.37863   0.4732
## Neg Pred Value         0.8910   0.7457  0.90924  0.89779   0.9611
## Prevalence             0.2685   0.3052  0.16007  0.14749   0.1188
## Detection Rate         0.1905   0.1001  0.08513  0.06202   0.0870
## Detection Prevalence   0.2845   0.1935  0.17434  0.16381   0.1839
## Balanced Accuracy      0.7905   0.5967  0.71282  0.65056   0.8113
```

```r
# library(neuralnet)
# nn.name <- names(training)
# nn.f <- as.formula(paste("classe ~", paste(nn.name[!nn.name %in% "classe"], collapse = " + ")))
# temp <- training
# temp$classe <- as.integer(temp$classe)
# nn.fit <- neuralnet(f, data = temp, linear.output = TRUE)
# rm(temp)
```

The accuracy is around 50 percent, which is quiet low.

### 4.4 - Xgboost

Then, I will try Neural Networks and check it's efficiency.


```r
library(xgboost)
xg.train <- as.matrix(training[,-20])
mode(xg.train) = "numeric"
xg.test <- as.matrix(testing[,-20])
mode(xg.test) = "numeric"
xg.train.target = as.matrix(as.integer(training[,20]) - 1)
xg.test.target = as.matrix(as.integer(testing[,20]) - 1)

# set the xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = 5,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 2,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
              )

# start the iteration and record the performance of each model
system.time(bst.cv <- xgb.cv(param = param, data = xg.train, label = xg.train.target, nfold = 4, nrounds = 200, prediction = TRUE, verbose = FALSE))
```

```
##    user  system elapsed 
##  131.19    6.70   76.58
```

```r
min.merror.idx = which.min(bst.cv$dt[, test.merror.mean])
bst.cv$dt[min.merror.idx,]
```

```
##    train.merror.mean train.merror.std test.merror.mean test.merror.std
## 1:                 0                0         0.039746        0.003348
```

The best cross-validation’s minimum error rate occured at 106th iteration. Its info is listed above.


```r
system.time(xg.fit <- xgboost(param = param, data = xg.train, label = xg.train.target, nrounds = min.merror.idx, verbose = 0))
```

```
##    user  system elapsed 
##   42.90    2.09   24.83
```

```r
pred <- predict(xg.fit, xg.test)
pred = matrix(pred, nrow = 5, ncol = length(pred) / 5)
pred = t(pred)
pred = max.col(pred, "last")
confusionMatrix(factor(xg.test.target + 1), factor(pred))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    1    2    3    4    5
##          1 1649   13    6    5    1
##          2   23 1098   15    0    3
##          3    2   17  992   13    2
##          4    3    2   36  920    3
##          5    2    4    9    4 1063
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9723          
##                  95% CI : (0.9678, 0.9763)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.965           
##  Mcnemar's Test P-Value : 0.009785        
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
## Sensitivity            0.9821   0.9683   0.9376   0.9766   0.9916
## Specificity            0.9941   0.9914   0.9930   0.9911   0.9961
## Pos Pred Value         0.9851   0.9640   0.9669   0.9544   0.9824
## Neg Pred Value         0.9929   0.9924   0.9864   0.9955   0.9981
## Prevalence             0.2853   0.1927   0.1798   0.1601   0.1822
## Detection Rate         0.2802   0.1866   0.1686   0.1563   0.1806
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9881   0.9798   0.9653   0.9839   0.9938
```

The overall accuracy is **over 95%**, which is very ideal.

### 4.5 - Result Combination

Finally, It's time to combine these model together. 


```r
svm.pred <- predict(svm.fit, newdata = testing)
rf.pred <- predict(rf.fit, testing, type = "response")
xg.pred <- predict(xg.fit, xg.test)
xg.pred = matrix(xg.pred, nrow = 5, ncol = length(xg.pred) / 5)
xg.pred = t(xg.pred)
xg.pred = max.col(xg.pred, "last")
xg.pred = toupper(letters[xg.pred])
predDF <- data.frame(svm.pred, rf.pred, xg.pred, classe = testing$classe)
# build model
combModFit <- randomForest(classe ~ ., data = predDF)
predComb <- predict(combModFit, predDF)
confusionMatrix(testing$classe, predComb)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1660    6    4    4    0
##          B   22 1107   10    0    0
##          C    2   15  992   15    2
##          D    4    0   24  935    1
##          E    0    0    5    0 1077
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9806         
##                  95% CI : (0.9768, 0.984)
##     No Information Rate : 0.2868         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9755         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9834   0.9814   0.9585   0.9801   0.9972
## Specificity            0.9967   0.9933   0.9930   0.9941   0.9990
## Pos Pred Value         0.9916   0.9719   0.9669   0.9699   0.9954
## Neg Pred Value         0.9934   0.9956   0.9912   0.9961   0.9994
## Prevalence             0.2868   0.1917   0.1759   0.1621   0.1835
## Detection Rate         0.2821   0.1881   0.1686   0.1589   0.1830
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9900   0.9873   0.9757   0.9871   0.9981
```

We can see that the accuracy is around **98%**, which should be called perfect. 

### 4.6 - Calculate For Submit

In usual, we need to retrain all the model with total dataset for a better result, but I skip these steps for convenience. I will calculate and submit the result directly.


```r
svm.pred <- predict(svm.fit, newdata = final)
rf.pred <- predict(rf.fit, final, type = "response")
xg.pred <- predict(xg.fit, as.matrix(final))
xg.pred = matrix(xg.pred, nrow = 5, ncol = length(xg.pred) / 5)
xg.pred = t(xg.pred)
xg.pred = max.col(xg.pred, "last")
xg.pred = toupper(letters[xg.pred])
predDF <- data.frame(svm.pred, rf.pred, xg.pred)
predComb <- predict(combModFit, predDF)
result <- data.frame(id = final$problem_id, pred = predComb)
# t(result)
```

## 5 - Summary

## 5.1 - Hardware & Software Env


```
## R version 3.2.5 (2016-04-14)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 10 x64 (build 10586)
## 
## locale:
## [1] LC_COLLATE=Chinese (Simplified)_China.936 
## [2] LC_CTYPE=Chinese (Simplified)_China.936   
## [3] LC_MONETARY=Chinese (Simplified)_China.936
## [4] LC_NUMERIC=C                              
## [5] LC_TIME=Chinese (Simplified)_China.936    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] randomForest_4.6-12 caret_6.0-71        lattice_0.20-34    
## [4] ggplot2_2.2.0       dplyr_0.5.0        
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.7        formatR_1.4        nloptr_1.0.4      
##  [4] plyr_1.8.4         iterators_1.0.8    tools_3.2.5       
##  [7] digest_0.6.10      lme4_1.1-12        evaluate_0.9      
## [10] tibble_1.2         gtable_0.2.0       nlme_3.1-128      
## [13] mgcv_1.8-15        Matrix_1.2-7.1     foreach_1.4.3     
## [16] DBI_0.5-1          yaml_2.1.13        parallel_3.2.5    
## [19] SparseM_1.72       stringr_1.1.0      knitr_1.14        
## [22] MatrixModels_0.4-1 stats4_3.2.5       grid_3.2.5        
## [25] nnet_7.3-12        R6_2.2.0           rmarkdown_1.0     
## [28] minqa_1.2.4        reshape2_1.4.1     car_2.1-3         
## [31] magrittr_1.5       scales_0.4.1       codetools_0.2-15  
## [34] htmltools_0.3.5    MASS_7.3-45        splines_3.2.5     
## [37] assertthat_0.1     pbkrtest_0.4-6     colorspace_1.2-6  
## [40] quantreg_5.29      stringi_1.1.2      lazyeval_0.2.0    
## [43] munsell_0.4.3
```
