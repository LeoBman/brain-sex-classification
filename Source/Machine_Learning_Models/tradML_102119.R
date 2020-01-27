
###### LIBRARIES ######
library(caret)
library(randomForest)
library(mlbench)
library(tidyverse)
library(ROCR)
library(ggplot2)
library(dplyr)
library(e1071)
library(kernlab)
library(ranger)

###### DATA INPUT #####
dat <- read.csv('/wdata/lbrueggeman/ukbb_sex/data/bvol_table.csv', header = T, stringsAsFactors = F)
touse <- read.csv('/wdata/lbrueggeman/ukbb_sex/data/subjects_to_use_sexresid.csv', stringsAsFactors = F)
names(touse) <- c("subject_id", "sex_resid")
dat <- dat %>%
  filter(subject_id %in% touse$subject_id)

dats <- scale(dat[,2:22])
datmerge <- dat
datmerge[,2:22] <- NULL
dats <- cbind(datmerge, dats)
dats[,8:9] <- NULL
dat <- dats

folds <- read_tsv('/wdata/lbrueggeman/ukbb_sex/data/folds.tsv')
names(folds) <- c("subject_id", "fold")
folds <- folds %>% 
  filter(subject_id %in% touse$subject_id)
folds_ctrl <- folds %>% group_by(fold) %>% nest() %>% pull(data) %>% lapply(unlist)
folds_ctrl[[1]] <- NULL

dats <- merge(dat, folds, by = "subject_id")
rownames(dats) <- dats$subject_id
dats$sex <- NULL
test <- dats %>% 
  filter(fold == "test")
train <- dats %>% 
  filter(fold != "test")

train_rownames <- rownames(train)
folds_out <- lapply(folds_ctrl, function(x) {which(train_rownames %in% x)})
folds_in <- lapply(folds_ctrl, function(x) {which(!(train_rownames %in% x))})

rownames(train) <- train$subject_id
train$subject_id <- NULL
train$fold <- NULL
rownames(test) <- test$subject_id
test$subject_id <- NULL
test$fold <- NULL

###### LINEAR REGRESSION ######
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, 
                       index = folds_in, indexOut = folds_out)

lm_model <- train(sex_resid ~.,
                  data = train,
                  method = "lm",
                  metric = "RMSE",
                  trControl = trctrl)

lm_predict <- predict(lm_model, test)
lm_rsquared <- (cor(lm_predict, test$sex_resid))^2

##### Permutations ######
set.seed(1234)
lm_perms <- data.frame("region" = NA, "r_squared" = NA, "iteration" = NA, "model_type" = NA)
i = 1
for (j in 1:10){
  
  for (i in 1:20) {
    test_shuf <- test
    test_shuf[,i] = sample(test_shuf[,i], replace=FALSE)
    
    lm_predict_shuffle <- predict(lm_model, test_shuf)
    lm_rsquared_shuffle <- (cor(lm_predict_shuffle, test_shuf$sex_resid))^2
    var <- colnames(test_shuf[i])
    vec <- c(var, lm_rsquared_shuffle, j, "lm_model")
    lm_perms <- rbind(lm_perms, vec, stringsAsFactors = F)
  }
}
write.csv(lm_perms, file = "lm_permutations.csv", quote = F, row.names = F)
##### RANDOM FOREST #####
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1,
                       index = folds_in, indexOut = folds_out)
rf_model <- train(sex_resid ~., 
                  data = train, 
                  method = "ranger", 
                  metric = "RMSE", 
                  tuneLength = 10, 
                  trControl = trctrl)

rf_predict <- predict(rf_model, test)
rf_rsquared <- (cor(rf_predict, test$sex_resid))^2

###### Permutations ######
set.seed(1234)
rf_perms <- data.frame("region" = NA, "r_squared" = NA, "iteration" = NA, "model_type" = NA)
i = 1
for (j in 1:10) {
  for (i in 1:20) {
    test_shuf <- test
    test_shuf[,i] = sample(test_shuf[,i], replace=FALSE)
    
    rf_predict_shuffle <- predict(rf_model, test_shuf)
    rf_rsquared_shuffle <- (cor(rf_predict_shuffle, test_shuf$sex_resid))^2
    var <- colnames(test_shuf[i])
    vec <- c(var, rf_rsquared_shuffle, j, "rf_model")
    rf_perms <- rbind(rf_perms, vec, stringsAsFactors = F)
  }
}

##### ENET #####
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, 
                       index = folds_in, indexOut = folds_out)
enet_model <- train(sex_resid ~.,
                    data = train, 
                    method = "enet",
                    trControl=trctrl,
                    preProc = c("center","scale"),
                    tuneLength = 10, 
                    metric = "RMSE")

enet_predict <- predict(enet_model,test)
enet_rsquared <- (cor(enet_predict, test$sex_resid))^2

##### Permutations #####
set.seed(1234)
enet_perms <- data.frame("region" = NA, "r_squared" = NA, "iteration" = NA, "model_type" = NA)
i = 1
for (j in 1:10) {
  for (i in 1:20) {
    test_shuf <- test
    test_shuf[,i] = sample(test_shuf[,i], replace=FALSE)
    
    enet_predict_shuffle <- predict(enet_model, test_shuf)
    enet_rsquared_shuffle <- (cor(enet_predict_shuffle, test_shuf$sex_resid))^2
    var <- colnames(test_shuf[i])
    vec <- c(var, enet_rsquared_shuffle, j, "e_net")
    enet_perms <- rbind(enet_perms, vec, stringsAsFactors = F)
  }
}

### bind model scores ####
permutations <- rbind(lm_perms, rf_perms, enet_perms)
permutations <- na.omit(permutations)
write.csv(permutations, file = "permutations_threemodels.csv", quote = F, row.names = F)
