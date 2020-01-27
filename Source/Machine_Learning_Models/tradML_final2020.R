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
library(klaR)
##### FILES #####
sex <- read.csv('/sdata/UK_Biobank/data_queries/taylor_keys/genetic_sex.out.csv', header = T, stringsAsFactors = F)
colnames(sex) <- c("subject_id", "reported_sex", "genetic_sex")
sex$reported_sex <- as.factor(sex$reported_sex)
ukbvol <- read.csv('/wdata/trthomas/cnn_sex_prediction/cnn_sex_tradML/brainregion_vol.out.csv',
                   header = T, stringsAsFactors = F)
ukbvol <- na.omit(ukbvol)
names(ukbvol) <- c(
  "subject_id",
  "peripheral_cortical_grey_norm",
  "peripheral_cortical_grey_notnorm",
  "ventricular_csf_norm",
  "ventricular_csf_notnorm",
  "grey_norm",
  "grey_notnorm",
  "white_norm",
  "white_notnorm",
  "brain_norm",
  "brain_notnorm",
  "thalamus_left",
  "thalamus_right",
  "caudate_left",
  "caudate_right",
  "putamen_left",
  "putamen_right",
  "pallidum_left",
  "pallidum_right",
  "hippocampus_left",
  "hippocampus_right",
  "amygdala_left",
  "amygdala_right",
  "accumbens_left",
  "accumbens_right",
  "brain_stem_fourth_ventricle")
rownames(ukbvol) <- ukbvol$subject_id
ukbvol <- ukbvol %>% 
  select(grey_notnorm, ventricular_csf_notnorm, white_notnorm, brain_notnorm, thalamus_left, thalamus_right,
         caudate_left, caudate_right, putamen_left, putamen_right, pallidum_left, pallidum_right, hippocampus_left,
         hippocampus_right, amygdala_left, amygdala_right, accumbens_left, accumbens_right, brain_stem_fourth_ventricle)
ukbvol <- as.data.frame(scale(ukbvol))
ukbvol$subject_id <- rownames(ukbvol)
ukbvol <- merge(ukbvol, sex, by = "subject_id")
ukbvol$genetic_sex <- NULL
ukbvol <- ukbvol %>% 
  select(subject_id, reported_sex, grey_notnorm, ventricular_csf_notnorm, white_notnorm, brain_notnorm, thalamus_left, thalamus_right,
         caudate_left, caudate_right, putamen_left, putamen_right, pallidum_left, pallidum_right, hippocampus_left,
         hippocampus_right, amygdala_left, amygdala_right, accumbens_left, accumbens_right, brain_stem_fourth_ventricle)
ukbvol <- ukbvol %>% 
  mutate(reported_sex = case_when(reported_sex == 0 ~ "F",
                                  reported_sex == 1 ~ "M"))
ukbvol$reported_sex <- as.factor(ukbvol$reported_sex)
##### FOLDS #####
folds <- read_tsv('/wdata/lbrueggeman/ukbb_sex/data/folds.tsv')
names(folds) <- c("subject_id", "fold")
ukbvol <- merge(ukbvol, folds, by = "subject_id")
test <- ukbvol %>% 
  filter(fold == "test")
train <- ukbvol %>% 
  filter(fold != "test")
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, savePred = T, classProbs = T)

# Logistic regression
lr_model <- train(as.factor(reported_sex) ~ grey_notnorm + ventricular_csf_notnorm + white_notnorm + brain_notnorm + thalamus_left
                  + thalamus_right + caudate_left + caudate_right + putamen_left + putamen_right + pallidum_left
                  + pallidum_right + hippocampus_left + hippocampus_right + amygdala_left + amygdala_right + accumbens_left
                  + accumbens_right + brain_stem_fourth_ventricle,
                  data = train,
                  method = "glm", 
                  metric = "Accuracy",
                  trControl = trctrl)

lr_predictions <- predict(lr_model, test, type = "prob")
colnames(lr_predictions) <- c("lr_F", "lr_M")

# Random forest
rf_model <- train(as.factor(reported_sex) ~ grey_notnorm + ventricular_csf_notnorm + white_notnorm + brain_notnorm + thalamus_left
                  + thalamus_right + caudate_left + caudate_right + putamen_left + putamen_right + pallidum_left
                  + pallidum_right + hippocampus_left + hippocampus_right + amygdala_left + amygdala_right + accumbens_left
                  + accumbens_right + brain_stem_fourth_ventricle,
                  data = train,
                  method = "ranger", 
                  metric = "Accuracy", 
                  tuneLength = 10, 
                  trControl = trctrl)

rf_predictions <- predict(rf_model, test, type = "prob")
colnames(rf_predictions) <- c("rf_F", "rf_M")

# Regularized Discriminant Analysis
trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, savePred = T, classProbs = T)
rda_model <- train(as.factor(reported_sex) ~ grey_notnorm + ventricular_csf_notnorm + white_notnorm + brain_notnorm + thalamus_left
                    + thalamus_right + caudate_left + caudate_right + putamen_left + putamen_right + pallidum_left
                    + pallidum_right + hippocampus_left + hippocampus_right + amygdala_left + amygdala_right + accumbens_left
                    + accumbens_right + brain_stem_fourth_ventricle,
                    data = train, 
                    method = "rda",
                    trControl=trctrl,
                    tuneLength = 10, 
                    metric = "Accuracy")

rda_predictions <- predict(rda_model, newdata = test , type = "prob")
colnames(rda_predictions) <- c("rda_F", "rda_M")

# Naive Bayes
nb_model <- train(as.factor(reported_sex) ~ grey_notnorm + ventricular_csf_notnorm + white_notnorm + brain_notnorm + thalamus_left
                  + thalamus_right + caudate_left + caudate_right + putamen_left + putamen_right + pallidum_left
                  + pallidum_right + hippocampus_left + hippocampus_right + amygdala_left + amygdala_right + accumbens_left
                  + accumbens_right + brain_stem_fourth_ventricle,
                  data = train, 
                  method = "nb",
                  trControl=trctrl,
                  tuneLength = 10, 
                  metric = "Accuracy")

nb_predictions <- predict(nb_model, newdata = test , type = "prob")
nb_predictions <- round(nb_predictions, digits = 7)
colnames(nb_predictions) <- c("nb_F", "nb_M")

# Test predictions
test_preds <- test %>% 
  dplyr::select(subject_id, reported_sex)
test_preds <- cbind(test_preds, lr_predictions, rf_predictions, rda_predictions, nb_predictions)
test_preds <- test_preds %>% 
  dplyr::select(subject_id, reported_sex, lr_M, rf_M, rda_M, nb_M)
colnames(test_preds) <- c("subject_id", "reported_sex", "logistic_regression_prob", "random_forest_prob", "regularized_discriminant_analysis_prob", "naive_bayes_prob")
write.csv(test_preds, file = '/wdata/trthomas/cnn_sex_prediction/final_2020/test_predictions_012220.csv', row.names = F, quote = F)
