# PLOTS FOR Brueggeman et al ISBI 2020
library(tidyverse)
library(RColorBrewer)
library(ROCR)
library(data.table)

########### AUROC PLOT

test_ids = read.csv(file='/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/allmodels_sex_predictions.csv', header=T,sep=',')[,'subject_id']

covars = read.csv('/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/covars.out.csv') %>%
  select(-c(X53.0.0, X53.1.0, X54.0.0, X54.1.0, X21001.0.0, X21001.1.0, X21002.0.0, X21002.1.0, X21003.0.0, X21003.1.0)) %>%
  setNames(., c('subject_id','date_imaged','imaging_center','bmi','weight','age_imaged','signal_to_noise','contrast_to_noise','x_pos','y_pos','z_pos','table_pos')) %>%
  na.omit()

df = read.csv(file='/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/allmodels_sex_predictions.csv', header=T,sep=',') %>%
  setNames(., c('subject_id','reported_sex','logistic regression','random forest','rLDA','naive bayes','CNN')) %>% 
  filter(subject_id %in% test_ids) %>% 
  left_join(., read.csv('/wdata/lbrueggeman/ukbb_sex/data/bvol_table.csv')) %>%
  left_join(., covars) %>% na.omit()

df$date_imaged = as.numeric(tstrsplit(df$date_imaged,'-')[[2]])/12 + as.numeric(tstrsplit(df$date_imaged,'-')[[1]])

df$`logistic regression` = lm(`logistic regression`~date_imaged + imaging_center+bmi+weight+age_imaged+signal_to_noise+contrast_to_noise+x_pos+y_pos+z_pos+table_pos, df)$residuals
df$`random forest` = lm(`random forest`~date_imaged + imaging_center+bmi+weight+age_imaged+signal_to_noise+contrast_to_noise+x_pos+y_pos+z_pos+table_pos, df)$residuals
df$`rLDA` = lm(rLDA~date_imaged + imaging_center+bmi+weight+age_imaged+signal_to_noise+contrast_to_noise+x_pos+y_pos+z_pos+table_pos, df)$residuals
df$`naive bayes` = lm(`naive bayes`~date_imaged + imaging_center+bmi+weight+age_imaged+signal_to_noise+contrast_to_noise+x_pos+y_pos+z_pos+table_pos, df)$residuals
df$`CNN` = lm(CNN~date_imaged + imaging_center+bmi+weight+age_imaged+signal_to_noise+contrast_to_noise+x_pos+y_pos+z_pos+table_pos, df)$residuals
df$`base model` = lm(brain_notnorm~date_imaged + imaging_center+bmi+weight+age_imaged+signal_to_noise+contrast_to_noise+x_pos+y_pos+z_pos+table_pos, df)$residuals

auroc = df %>%
  select(subject_id, reported_sex, `logistic regression`, `random forest`, rLDA, `naive bayes`, CNN, `base model`) %>%
  pivot_longer(., cols=c(`logistic regression`, `random forest`, rLDA, `naive bayes`, CNN, `base model`)) %>%
  group_by(name) %>% 
  group_modify( ~ data.frame(auroc = performance(prediction(.x$value, .x$reported_sex),'auc')@y.values[[1]]))



df %>%
  select(subject_id, reported_sex, `logistic regression`, `random forest`, rLDA, `naive bayes`, CNN, `base model`) %>%
  setNames(., nm = c('subject_id', 'reported_sex', 'logistic regression (0.683)', 'random forest (0.683)', 'rLDA (0.680)', 'naive bayes (0.637)', 'CNN (0.849)', 'base model (0.627)')) %>%
  pivot_longer(., cols=c(`logistic regression (0.683)`, `random forest (0.683)`, `rLDA (0.680)`, `naive bayes (0.637)`, `CNN (0.849)`, `base model (0.627)`)) %>%
  group_by(name) %>% 
  group_modify( ~ data.frame(x_val = performance(prediction(.x$value, .x$reported_sex),'tpr','fpr')@x.values[[1]], y_val = performance(prediction(.x$value, .x$reported_sex),'tpr','fpr')@y.values[[1]])) %>% 
  ggplot(., aes(x=x_val, y=y_val, group=name, color=name)) +
  geom_abline(intercept=0,slope=1, lty=2) +
  geom_line(size=2, alpha=.5) +
  theme_light() +
  theme(legend.position = c(.7,.2), legend.title = element_blank(), legend.text = element_text(size=15)) +
  scale_color_brewer(type='qual',palette = 'Dark2') +
  ylab('True Positive Rate') + xlab('False Positive Rate') +
  ggsave('/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Paper/Figures/AUROC_models.pdf', device='pdf', width = 7, height=5, units='in')



########### REGION PLOT


structure_guide = list('first_10' = 'thalamus_left',
                       'first_11' = 'caudate_left',
                       'first_12' = 'putamen_left',
                       'first_13' = 'pallidum_left',
                       'first_16' = 'brain_stem_fourth_ventricle',
                       'first_17' = 'hippocampus_left',
                       'first_18' = 'amygdala_left',
                       'first_26' = 'accumbens_left',
                       'first_49' = 'thalamus_right',
                       'first_50' = 'caudate_right',
                       'first_51' = 'putamen_right',
                       'first_52' = 'pallidum_right',
                       'first_53' = 'hippocampus_right',
                       'first_54' = 'amygdala_right',
                       'first_58' = 'accumbens_right',
                       'fast_1' = 'csf',
                       'fast_2' = 'grey',
                       'fast_3' = 'white')

covars = read.csv('/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/covars.out.csv') %>%
  dplyr::select(-c(X53.0.0, X53.1.0, X54.0.0, X54.1.0, X21001.0.0, X21001.1.0, X21002.0.0, X21002.1.0, X21003.0.0, X21003.1.0)) %>%
  setNames(., c('subject_id','date_imaged','imaging_center','bmi','weight','age_imaged','signal_to_noise','contrast_to_noise','x_pos','y_pos','z_pos','table_pos')) %>%
  na.omit()
covars$date_imaged = as.numeric(tstrsplit(covars$date_imaged,'-')[[2]])/12 + as.numeric(tstrsplit(covars$date_imaged,'-')[[1]])
colnames(covars) = paste('covars_',colnames(covars),sep='')


bvols = read.csv('/wdata/lbrueggeman/ukbb_sex/data/bvol_table.csv') %>%
  dplyr::select(-c(peripheral_cortical_grey_notnorm, brain_norm))
bvols_guide = list('ventricular_csf_notnorm' = 'csf',
                   'grey_notnorm' = 'grey',
                   'white_notnorm' = 'white',
                   'brain_notnorm' = 'total_brain')
colnames(bvols)[match(names(bvols_guide), colnames(bvols))] = bvols_guide
colnames(bvols)[2:20] = paste('bvols_',colnames(bvols)[2:20],sep='')
bvols$sex_resid = NULL

first_results = read.csv('/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/first_pred_df.csv')
fast_results = read.csv('/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/fast_pred_df.csv')

results = inner_join(first_results, fast_results)
colnames(results)[match(names(structure_guide), colnames(results))] = structure_guide
colnames(results)[c(1:15,17:19)] = paste('CNN_',colnames(results)[c(1:15,17:19)],sep='')

results = results %>%
  left_join(.,covars, by=c('labels' = 'covars_subject_id')) %>% 
  left_join(.,bvols, by=c('labels' = 'subject_id')) %>%
  na.omit() 

results$bvols_total_brain = NULL
results = results %>%
  map_at(.,
         .at=colnames(results)[grep('CNN|bvols',colnames(results))],
         ~ lm(. ~ results$covars_age_imaged + results$covars_bmi + results$covars_contrast_to_noise + results$covars_date_imaged + results$covars_imaging_center + results$covars_signal_to_noise + results$covars_table_pos + results$covars_weight + results$covars_x_pos + results$covars_y_pos + results$covars_z_pos)$residuals
) %>%
  data.frame()

results = results[,-grep('covars', colnames(results))]

results = results %>%
  pivot_longer(., cols=c(colnames(results)[grep('CNN',colnames(results))],colnames(results)[grep('bvols',colnames(results))])) %>%
  separate(., name, into=c('source', 'struct'), sep='_', extra='merge', remove=F)

results$source[results$source == 'bvols'] = 'volume-based'

auroc = results %>%
  group_by(source, struct, name) %>% 
  group_modify( ~ data.frame(auroc = performance(prediction(.x$value, .x$sex),'auc')@y.values[[1]])) 

foo = auroc[auroc$source == 'CNN',] %>% arrange(auroc) 
foo = foo$struct


auroc$struct = factor(auroc$struct, levels=foo)
auroc$level = 'subcortical'
auroc$level[auroc$struct %in% c('grey','csf','white')] = 'cortical'

auroc %>%
  ggplot(., aes(x=struct, y=auroc, fill=source)) +
  geom_hline(yintercept=0.5, lty=1, alpha=0.5, lwd=2) +
  geom_linerange(aes(x=struct, ymin=0,ymax=auroc), size=3, position = position_dodge(width=.6), color='grey') +
  geom_point(size=5, position = position_dodge(width=.6), pch=21) +
  coord_flip() +
  facet_grid(rows = vars(level), space='free_y', scales = 'free_y', switch='y') +
  theme_minimal() +
  theme(strip.background = element_rect(fill='grey90', colour = NA)) +
  theme(strip.text.y = element_text(size=30),
        axis.text.y.left = element_text(hjust=1, size=25),
        axis.title.y.left = element_blank(),
        legend.position = c(.87,.15), legend.title = element_text(size=18, hjust=0.5), legend.text = element_text(size=15), legend.background = element_rect(color='black'),
        axis.text.x = element_text(size=20),
        axis.title.x = element_text(size=25)) +
  ylim(c(0,1)) + 
  ggsave('/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Paper/Figures/AUROC_regions.pdf', device='pdf', width = 12, height=16, units='in')



############ region aurocs correlation
library(Hmisc)
foo = read.csv('/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/region_aurocs.csv')

bar = foo[foo$source == 'CNN',] %>% arrange(struct)
nix = foo[foo$source == 'volume-based',] %>% arrange(struct)

rcorr(bar$auroc, nix$auroc, type='spearman')

wilcox.test(bar[bar$level == 'subcortical','auroc'], nix[nix$level == 'subcortical','auroc'])

















results %>%
  group_by(source, struct) %>% 
  group_modify( ~ data.frame(x_val = performance(prediction(.x$value, .x$reported_sex),'tpr','fpr')@x.values[[1]], y_val = performance(prediction(.x$value, .x$reported_sex),'tpr','fpr')@y.values[[1]])) %>% head()
  
  ggplot(., aes(x=x_val, y=y_val, group=name, color=name)) +
  geom_line(size=1) +
  theme_light() +
  theme(legend.position = c(.7,.2), legend.title = element_blank()) +
  scale_color_brewer(type='qual',palette = 'Set1') +
  geom_abline(intercept=0,slope=1) +
  ylab('True Positive Rate') + xlab('False Positive Rate')

  



















base_df = read.csv('/wdata/lbrueggeman/ukbb_sex/data/bvol_table.csv')

ukbvol <- read.csv('/wdata/trthomas/cnn_sex_prediction/cnn_sex_tradML/brainregion_vol.out.csv', header = T, stringsAsFactors = F)
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

region_results = first_results %>%
  left_join(., fast_results) %>% 
  left_join(.,base_df[,c('subject_id','sex')], by=c('labels' = 'subject_id'))

region_results = region_results %>%
  pivot_longer(.,cols=c('first_10','first_11', 'first_12','first_13','first_16','first_17','first_18','first_26','first_49','first_50','first_51','first_52','first_53','first_54','first_58','fast_1','fast_2','fast_3')) %>% head()

ggplot(region_results)








first_results = first_results %>%
  
  
  performance(prediction(first_results$first_11, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_12, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_13, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_16, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_17, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_18, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_26, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_49, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_50, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_51, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_52, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_53, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_54, first_results$sex), 'auc')@y.values
performance(prediction(first_results$first_58, first_results$sex), 'auc')@y.values

head(first_results)


































auroc = read.csv(file='/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/allmodels_sex_predictions.csv', header=T,sep=',') %>% 
  left_join(., base_df[,c('subject_id','base_pred')]) %>%
  setNames(., c('subject_id','reported_sex','logistic regression','random forest','rLDA','naive bayes','CNN', 'base model')) %>%
  pivot_longer(., cols=c(`logistic regression`, `random forest`, rLDA, `naive bayes`, CNN, `base model`)) %>%
  group_by(name) %>% 
  group_modify( ~ data.frame(auroc = performance(prediction(.x$value, .x$reported_sex),'auc')@y.values[[1]]))

bar = read.csv('/wdata/lbrueggeman/ukbb_sex/data/bvol_table.csv')

df = read.csv(file='/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/allmodels_sex_predictions.csv', header=T,sep=',') %>%
  setNames(., c('subject_id','reported_sex','logistic regression','random forest','rLDA','naive bayes','CNN')) %>%
  pivot_longer(., cols=c(`logistic regression`, `random forest`, rLDA, `naive bayes`, CNN)) %>%
  group_by(name) %>% 
  group_modify( ~ data.frame(auroc = performance(prediction(.x$value, .x$reported_sex),'auc')@y.values[[1]]))



  
read.csv(file='/wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/allmodels_sex_predictions.csv', header=T,sep=',') %>%
  setNames(., c('subject_id','reported_sex','logistic regression (0.88)','random forest (','rLDA','naive bayes','CNN')) %>%
  pivot_longer(., cols=c(`logistic regression`, `random forest`, rLDA, `naive bayes`, CNN)) %>%
  group_by(name) %>% 
  group_modify( ~ data.frame(x_val = performance(prediction(.x$value, .x$reported_sex),'tpr','fpr')@x.values[[1]], y_val = performance(prediction(.x$value, .x$reported_sex),'tpr','fpr')@y.values[[1]])) %>% 
  ggplot(., aes(x=x_val, y=y_val, group=name, color=name)) +
  geom_line(size=1) +
  theme_light() +
  theme(legend.position = c(.7,.2)) +
  scale_color_brewer(type='qual',palette = 'Set1')


lapply(., function(x) data.frame(x_val = as.numeric(x@x.values[[1]]), y_val = as.numeric(x@y.values[[1]])))
  



as.numeric(perf@x.values)
perf@x.values[[1]] %>% head()


pred = prediction(df$CNN, df$reported_sex)
perf = performance(pred,'tpr','fpr')



cols = lapply(RColorBrewer::brewer.pal(5,'Set1'), scales::alpha, 0.8)

plot(perf, col=cols[1])
plot(performance(prediction.obj=prediction(df$logistic_regression_prob, df$reported_sex),measure='tpr',x.measure = 'fpr'), add=T, lwd=5, col=cols[2])
plot(performance(prediction.obj=prediction(df$random_forest_prob, df$reported_sex),measure='tpr',x.measure = 'fpr'), add=T, lwd=5, col=cols[3])
plot(performance(prediction.obj=prediction(df$regularized_discriminant_analysis_prob, df$reported_sex),measure='tpr',x.measure = 'fpr'), add=T, lwd=5, col=cols[4])
plot(performance(prediction.obj=prediction(df$naive_bayes_prob, df$reported_sex),measure='tpr',x.measure = 'fpr'), add=T, lwd=5, col=cols[5])


