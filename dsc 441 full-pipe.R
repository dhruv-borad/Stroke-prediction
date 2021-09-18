library(dplyr)
library(tidyverse)
library(ggplot2)
library(caret)
library(e1071)
library(fastDummies)
library(ggcorrplot)
library(corrplot)
library(stats)
library(factoextra)
library(cluster)
library(rpart)
library(rattle)
library(boot)
library(pROC)
library(mlbench)

setwd("C:\\Users\\dhruv\\Documents\\csv files")

df <- read.csv("C:\\Users\\dhruv\\Documents\\csv files\\healthcare-dataset-stroke-data.csv")%>% 
  rename_all(tolower)
head(df)

df$stroke<- factor(df$stroke, levels = c(0,1), labels = c("No", "Yes"))
df$gender<-as.factor(df$gender)
df$hypertension<- factor(df$hypertension, levels = c(0,1), labels = c("No", "Yes"))
df$heart_disease<- factor(df$heart_disease, levels = c(0,1), labels = c("No", "Yes"))
df$ever_married<-as.factor(df$ever_married)
df$work_type<-as.factor(df$work_type)
df$residence_type<-as.factor(df$residence_type)
df$smoking_status<-as.factor(df$smoking_status)
df$bmi<-as.numeric(df$bmi)

summary(df)

# data exploration
ggplot(df,aes(x=gender,y=stroke)) + geom_col()
ggplot(data = df, aes(x=gender,fill=gender))+geom_bar()
ggplot(df,aes(x=as.factor(stroke),y=age)) + geom_boxplot() + xlab("Stroke")
ggplot(df,aes(x=stroke,y=avg_glucose_level)) + geom_col()
ggplot(data = df, aes(x=work_type,fill=work_type))+geom_bar()+
  theme(axis.text.x =element_text(angle = 45) )

ggplot(data=df,aes(x=work_type, fill=stroke))+geom_bar(position="fill")+
  coord_cartesian(ylim = c(0,0.25))

# there might be class imbalance here 
ggplot(df,aes(x=gender,y=age,fill=as.factor(stroke))) + geom_col() + labs(fill = "Stroke")
sum(df$stroke == "No")
sum(df$stroke == "Yes")

# data cleaning 
sum(is.na(df))
df <- na.omit(df)
any(df=="N/A")
count(df[df$bmi == "N/A",])
count(df_num[df_num$smoking_status == "Unknown",])

df_num <- df
# Unknown in smoking_status means that the information is unavailable for this patient
df_num <- df_num[!(df_num$smoking_status == "Unknown"),]
df_num <- na.omit(df_num)
df_num <- subset(df_num,select = -id)

# data preprocessing
df_num$gender <- ifelse(df_num$gender =="Male",1,0)
df_num$ever_married <- ifelse(df_num$ever_married == "Yes",1,0)
df_num$residence_type <- ifelse(df_num$residence_type == "Urban",1,0)
df_num$bmi <- as.numeric(df_num$bmi)
df_num$hypertension <- ifelse(df_num$hypertension == "Yes",1,0)
df_num$heart_disease <- ifelse(df_num$heart_disease == "Yes",1,0)
df_num$stroke <- ifelse(df_num$stroke == "Yes",1,0)

# private -> 4, self-employed -> 5, govt_job -> 2, children -> 1, never_worked -> 3
df_num$work_type <- as.integer(as.factor(df_num$work_type))

# formerly smoked -> 1, never smoked -> 2, smokes -> 3, 4- unknown
df_num$smoking_status <- as.integer(as.factor(df_num$smoking_status))
glimpse(df_num)
summary(df_num)

# removed both outlier 
ggplot(df_num,aes(x=as.factor(stroke),y=age)) + geom_boxplot() + xlab("Stroke")

# correlation plot
corrplot(cor(df_num),method = "number")
ggcorrplot(cor(df_num),hc.order = TRUE)

# binning
df_num_bins <- df_num
df_num_bins <- df_num_bins %>% mutate(glucosefactor = cut(avg_glucose_level,
                                                     breaks = c(-Inf,70,140,Inf),
                                                     labels=c("low","normal","high")))
# smoothing
low <- df_num_bins %>% 
  filter(glucosefactor == 'low') %>% 
  mutate(avg_glucose_level = mean(avg_glucose_level, na.rm = T))
normal <- df_num_bins %>% 
  filter(glucosefactor == 'normal') %>% 
  mutate(avg_glucose_level = mean(avg_glucose_level, na.rm = T))
high <- df_num_bins %>% 
  filter(glucosefactor == 'high') %>% 
  mutate(avg_glucose_level = mean(avg_glucose_level, na.rm = T))

df_num_bins <- bind_rows(list(low, normal, high))
head(df_num_bins)

# whenever i am plotting the avg glucose levels or age. the labels on that y axis 
# are in the 10^5. while the values are way smaller. so when i tried putting
#  scale_y_continuous to adjust the values it is messed up
ggplot(df_num_bins,aes(x=stroke,y=avg_glucose_level,fill=glucosefactor)) + geom_col()

# min-max normalization 
normalize = function(x){
  return((x-min(x,na.rm = TRUE))/(max(x,na.rm = TRUE)-min(x,na.rm = TRUE)))
}

# 2 is for all rows and 1 for all columns 
df_min_max <- df_num_bins
df_min_max$bmi <- as.data.frame(apply(df_num_bins[9],2, normalize))
head(df_min_max)
summary(df_min_max)

# clustering
df_num <- subset(df_num,select = -stroke)

train_control = trainControl(method = "cv", number = 10)
preproc = c("center", "scale")

# k means
fviz_nbclust(df_num, kmeans, method = "wss") # 3 or 4
fviz_nbclust(df_num, kmeans, method = "silhouette") # 2
fit <- kmeans(df_num, centers = 4, nstart = 25)
fit
fviz_cluster(fit, data = df_num)


preproc <- preProcess(df_num, method=c("center", "scale"))
df_num <- predict(preproc, df_num)
pca = prcomp(df_num)
rotated_data = as.data.frame(pca$x)
rotated_data$color <- df_num_bins$stroke  
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point(alpha = 0.3)

rotated_data$clusters = as.factor(fit$cluster)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = clusters)) + geom_point()

# HAC
dist_mat <- dist(df_num, method = 'euclidean')
hfit <- hclust(dist_mat, method = 'complete')
plot(hfit)

# try with gower for numerical and categorical predictors
df_num_gower <- df_num_bins 
df_num_gower <- subset(df_num_bins,select = -glucosefactor)
df_num_gower$stroke <- ifelse(df_num_gower$stroke ==1,"Yes","No")


dist_mat2 <- daisy(df_num, metric = "gower")
summary(dist_mat2)
fviz_nbclust(df_num, FUN = hcut, method = "wss") # 5 or 6 , 2 3 
fviz_nbclust(df_num, FUN = hcut, method = "silhouette") # 2

rotated_data$color <- ifelse(rotated_data$color =="1","Yes","No")
h4 <- cutree(hfit, k=4)
fviz_cluster(list(data = df_num, cluster = h4))
rotated_data$clusters = as.factor(h4)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = clusters)) + geom_point()
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point()+
  scale_colour_discrete("Stroke")

result <- data.frame(Stroke = rotated_data$color, HAC4 = h4, Kmeans = fit$cluster)
head(result, n = 20)

# cross tab for HAC
result %>% group_by(HAC4) %>% select(HAC4, Stroke) %>% table()

# cross tab for k means 
result %>% group_by(Kmeans) %>% select(Kmeans, Stroke) %>% table()
# the results of HAC and kmeans are very similar, we can say that kmeans
# is performing slightly better because in the 4th cluster data is better 
# clustered than HAC.


# at higher number of clusters this metric usually plays a more crucial role
# we try with 6 clusters and lets try average, median, centroid 
# and single linage metrics.

  # The distance between two clusters is the distance between mean
  # of the elements in cluster 1 and the mean of the elements in cluster 2.
  # average

hfit <- hclust(dist_mat, method = 'average')
h6 <- cutree(hfit, k=6)
rotated_data$clusters = as.factor(h6)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = clusters)) + geom_point(alpha = 0.5)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point(alpha = 0.5)+
  scale_colour_discrete("Stroke")


# median - he distance between two clusters is the distance between
# the median of elements in cluster 1 and the median of elements in cluster 2.
hfit <- hclust(dist_mat, method = 'median')
h6 <- cutree(hfit, k=6)
rotated_data$clusters = as.factor(h6)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = clusters)) + geom_point(alpha = 0.5)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point(alpha = 0.5)


# The centroid version is using the distance between cluster centroids.The distance
# between two clusters is the distance between the centroid for cluster 1 and the
# centroid for cluster 2.
hfit <- hclust(dist_mat, method = 'centroid')
h6 <- cutree(hfit, k=6)
rotated_data$clusters = as.factor(h6)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = clusters)) + geom_point(alpha = 0.5)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point(alpha = 0.5)


# The single linkage method adopts a 'friends of friends' clustering strategy according
# to the hclust function documentation. The distance between two clusters is defined as
# the minimum value of all pairwise distances for the elements in cluster 1 and the
# elements in cluster 2.
hfit <- hclust(dist_mat, method = 'single')
h6 <- cutree(hfit, k=6)
rotated_data$clusters = as.factor(h6)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = clusters)) + geom_point(alpha = 0.5)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = color)) + geom_point(alpha = 0.5)


# Logistic regression

stroke1 <- glm(stroke~ gender+ age + hypertension + avg_glucose_level + smoking_status + bmi,
               data=df_num, family = binomial(link = "logit"))
summary(stroke1)

stroke2<- glm(stroke~ age + hypertension + avg_glucose_level + bmi ,
              data=df_num, family = binomial(link = "logit"))
summary(stroke2)

stroke3<- glm(stroke~ age + heart_disease + work_type + residence_type + avg_glucose_level + bmi,
              data=df_num, family = binomial(link = "logit"))
summary(stroke3)

stroke4<- glm(stroke~ age + heart_disease + avg_glucose_level ,
              data=df_num, family = binomial(link = "logit"))
summary(stroke4)

stroke5<- glm(stroke~ age + ever_married + avg_glucose_level ,
              data=df_num, family = binomial(link = "logit"))
summary(stroke5)

# AIC (Akaike information criterion) is a mathematical method for
# evaluating how well a model fits the data it was generated from
# AIC is lower for model 2 among the 5 models performed.

#Use 5-fold cross validation to assess models' quality
model21<- glm(stroke~ age + avg_glucose_level + bmi, data=df_num,
               family = binomial(link = "logit"))
model22<- glm(stroke~ age * hypertension + avg_glucose_level + bmi , data=df_num,
               family = binomial(link = "logit"))
model23<- glm(stroke~ age + hypertension * avg_glucose_level + bmi, data=df_num,
               family = binomial(link = "logit"))
model24<- glm(stroke~ age + gender * hypertension + avg_glucose_level + bmi, data=df_num,
               family = binomial(link = "logit"))
model25<- glm(stroke~ age + hypertension * bmi + avg_glucose_level, data=df_num,
               family = binomial(link = "logit"))


aic_values <- AIC(model21, model22, model23, model24, model25)
aic_values
# with the lowest value of AIC model 22 is best to predict if the patient had the stroke or not
summary(model22)


# Decision tree
df_num$stroke <- ifelse(df_num$stroke == 1,"Yes","No")
df_num$stroke <- as.factor(df_num$stroke)

set.seed(372)
index = createDataPartition(y=df_num$stroke, p=0.7, list=FALSE)
train_set = df_num[index,]
test_set = df_num[-index,]
train_control = trainControl(method = "cv", number = 10)

hypers = rpart.control(minsplit =  500, maxdepth = 3, minbucket = 2000)

tree1 <- train(stroke ~., data = train_set, method = "rpart1SE", trControl = train_control)
tree1

pred_tree <- predict(tree1, test_set)
cm <- confusionMatrix(as.factor(test_set$stroke), pred_tree)
cm

fancyRpartPlot(tree1$finalModel, caption = "")

metrics <- as.data.frame(cm$byClass)
metrics

# decision tree
pred_prob2 <- predict(tree1, test_set, type = "prob")
head(pred_prob2)
roc_obj2 <- roc((test_set$stroke), pred_prob2[,1])
plot(roc_obj2, print.auc=TRUE)

# knn for AUC
knn <- train(stroke ~., data = train_set, method = "knn", trControl = train_control, tuneLength = 20)
knn
pred_df <- predict(knn, test_set)
confusionMatrix(test_set$stroke, pred_df)
pred_prob <- predict(knn, test_set, type = "prob")
roc_obj <- roc((test_set$stroke), pred_prob[,1])
plot(roc_obj, print.auc=TRUE)


# precision = TP/(TP+FP), recall =  TP/(TP+FN)



