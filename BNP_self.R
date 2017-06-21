setwd("C:\\Users\\Vaibhav\\Desktop\\Chaps\\ML\\Assignments\\Insurance-BNP-Paribas")

#When number of rows is more than 1 Lac, then we use fread to read the file.
library(data.table)
train = fread("train.csv")
pred=fread("test.csv")

dim(train)
dim(pred)

#There is a difference of 1 col in train and pred data

#Let's understand what is the difference in the column names of train data and pred data:
setdiff(names(train),names(pred))

names(train)
names(pred)
#So pred data does not have the col "target" - which is our target variable, rest all the cols are same

library(dplyr)
glimpse(train)

#We notice too many missing values. 
#Finding the number of missing values in each column
colSums(is.na(train))

#Deleting all the variables having NAs > 40000
ret=NULL
p=colSums(is.na(train))
j=1
for(i in 1:length(p))
{
  if(p[i]<40000)
  {
    ret[j]=i
    j=j+1
  }
}

# Now ret has all the cols which need to be retained i.e the columns having less than 40% NAs
ret
length(ret)
new_train=train[,ret, with=FALSE] # put with = false for ret
head(new_train)

#The same cols we need to retain for pred data also but pred data doesn't have "target" variable
ret[1]
ret-1
ret_pred=c(ret[1], (ret-1)[3:length(ret)])
new_pred=pred[,ret_pred, with=FALSE]

#Check if the same columns have been retained for train data and pred data
names(new_train)
names(new_pred)
length(names(new_train))
glimpse(new_train)

#Checking the number of NAs in train and pred data
colSums(is.na(new_train))
colSums(is.na(new_pred))
#Check if there are NAs in target variable
sum(is.na(new_train$target))


#Applying MICE to impute NAs in independent variables of train data
imp_mice=mice(new_train[,-c(1,2)], seed=100)#since 1st col. corresponds to ID and 2nd col. is target variable
imputed=complete(imp_mice, 1) 
names(new_train)
train=cbind(new_train[,c(1:2)], imputed)
colSums(is.na(train))

#Applying MICE to impute NAs in independent variables of pred data
imp_mice_pred = mice(new_pred[,-c(1,2)], seed=100)
imputed_pred=complete(imp_mice_pred,1)
pred_d=cbind(new_pred[,c(1:2)], imputed_pred)
colSums(is.na(pred_d))

#Applying PCA
#Seperating character and numeric vars
train_char=data.frame(1:nrow(train))
train_num=data.frame(1:nrow(train))
train=as.data.frame(train)

#Seperating the numeric variables from character variables in train data
for(i in 1:ncol(train))
{
  print(i)
    if(class(train[,i])=="character")
  {
    train_char=cbind(train_char, train[,i])
    names(train_char)[length(train_char)]=names(train)[i]
  }else 
  { 
    train_num=cbind(train_num, train[,i])
    names(train_num)[length(train_num)]=names(train)[i]
  }
}
train_char=train_char[,-1]
train_num=train_num[,-1]

#Same way seperate the numerical variables and character variables of the pred

pred_char=data.frame(1:nrow(pred_d))
pred_num=data.frame(1:nrow(pred_d))
pred_d=as.data.frame(pred_d)
head(pred_d)

i=1
for(i in 1:ncol(pred_d))
{
  print(i)
  if(class(pred_d[,i])=="character")
  {
    pred_char=cbind(pred_char, pred_d[,i])
    names(pred_char)[length(pred_char)]=names(pred_d)[i]
  }else 
  { 
    pred_num=cbind(pred_num, pred_d[,i])
    names(pred_num)[length(pred_num)]=names(pred_d)[i]
  }
}
head(pred_char)
head(pred_num)
pred_char=pred_char[,-1]
pred_num=pred_num[,-1]
names(train_num)
names(pred_num)

#Applying pca on numerical variables of train data
pr.out=prcomp(train_num[,-c(1:2)], scale=TRUE) #excluding id and target variable
pr.out$x #new pcomp values
pr.out$rotation
pr.out$sdev
#so there are 13 principal components 
#variance explained by each principal component:
var=pr.out$sdev^2

#proportion of variance explained by each principal component
pve=var/sum(var)
pve
cumsum(pve)
plot(cumsum(pve), xlab ="Principal Component", ylab =" Cumulative Proportion of Variance Explained ", ylim=c(0,1), type="b") 

#PC 7 explains 99% of variance, so we'll select upto PC7

#Prediction of principal components on pred data
names(pred_num)
pca_pred=predict(pr.out,pred_num[,-1])
#this has scaled the pred data using center and scale of train data
#    then it will multiply the train data and convert pred data into principal components

#Next let's concatenate the character data with numerical principal components
predictor_vars= cbind(pr.out$x[,1:7], train_char)
training=cbind(train_num[,c(1:2)], predictor_vars)
head(training)

#Concatenating pred data set
pred_data=cbind(pred_num[,1], pca_pred[,1:7], pred_char)
#no target variable in our pred data
head(pred_data)
names(pred_data)[1]="ID"

#Splitting train data into train and test
library(caTools)
spl<-sample.split(training$target,0.7)
model_train<-subset(training,spl==T)
mod_test<-subset(training,spl==F)


#Target variable is target

library(xgboost)
library(Matrix)

#first convert the numerical vars into sparse matrix
head(mod_train)
head(mod_test)
#1st 7 are numerical predictor cols

sparse_matrix<-sparse.model.matrix(target~.-1-ID,model_train)
#-1 since we don't want the intercept
sparse_matrix@Dim
write.csv(as.matrix(sparse_matrix), "C:\\Users\\Vaibhav\\Desktop\\Chaps\\ML\\Assignments\\Insurance-BNP-Paribas\\sparse_matrix.csv")
dtrain<-xgb.DMatrix(data=sparse_matrix,label=model_train$target)

#Making the xgboost model to minimize test error
head(mod_test)
sparse_mod_test<-sparse.model.matrix(target~.-1-ID,data=mod_test)
sparse_mod_test@Dim
dtest<-xgb.DMatrix(data=sparse_mod_test,label=mod_test$target)

watchlist<-list(test=dtest) #we want to keep a watchlist only on test error
params=list(eta=0.01,objective="binary:logistic",max_depth=9)
model_xgb<-xgb.train(data=dtrain,params = params, nrounds = 234,early_stopping_rounds = 100, watchlist = watchlist )
#22.02% test error
imp<-xgb.importance(colnames(sparse_matrix),model = model_xgb)
imp #
xgb.plot.tree(colnames(sparse_matrix),model = model_xgb,n_first_tree = 2)
xgb.plot.importance(imp,numberOfClusters = 1)

#names(model_train)
#xgb.importance(feature_names = colnames(model_train[,-length(names(model_train))]),model = model_xgb,data = dtrain,label = model_train$target)

head(train_imputed)

#Applying prediction of xgboost on pred data

#Making the sparse matrix 
head(pred_data)
sparse_pred<-sparse.model.matrix(~.-1-ID,data=pred_data)
#Since there is no target var hence the above command

#Converting into xgboost type
dpred<-xgb.DMatrix(data=sparse_pred)

#Applying prediction on dpred
pred_xgb<-predict(model_xgb,dpred)
pred_xgb

final=cbind(pred_data$ID, pred_xgb)
final=as.data.frame(final)
names(final)
names(final)=c("ID", "Pred_Prob")
final

# Since there is no preference of errors, we'll assume threshold=.5
final$prediction=ifelse(final$Pred_Prob>.5,1,0)