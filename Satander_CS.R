#                   GNU GENERAL PUBLIC LICENSE
#                       Version 3, 29 June 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#
#                            Preamble
#
#  The GNU General Public License is a free, copyleft license for
#software and other kinds of works.
#
#  The licenses for most software and other practical works are designed
#to take away your freedom to share and change the works.  By contrast,
#the GNU General Public License is intended to guarantee your freedom to
#share and change all versions of a program--to make sure it remains free
#software for all its users.  We, the Free Software Foundation, use the
#GNU General Public License for most of our software; it applies also to
#any other work released this way by its authors.  You can apply it to
#your programs, too.
# 


# lecture des donnees
training <- read.csv("datas/train.csv")
validing <- read.csv("datas/test.csv")


install.packages("glmnet")
require(glmnet)
install.packages("dummies")
require(dummies)
install.packages("pROC")
require(pROC)
install.packages("ROCR")
require(ROCR)
install.packages("rpart")
require(rpart)
install.packages("rpart.plott")
require(rpart.plot)
install.packages("ggplot2")
require(ggplot2)
install.packages("ggthemes")
require(ggthemes)
install.packages("randomForest")
library(randomForest)
install.packages("xgboost")
library(xgboost)
install.packages("caret")
library(caret)
install.packages("Matrix")
library(Matrix)

DF<-training
DFV<-validing

# voir si des valeur sont NA
DF[is.na(DF)] 
DFV[is.na(DFV)]
str(DF)

eval<-sample(1:dim(DF)[1],20000)
test=DF[eval,]
train=DF[-eval,]

modele.null = glm(factor(TARGET)~1, family = binomial, data = train)
modele.full = glm(factor(TARGET)~., family = binomial, data = train)

step(modele.null, scope = list (lower=modele.null, upper=modele.full), direction = "forward")



modPenalise1 = glm(formula = factor(TARGET) ~ num_meses_var5_ult3 + var15 + 
                     saldo_var30 + var38 + num_var22_ult3 + ind_var30_0 + num_var30 + 
                     num_op_var39_efect_ult1 + ind_var31_0 + saldo_medio_var8_hace2 + 
                     ind_var8_0 + num_var5 + saldo_var5 + ind_var8 + ind_var30 + 
                     num_var45_ult3 + num_var43_recib_ult1 + num_var22_ult1 + 
                     var3 + num_med_var45_ult3 + num_var32_0 + num_var42 + ind_var13 + 
                     saldo_var13_corto + ind_var20_0 + saldo_medio_var12_hace3 + 
                     num_meses_var12_ult3 + saldo_medio_var13_corto_hace3 + num_ent_var16_ult1 + 
                     imp_op_var39_ult1 + saldo_var25 + num_var4 + num_var30_0 + 
                     num_meses_var13_largo_ult3 + num_var13 + ind_var37_cte + 
                     var36 + ind_var32_cte + imp_op_var41_efect_ult3 + num_op_var39_efect_ult3 + 
                     imp_op_var39_comer_ult3 + saldo_medio_var8_hace3 + num_sal_var16_ult1 + 
                     num_meses_var39_vig_ult3 + ind_var39_0 + saldo_medio_var8_ult1 + 
                     saldo_medio_var8_ult3 + num_trasp_var11_ult1, family = binomial, 
                   data = train)

summary(modPenalise1)

mod2 = glm(formula = factor(TARGET) ~., family = binomial, data = train)
summary(mod2)

modTest0.predict=predict.glm(modPenalise1, type="response", newdata=test) 
mod0.predict=predict.glm(mod2, type="response", newdata=test) 
table(ifelse(modTest0.predict>0.2, 1,0), as.factor(test$TARGET)) 
table(ifelse(mod0.predict>0.5, 1,0), as.factor(test$TARGET)) 

r = roc(test$TARGET, modTest0.predict)  # best AUC

plot(r)
r$auc

r2 = roc(test$TARGET, mod0.predict) 

plot(r2)
r2$auc


##### validation #####

modPenalise2.predict=predict.glm(modPenalise1, type="response", newdata=DFV) #donne un score, il permet de "classer" les r?sultats
str(modPenalise2.predict)

#DFV$TARGET = ifelse(predict.glm(modPenalise2.predict, newdata=DFV, type="response")>0.5,1,0)
DFV$TARGET = predict.glm(modPenalise1, newdata=DFV, type="response")
str(DFV$TARGET)
DFV.predict<-data.frame(DFV$ID)
str(DFV.predict)
DFV.predict$TARGET<-DFV$TARGET
str(DFV.predict)

names(DFV.predict)[1]<-"ID"
names(DFV.predict)[2]<-"TARGET"

write.csv(DFV.predict, file = "./resultats/Predict_GLM.csv", row.names = FALSE) 

#0.797 AUC


####### XGBOOST ########




eval<-sample(1:dim(DF)[1],20000)
testxgb=DF[eval,]
trainxgb=DF[-eval,]

DFV$TARGET = -1
sparse_test = sparse.model.matrix(testxgb$TARGET~., data = testxgb)
sparse_train = sparse.model.matrix(trainxgb$TARGET~., data = trainxgb)
sparse_valid <- sparse.model.matrix(TARGET ~., data = DFV)
dtrain = xgb.DMatrix(data = sparse_train, label = trainxgb$TARGET)
dtest = xgb.DMatrix(data = sparse_test, label = testxgb$TARGET)
watchlist <- list(train=dtrain, test=dtest)



bst <- xgb.train(data=dtrain, max_depth=4, eta=1, nthread = 2, nrounds=10, watchlist=watchlist, objective = "binary:logistic")

# cross validation = cv 
# trouver les hyperparametres optimises
# preciser cv 
cv.ctrl = caret::trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                       allowParallel=T)

# preciser les parametres a tester 
xgb.grid = expand.grid(nrounds = 500,
                       max_depth = c(6,8),
                       eta = c(0.1,0.3,1),
                       gamma = c(0.1,0.3,1),
                       colsample_bytree = c(0.1,0.3,1),  
                       min_child_weight = c(1,10), 
                       subsample = c(0.1,0.3,1)
)


set.seed(100)
# train dans le package caret 
# tester les parametre
# "metric = objective" pour les regressions logistiques 
xgb_tune = caret::train(TARGET~.,
                 data=trainxgb,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="RMSE",
                 nthread =3
)
# nthread: utilsier combien de coeur d'ordinateur 

print(xgb_tune)


param = list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.025,
                max_depth           = 8,
                subsample           = 0.7,
                colsample_bytree    = 0.65
)

bst = xgb.train(params = param, data=dtrain, nrounds = 250, verbose = 2, 
              watchlist = watchlist, nthread =3, maximize = FALSE)

pred1 = predict(bst, dtest) #best pred
pred1
table(ifelse(pred1>0.5, 1,0), as.factor(testxgb$TARGET)) 

rxgb = roc(testxgb$TARGET, pred1) 
rxgb$auc
plot(rxgb)

########## version pour valid ##########

pred_valid <- predict(bst, newdata = sparse_valid)
str(pred_valid)
pred_valid.predict = data.frame(DFV$ID,pred_valid)



str(pred_valid.predict)


names(pred_valid.predict)[1]<-"ID"
names(pred_valid.predict)[2]<-"TARGET"

write.csv(pred_valid.predict, file="./resultats/Predict_XGBOOST.csv", row.names = FALSE)

#0.837 AUC


# ****************************************
# ** importance des variables - xgboost ** 
# ****************************************
feature_name <- dimnames(sparse_train)[[2]]
importance_matrix <- xgb.importance(feature_name,
                                    model = bst
)
pdf("./resultats/variableImportance_xgboost.pdf") 
print(
  xgb.plot.importance(importance_matrix[1:20, ])
)
dev.off()


