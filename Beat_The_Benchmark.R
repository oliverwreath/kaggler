
library(caret)

#Approach Random Forest 
pp = c("center", "scale") #c("ica") 
trainCtrl = trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, returnResamp = "final") #, classProbs = TRUE)

TUNE_LEN = 10 
model <- train(revenue~. , data = trainData, tuneLength = TUNE_LEN, trControl = trainCtrl )#,tuneLength = TUNE_LEN, trControl = trainCtrl   , preProc = pp
plot(model)
model 
#0.4045332
#Make a Prediction 
prediction <- predict(model, testData) 
#Make Submission 
submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
colnames(submit) <- c("Id", "Prediction") 
write.csv(submit, "rsub_Boruta_caretRF_v4.csv", row.names=FALSE, quote=FALSE) 

#Approach tuneRF 
model <- tuneRF(trainData[,-dim(trainData)[2]], trainData[,dim(trainData)[2]], stepFactor=1.5, doBest=TRUE) 
plot(model)
#0.3917587 
#Make a Prediction 
prediction <- predict(model, testData) 
#Make Submission 
submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
colnames(submit) <- c("Id", "Prediction") 
write.csv(submit, "rsub_tuneRF_v4.csv", row.names=FALSE, quote=FALSE) 

library('extraTrees') 
model = extraTrees( as.matrix(trainData[,-dim(trainData)[2]] ), as.numeric(trainData[,dim(trainData)[2]]) ) 
plot(model)
options( java.parameters = "-Xmx20g" )
prediction = predict(model, testData) #, probability=TRUE
submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
colnames(submit) <- c("Id", "Prediction") 
write.csv(submit, "rsub_extraTrees_v4.csv", row.names=FALSE, quote=FALSE) 


library('e1071') 
##build our model
model <- svm(formula, data = trainData, cross = 3) 
summary(model) plot(model)
##run model against test set
prediction <- predict(model, testData) 
##generate output
submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
colnames(submit) <- c("Id", "Prediction") 
write.csv(submit, "rsub_SVM_e1071_v4.csv", row.names=FALSE, quote=FALSE) 

#knn
tryCatch({
	##build our model
	model <- train(formula, method = "knn", verbose = FALSE,  
	                     data = trainData, tuneLength = TUNE_LEN, trControl = trainCtrl) 
	summary(model); plot(model) 
	##get best prediction 
	prediction <- predict(model, newdata = testData)
	##generate output
	submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
	colnames(submit) <- c("Id", "Prediction") 
	write.csv(submit, "rsub_KNN_v4.csv", row.names=FALSE, quote=FALSE) 
}, error = function(err) {
  print(paste("MY_ERROR:  ", err))
})

# library(caret)
# #avnet
# tryCatch({
# 	library('nnet') 
# 	##train
# 	model <- train(formula, method = "avNNet", verbose = FALSE,  
# 	                    data = trainData, tuneLength = TUNE_LEN, trControl = trainCtrl) 
# 	summary(model); plot(model) 
# 	##predict 
# 	prediction <- predict(model, newdata = testData)
# 	##generate output
# 	submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
# 	colnames(submit) <- c("Id", "Prediction") 
# 	write.csv(submit, "rsub_avNNet_v4.csv", row.names=FALSE, quote=FALSE) 
# }, error = function(err) {
#   print(paste("MY_ERROR:  ", err))
# })

# tryCatch({
# 	library('nnet') 
# 	##train
# 	model <- train(formula, method = "nnet", verbose = FALSE,  
# 	                    data = trainData, tuneLength = TUNE_LEN, trControl = trainCtrl) 
# 	summary(model); plot(model) 
# 	##predict 
# 	prediction <- predict(model, newdata = testData)
# 	##generate output
# 	submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
# 	colnames(submit) <- c("Id", "Prediction") 
# 	write.csv(submit, "rsub_nnet_v4.csv", row.names=FALSE, quote=FALSE) 
# }, error = function(err) {
#   print(paste("MY_ERROR:  ", err))
# })

# tryCatch({
# 	library('nnet') 
# 	##train
# 	model <- train(formula, method = "dnn", verbose = T,  
# 	                    data = trainData, tuneLength = TUNE_LEN, trControl = trainCtrl) 
# 	summary(model); plot(model) 
# 	##predict 
# 	prediction <- predict(model, newdata = testData)
# 	##generate output
# 	submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
# 	colnames(submit) <- c("Id", "Prediction") 
# 	write.csv(submit, "rsub_dnn_v4.csv", row.names=FALSE, quote=FALSE) 
# }, error = function(err) {
#   print(paste("MY_ERROR:  ", err))
# })
# Error - final tuning parameters could not be determined

# #glmnet 
# tryCatch({
# 	library('glmnet') 
# 	##train
# 	fit.glmnet <- cv.glmnet(as.matrix(trainData[,1:dim(trainData)[2]-1]), train_y, family="mgaussian") 
# 	summary(fit.glmnet) 
# 	plot(fit.glmnet) 
# 	##predict 
# 	prediction <- predict(fit.glmnet, as.matrix(testData), type="response") 
# 	# predict.glmnet = max.col(matrix(predict.glmnet,dim(predict.glmnet)[1],dim(predict.glmnet)[2]))
# 	##generate output 
# 	submit <- as.data.frame(cbind(seq(0, length(prediction) - 1, by=1), exp(prediction))) 
# 	write.csv(submit, file="rsub_glmnet_v4.csv", row.names=FALSE, quote=FALSE) 
# }, error = function(err) {
#   print(paste("MY_ERROR:  ", err))
# })
# Error in mrelnet(x, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : NA/NaN/Inf in foreign function call (arg 5)\n"


