library(doMC)
registerDoMC(cores=4)
library(compiler)
#read in datasets 
setwd("/Users/skywalkerhunter/Downloads/kaggle/Restaurant Revenue Prediction") 
enableJIT(1)
set.seed(123) 
formula = revenue ~ .

train <- read.csv("data/train.csv")
test  <- read.csv("data/test.csv")

n.train <- nrow(train)

test$revenue <- 1
myData <- rbind(train, test)
rm(train, test)
 
featureEngineering <- function(myData){ 
	#Consolidate Cities
	myData$City                                      <- as.character(myData$City)
	myData$City[myData$City.Group == "Other"]        <- "Other"
	myData$City[myData$City == unique(myData$City)[4]] <- unique(myData$City)[2]
	myData$City                                      <- as.factor(myData$City)
	myData$City.Group                                <- NULL

	# myData$yy = as.numeric(myData$yy / 1000)
  
  #expand factor features 
	typeInd = model.matrix( ~ Type - 1, data=myData )
	cityInd = model.matrix( ~ City - 1, data=myData )
	
	myData = cbind(typeInd, cityInd, myData)
	myData$Type = NULL
	myData$City = NULL

	#Log Transform P Variables and Revenue
	myData[, paste("P", 1:37, sep="")] <- log(1 +myData[, paste("P", 1:37, sep="")])

	myData$revenue <- log(myData$revenue)

	return(myData)
}

myData = featureEngineering(myData)

library(Boruta)

featureSelectionBoruta <-  function(myData, n.train){
	important <- Boruta(revenue~., data=myData[1:n.train, ])

	return(important)
}

important = featureSelectionBoruta(myData, n.train)
trainData = myData[1:n.train, c(important$finalDecision != "Rejected", TRUE)]
train_y = trainData[,dim(trainData)[2]]
testData = myData[-c(1:n.train), c(important$finalDecision != "Rejected", TRUE)]
testData$revenue = NULL 

