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
	#convert date field to separate columns
	dateOpen = as.POSIXlt(myData$Open.Date, format="%m/%d/%Y")
	myData$day = day(dateOpen)
	myData$month = month(dateOpen)
	myData$year = year(dateOpen)
	myData$year = as.numeric(myData$year) / 1000
  
  currentTime = as.POSIXlt("2015-01-01")
  #timeElapsed = currentTime - dateOpen
	myData$Open.Date = NULL

	#Consolidate Cities
	myData$City                                      <- as.character(myData$City)
	myData$City[myData$City.Group == "Other"]        <- "Other"
	# myData$City[myData$City == unique(myData$City)[4]] <- unique(myData$City)[2]
	myData$City.Group                                <- NULL
  
  	#expand factor features 
	# typeInd = model.matrix( ~ Type - 1, data=myData )
	cityInd = model.matrix( ~ City - 1, data=myData )
	
	myData = cbind(cityInd, myData)
	myData$Type = NULL
	myData$City = NULL

	#Log Transform P Variables and Revenue
	myData[, paste("P", 1:37, sep="")] <- log(1 +myData[, paste("P", 1:37, sep="")])
  
  revenue = myData$revenue
	myData$revenue = NULL
	myData$revenue = revenue
	myData$revenue <- log(myData$revenue)

	return(myData)
}
library(lubridate)
myData = featureEngineering(myData)

myFeatureSelection <- function(n.train, myData){
  x <- myData[1:n.train,"revenue"]
  y <- myData[1:n.train,c(1:45)]
  corResult = cor(y, x)
  plot(1:45, corResult)
  df = as.data.frame(corResult)
  df = cbind(df, rownames(df))
  df$V1 = abs(df$V1)
  
  return(df$V1 > 0.1)
}
importanceArray <- myFeatureSelection(n.train, myData)
  
library(Boruta)
featureSelectionBoruta <-  function(myData, n.train){
	important <- Boruta(revenue~., data=myData[1:n.train, ])
	plot(important)
	important$finalDecision[important$finalDecision != "Rejected"]
	return(important)
}
important = featureSelectionBoruta(myData, n.train)
importanceArray = important$finalDecision != "Rejected"


trainData = myData[1:n.train, c(importanceArray, T)]
train_y = trainData[,dim(trainData)[2]]
testData = myData[-c(1:n.train), c(importanceArray, F)]


