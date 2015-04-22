library(doMC)
registerDoMC(cores=4)
library(compiler)
#read in datasets 
setwd("/Users/skywalkerhunter/Downloads/kaggle/Restaurant Revenue Prediction") 
enableJIT(1)
set.seed(123) 
formula = revenue ~ .

options(digits=16)

#aggregate(. ~ index, data = rbind(data1,data2), mean)

my.csv.dir = "meta/"
files     <- list.files(my.csv.dir, full.names = T)
data.list <- lapply(files, read.csv, colClasses = c("integer", "numeric"))
data.cat  <- do.call(cbind, data.list)
IdColumn = data.cat$Id
data.cat$Id = NULL
data.cat$Id = NULL
data.cat$Id = NULL
data.cat$Id = NULL
data.cat$Id = NULL
data.agg <- rowMeans(data.cat, na.rm = FALSE, dims = 1)

#write results 
submit <- as.data.frame(cbind(IdColumn, data.agg)) 
colnames(submit) <- c("Id", "Prediction") 
write.csv(submit, "rsub_ensemble_v4.csv", row.names=FALSE, quote=FALSE) 


