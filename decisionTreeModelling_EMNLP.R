# Code and data used to train conditional inference tree classifier on data from
# speakers of US and NZ English, in an attempt to replicate human
# classifications from experimental data. 

# I'll do my best to comment the code throughly as I go through. Please direct 
# any questions to me at rctatman@uw.edu.

# getting set up ----
# load in libraries
library(partykit)
library(ggplot2)
library(vecsets)
library(caret)
library(plyr)
library(phonR)
library(vowels)
library(e1071)
library(gplots)

# load in data 
data <- read.table(file= "formantValues.csv",
                   sep = "\t", header = T)
data2 <- read.table(file= "formantValuesUS.csv",
                    sep = "\t", header = T)

# check to make sure everything looks good
head(data)
head(data2)

# gets rid of F1 and F2 measures more than two sd from the mean of F1 and F2 for
# each vowel. Requires columns named "vowel", "F1" and "F2
stripOutliers <- function(inputData){
  vowels <- levels(inputData$vowel)
  trimmedMesaures <- inputData[0,]
  for(i in 1:length(vowels)){
    vowelMeasure <- inputData[inputData$vowel == vowels[i],]
    rangeF1 <- c(mean(vowelMeasure$F1) - 1*sd(vowelMeasure$F1),
                 mean(vowelMeasure$F1) + 1*sd(vowelMeasure$F1))
    rangeF2 <- c(mean(vowelMeasure$F2) - 1*sd(vowelMeasure$F2),
                 mean(vowelMeasure$F2) + 1*sd(vowelMeasure$F2))
    whereNormedVowels <-  findInterval(vowelMeasure$F1, rangeF1) == 1 & 
      findInterval(vowelMeasure$F2, rangeF2) == 1
    trimmedMeasuresAdd <- vowelMeasure[whereNormedVowels, ]
    trimmedMesaures <- rbind(trimmedMesaures, trimmedMeasuresAdd)
  }
  return(trimmedMesaures)
}

# remove outliers. Formant measures for this data were taken automatically, so
# we're doing this to get rid of any egrigious formant-tracking errors.
data <- stripOutliers(data)
colnames(data2) <- c("soundname", "vowel", "F1", "F2", "F3", "speaker")
data2 <- stripOutliers(data2)
colnames(data2) <- c("soundname", "intervalname", "F1", "F2", "F3", "speaker")

# combine two data files, but only look at vowels we're interested in
selectedData <- rbind(data[data$intervalname == "heed",],
                      data[data$intervalname == "hid",],
                      data[data$intervalname == "head",],
                      data[data$intervalname == "had",],
                      data2)

# clean up file names. Speakers Y22 and Y23 are both from NZ. 
selectedData$soundname <- factor(strtrim(selectedData$soundname, 3 ),
                                 levels = c("Y22", "Y23" , "US" ),
                                 labels = c("Y22", "Y23" , "US"))

# now remove "heed" (since it's not confusable for human particiapants, and therefor not really of interest)
noHeed <- selectedData[selectedData$intervalname != "heed",]
noHeed$speaker <- mapvalues(noHeed$speaker, c("y22", "y23"), c("NZ", "NZ"))
selectedData$speaker <- mapvalues(selectedData$speaker, c("y22", "y23"), c("NZ", "NZ"))

# basic decision trees ----

# this model fits really well and seems to model what we think the human participants are doing
# it does better than the same model without speaker, and won't include F3 even if given the chance
fit <- ctree(intervalname ~ F1 + F2 + speaker, data=droplevels(selectedData))
plot(fit) 
length(vintersect(predict(fit), selectedData$intervalname))/length(predict(fit)) #accuracy
confusionMatrix(droplevels(selectedData$intervalname), predict(fit)) #confusion matrix 

# prettified plot of tree
fit <- ctree(intervalname ~ F1 + F2 + speaker, data=droplevels(selectedData))
class(fit)  # different class from before

plot(fit, gp = gpar(fontsize = 10),     # font size changed to 6
     inner_panel=node_inner, terminal_panel = node_barplot,
     tp_args = list(id = F, rot = 45),
     ip_args=list(id = FALSE)
)

# data with wrong demographics ----
# 
# but what would really convince me that this
# model was preforming in a human-like way would be if it make human-like 
# mistakes. So let's generate some random data

F1Range <- range(noHeed$F1)
F2Range <- range(noHeed$F2)

# make up a fake table
soundname <- rep("name", 1000)
intervalname <- rep("word", 1000)
F1 <- runif(1000, min = F1Range[1], max = F1Range[2])
F2 <- runif(1000, min = F2Range[1], max = F2Range[2])
F3 <- rep(0.00, 1000)

# "NZ" data
speaker <- c(rep('NZ', 999), 'US') # if speaker only has one level, both predictions are the same
fakeNZ <- data.frame(soundname, intervalname, F1, F2, F3, speaker)

# "US" data
speaker <- c(rep('US', 999), 'NZ')
fakeUS <- data.frame(soundname, intervalname, F1, F2, F3, speaker)

# combined "US" and "NZ" data
speaker <- sample(c('US','NZ'), 1000, replace= T)
fake <- data.frame(soundname, intervalname, F1, F2, F3, speaker)

# uncomment to use real data mislablled instead
# fakeUS <- cbind(selectedData[,1:5], speaker = c(rep('US', dim(selectedData)[1]- 1), 'NZ'))
# fakeNZ <- cbind(selectedData[,1:5], speaker = c(rep('NZ', dim(selectedData)[1]- 1), 'NZ'))

# check to make sure the column calsses are teh same
sapply(fakeNZ[1,], class)
sapply(fakeUS[1,], class)
sapply(noHeed[1,], class)

# classify our fake data
USpredictions <- predict(fit, fakeUS)
NZpredictions <- predict(fit, fakeNZ)
predictions <- predict(fit, fake)

# confusion matrix for each data set (use with mislabelled real data from line
# 120)
# confusionMatrixHeatmap(confusionMatrix(NZpredictions, droplevels(selectedData$intervalname)))
# title(main = "All data labelled 'NZ'")
# confusionMatrixHeatmap(confusionMatrix(USpredictions, droplevels(selectedData$intervalname)))
# title(main = "All data labelled 'US'")

# confusion matrix for real data
confusionMatrix(NZpredictions, USpredictions)
confusionMatrix(USpredictions, USpredictions)

# plot of all our predictions
plot(F1, F2, col = NZpredictions)
plot(F1, F2, col = USpredictions)
qplot(rev(F2), F1, col = predictions, shape = speaker,  size = 1.5)

# NZ vowel plot (based on classifier)
plotVowels(F1, F2, NZpredictions, alpha.tokens = 0.3, 
           pch.means = NZpredictions, cex.means = 2, plot.means = T,
           plot.tokens = F, pretty = TRUE)
title("US Data")
# US vowel plot (based on classifier)
plotVowels(F1, F2, USpredictions, alpha.tokens = 0.3, 
           pch.means = USpredictions, cex.means = 2, plot.means = T,
           plot.tokens = F, pretty = TRUE)
title("NZ Data")


# vowel plot for US and NZ data together
plotVowels(F1, F2, predictions, group = speaker, 
           pch.tokens = predictions, 
           alpha.tokens = 0.3, pch.means = predictions, cex.means = 2, plot.means = T,
           plot.tokens = F, var.col.by = speaker, pretty = TRUE, 
           legend.kwd = "bottomright",
           xlim = c(3000,1600))
title("Classification \n")
# compare this to the actual vowel spaces
plotVowels(selectedData$F1, selectedData$F2, selectedData$intervalname, 
           group = selectedData$speaker, 
           pch.tokens = selectedData$intervalname, 
           alpha.tokens = 0.3, pch.means = selectedData$intervalname,
           cex.means = 2, plot.means = T,
           plot.tokens = F, var.col.by = selectedData$speaker, pretty = TRUE, 
           legend.kwd = "bottomright",
           xlim = c(3000,1600))
title("Training Tokens \n")
# it's not perfect, but it looks fiarly good! 


# so this classifier is correctly tracking human participant behaviour--in
# particular classifyingg NZ "head" tokens as US "hid" tokens

#### ConfusionMatrixHeatmap this function takes the output of the
#confusionMatrix function from the caret package and plots the table portion in
#the plot object with a heatmap. Default colors are red for high densitity and
#blue for low density.
confusionMatrixHeatmap <- function(matrix, startColor = "lightblue", endColor = "red3"){ 
  my.colors<-colorRampPalette(c(startColor, endColor))   
  color.df<-data.frame(COLOR_VALUE=seq(min(matrix$table),max(matrix$table),1),                	color.name=my.colors(max(matrix$table)+1))
  a <- matrix$table + 1 #allows plotting when the matrix contains a 0  
  heatColors <- matrix(color.df[a[],2], nrow = dim(matrix$table)[1], ncol = dim(matrix$table)[2])  
  textplot(matrix$table, col.data = heatColors)
} 


                
