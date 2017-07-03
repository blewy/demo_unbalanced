library(caret) 
library(ggplot2) 
library(grid) 
library(gridExtra) 
library(unbalanced) 
library(tidyverse)
  
set.seed(2969) 
imbal_train <- twoClassSim(100, intercept = -13, linearVars = 0) 
prop.table(table(imbal_train$Class))


recode<- function(y.data) {
  ifelse(y.data=='Class2',1,0) 
}

decode<- function(y.coded_data) {
  ifelse(y.coded_data==1,'Class2','Class1') 
}


# Set shape by cond 
p<- ggplot(imbal_train, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Original Imbalance") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p 

# Condensed Nearest Neighbor selects the subset of instances that are able to correctly classifing the
#original datasets using a one-nearest neighbor rule.
#
# Y=the response variable of the unbalanced dataset. It must be a binary factor where the majority class is coded as 0 and the minority as 1.

set.seed(2969) 
ubCNN.data<- ubCNN(imbal_train %>% select(-Class), recode(imbal_train$Class), k = 1, verbose = T)
ubCNN.data.recoded<-cbind(ubCNN.data$X,Class=decode(ubCNN.data$Y))
prop.table(table(ubCNN.data.recoded$Class))

# Set shape by cond 
p2<- ggplot(ubCNN.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Condensed Nearest Neighbor") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p2


# Edited Nearest Neighbor removes any example whose class label differs from the class of at least two of its three nearest neighbors.
# 

set.seed(2969) 
ubENN.data<- ubENN(imbal_train %>% select(-Class), recode(imbal_train$Class), k = 5, verbose = TRUE)
ubENN.data.recoded<-cbind(ubENN.data$X,Class=decode(ubENN.data$Y))
prop.table(table(ubENN.data.recoded$Class))

# Set shape by cond 
p3<- ggplot(ubENN.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Edited Nearest Neighbor") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p3


# Neighborhood Cleaning Rule modifies the Edited Nearest Neighbor method by increasing the role
# of data cleaning. Firstly, NCL removes negatives examples which are misclassified by their 3-
# nearest neighbors. Secondly, the neighbors of each positive examples are found and the ones belonging
# to the majority class are removed.
# 

ubNCL.data<- ubNCL(imbal_train %>% select(-Class), recode(imbal_train$Class), k = 3, verbose = TRUE)
ubNCL.data.recoded<-cbind(ubNCL.data$X,Class=decode(ubNCL.data$Y))
prop.table(table(ubNCL.data.recoded$Class))

# Set shape by cond 
p4<- ggplot(ubNCL.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Neighborhood Cleaning Rule") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p4


# One Side Selection is an undersampling method resulting from the application of Tomek links
# followed by the application of Condensed Nearest Neighbor.

ubOSS.data <- ubOSS(imbal_train %>% select(-Class), recode(imbal_train$Class), verbose = TRUE)
ubOSS.data.recoded<-cbind(ubOSS.data$X,Class=decode(ubOSS.data$Y))
prop.table(table(ubOSS.data.recoded$Class))


# Set shape by cond 
p5<- ggplot(ubOSS.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("One Side Selection") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p5


#The function replicates randomly some instances from the minority class in order to obtain a final
#dataset with the same number of instances from the two classes.
#

ubOver.data <- ubOver(imbal_train %>% select(-Class), recode(imbal_train$Class), k = 0, verbose=TRUE)
ubOver.data.recoded<-cbind(ubOver.data$X,Class=decode(ubOver.data$Y))
prop.table(table(ubOver.data.recoded$Class))

#If K=0: sample with replacement from the minority class until we have the same number of instances
#in each class. If K>0: sample with replacement from the minority class until we have
#k-times the orginal number of minority instances


# Set shape by cond 
p6<- ggplot(ubOver.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Over Sampling") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p6


# Function that implements SMOTE (synthetic minority over-sampling technique)
ubSMOTE.data <- ubSMOTE(imbal_train %>% select(-Class), as.factor(recode(imbal_train$Class)), perc.over = 200, k = 5, perc.under = 200, verbose = TRUE)
ubSMOTE.data.recoded<-cbind(ubSMOTE.data$X,Class=decode(ubSMOTE.data$Y))
prop.table(table(ubSMOTE.data.recoded$Class))


# Set shape by cond 
p7<- ggplot(ubSMOTE.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("SMOTE") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p7



# The function finds the points in the dataset that are tomek link using 1-NN and then removes only
# majority class instances that are tomek links.
# 

ubTomek.data <- ubTomek(imbal_train %>% select(-Class), as.factor(recode(imbal_train$Class)), verbose = TRUE)
ubTomek.data.recoded<-cbind(ubTomek.data$X,Class=decode(ubTomek.data$Y))
prop.table(table(ubTomek.data.recoded$Class))

# Set shape by cond 
p8<- ggplot(ubTomek.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Tomek Link") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))  
p8

# The function removes randomly some instances from the majority (negative) class and keeps all
# instances in the minority (positive) class in order to obtain a more balanced dataset. It allows two
# ways to perform undersampling: i) by setting the percentage of positives wanted after undersampling
# (percPos method), ii) by setting the sampling rate on the negatives, (percUnder method). For
# percPos, "perc"has to be (N.1/N * 100) <= perc <= 50, where N.1 is the number of positive and
# N the total number of instances. For percUnder, "perc"has to be (N.1/N.0 * 100) <= perc <= 100,
# where N.1 is the number of positive and N.0 the number of negative instances.
# ubUnder(X, Y, perc = 50, method = "percPos", w = NULL)


ubUnder.data <- ubUnder(imbal_train %>% select(-Class), as.factor(recode(imbal_train$Class)), perc = 50, method = "percPos", w = NULL)
ubUnder.data.recoded<-cbind(ubUnder.data$X,Class=decode(ubUnder.data$Y))
prop.table(table(ubUnder.data.recoded$Class))

# Set shape by cond 
p9<- ggplot(ubUnder.data.recoded, aes(x=TwoFactor1, y=TwoFactor2, shape=Class,color=Class)) + geom_point()+ theme(legend.position=c(1,0.2),legend.justification=c(1,1))+ggtitle("Under Sampling") + coord_cartesian(xlim = c(-5, 5),ylim = c(-5, 5))
p9


# Juntar todos os gráficos
grid.arrange(p,p2,p3,p4,p5,p6,p7,p8,p9, ncol=3,nrow=3)










