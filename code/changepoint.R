# Load libraries
library("tidyverse")
library("ggpubr")
library("scatterplot3d")
library("patchwork")
library("readr")

#install.packages("bcp")
library(bcp)

# install.packages("changepoint")
library(changepoint)

library(readxl)
library(tidyverse)

args = commandArgs(trailingOnly=TRUE)
current_file <- args[1]
type <- args[2]

if (type == "pca") {
    index <- 2
    sep <- ","
    header <- TRUE
}

if (type == "norm") {
    index <- 1
    sep <- " "
    header <- FALSE
}

rawdata <- read.csv(current_file, header=header, sep=sep)
rawdata <- data.matrix(rawdata)
#rawdata <- rawdata[!apply(rawdata==max(rawdata[,1]),1,any),]

# Perform change point analysis
mvalue = cpt.mean(rawdata[,index], penalty="None", method="BinSeg", Q=1) #mean changepoints using BinSeg
pt = cpts(mvalue)
print(mvalue)