# Set working directory
setwd("./")

pca <- function(rawdata, fileout){
  
  # Set up grids
  epsilon <- seq(from=1,to=ncol(rawdata),by=1)
  time <- 1:nrow(rawdata)
  
  # Smooth data at each scale
  smoothdata <- apply(rawdata,2,function(x) supsmu(time,x,bass=5)$y)
  
  # Calculate principal components
  prcomps <- prcomp(smoothdata,center=TRUE,scale=TRUE)
  
  # Smooth time series at each scale
  prcomps$x <- apply(prcomps$x,2,function(x) supsmu(time,x)$y)
  
  # Extract time series
  ts <- prcomps$x[,1]
  
  # Create a dataframe for writing
  ts_df <- data.frame(x=time, value=ts)
  
  # Write time series to a file
  write.csv(ts_df, fileout, row.names = FALSE)
  
  return
}

print("running...")

args = commandArgs(trailingOnly=TRUE)
current_file = args[1]
fileout_name = args[2]

print("Performing PCA on ")
print(current_file)

# Load CROCKER
crocker <- read.csv(current_file, header=FALSE)
crocker <- data.matrix(crocker)

#truncate to first 900 timesteps
trunc_crocker = head(crocker, 900)

# Run PCA
pca(trunc_crocker, fileout_name)

print("Saved PCA results to ") 
print(fileout_name)

print("done")
