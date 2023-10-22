
X <- matrix(0, nrow = 2, ncol = 3)
X[1,] <- c(1.1, 0.8, 0.7)
X[2,] <- c(0.9, 1.0, 1.2)
std <- c(0.1, 0.2, 0.3)
log_norm1 <- dnorm(X[,1], mean = 0, sd = std[1], log = TRUE)
log_norm2 <- dnorm(X[,2], mean = 0, sd = std[2], log = TRUE)
log_norm3 <- dnorm(X[,3], mean = 0, sd = std[3], log = TRUE)
sum(log_norm1 + log_norm2 + log_norm3)


