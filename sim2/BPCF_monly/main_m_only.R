## SIM 2

library(Rcpp); library(MCMCpack); library(rootSolve)
rm(list=ls())
sourceCpp("src/MCMC_monly.cpp", rebuild = T)

num_trial = 200
n=300; P=5; m=150; n.iter=10000

YE.final <- ME.final <- matrix(nrow=200, ncol=5)

for(test_case in 1:200) {
  cat("Testing ", test_case, "of", num_trial, "at", format(Sys.time(), "%H:%M:%S"), "\n")
  # sep --------------

  source("source/data.R")
  
  f <- function(scale) qcauchy(0.75, 0, scale) - 2*sd(M_out-mean(M_out))   # first
  sigma_mu_m_mu_sigma <- uniroot.all(f, c(0.1^5, 100))
  
  f <- function(sd) qnorm(0.75, 0, sd) - sd(M_out-mean(M_out))             # second
  sigma_mu_m_tau_sigma <- uniroot.all(f, c(0.1^5, 100))
  
  f <- function(scale) qcauchy(0.75, 0, scale) - 2*sd(Y_out-mean(Y_out))   # first
  sigma_mu_y_mu_sigma <- uniroot.all(f, c(0.1^5, 100))
  
  f <- function(sd) qnorm(0.75, 0, sd) - sd(Y_out-mean(Y_out))             # second
  sigma_mu_y_tau_sigma <- uniroot.all(f, c(0.1^5, 100))
  
  PS.fit <- glm(Y_trt~Xpred, family=binomial())
  PS <- predict(PS.fit, type="response")
  
  rcpp = MCMC(Xpred, Y_trt, M_out, Y_out, as.numeric(PS), p.grow, p.prune, p.change, m, 50, m, 50, nu, lambda_m, lambda_y, alpha, beta, n.iter, sigma_mu_m_tau_sigma, sigma_mu_m_mu_sigma, sigma_mu_y_tau_sigma, sigma_mu_y_mu_sigma)
  
 
 #------ Posterior Summary
 YE <- matrix(nrow=5, ncol=500)
 ME <- matrix(nrow=5, ncol=500)
 m.interval <-  c(-5.8506229, -1.7749747, -0.6043677,  0.4056561,  1.5751424,  5.6499337)
 for(i in 1:500){
   for(j in 1:5){
     temp.u <- m.interval[j+1]
     temp.l <- m.interval[j]
     temp.ind <- which(rcpp$predicted_S[,i] >= temp.l & rcpp$predicted_S[,i] < temp.u)
     ME[j,i] <- mean(rcpp$predicted_S[temp.ind,i])
     YE[j,i] <- mean(rcpp$predicted_Y[temp.ind,i])
   }
 }
 YE.final[test_case, ] <- rowMeans(YE)
 ME.final[test_case, ] <- rowMeans(ME)
}


Y.true <- c(4.2846492,  1.7381770,  0.1491125, -1.4393298, -3.9864719)
M.true <- c(-2.85643282, -1.15878464, -0.09940835,  0.95955321,  2.65764795)

biasY <- colMeans(YE.final,na.rm=T) - Y.true
mseY <- apply((YE.final - matrix(Y.true, nrow=200, ncol=5, byrow=T))^2, 2, mean,na.rm=T)

biasM <- colMeans(ME.final,na.rm=T) - M.true
mseM <- apply((ME.final - matrix(M.true, nrow=200, ncol=5, byrow=T))^2, 2, mean,na.rm=T)

