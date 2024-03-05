## SIM 1

library(Rcpp); library(MCMCpack); library(rootSolve)
rm(list=ls())
sourceCpp("src/MCMC_bart.cpp", rebuild = T)

num_trial = 200
n=300; P=7; m=150; n.iter=10000; n.iter_ps=5000

YE.final <- ME.final <- matrix(nrow=200, ncol=5)

for(test_case in 1:200) {
  cat("Testing ", test_case, "of", num_trial, "at", format(Sys.time(), "%H:%M:%S"), "\n")
  # sep --------------

  source("source/data.R")

  
  PS.fit <- glm(Y_trt~Xpred, family=binomial())
  PS <- predict(PS.fit, type="response")
  
  rcpp = MCMC(Xpred, Y_trt, M_out, Y_out, PS, p.grow, p.prune, p.change, m, nu, lambda_m, lambda_y, alpha, beta,  n.iter)
  
  #------ Posterior Summary
  YE <- matrix(nrow=5, ncol=500)
  ME <- matrix(nrow=5, ncol=500)
  m.interval <-  c(2.004158, 2.256113, 2.525802, 2.841687, 3.278662, 5.090102)
  
  SS <- rcpp$predicted_M1-rcpp$predicted_M0
  YY <- rcpp$predicted_Y1-rcpp$predicted_Y0
  for(i in 1:500){
    for(j in 1:5){
      temp.u <- m.interval[j+1]
      temp.l <- m.interval[j]
      temp.ind <- which(SS[,i] >= temp.l & SS[,i] < temp.u)
      ME[j,i] <- mean(SS[temp.ind,i])
      YE[j,i] <- mean(YY[temp.ind,i])
    }
  }
  YE.final[test_case, ] <- rowMeans(YE)
  ME.final[test_case, ] <- rowMeans(ME)
}

Y.true <- c(-4.539873,  -5.711521,  -7.180378,  -9.278763, -14.108190)
M.true <- c(2.129458, 2.388615, 2.678080, 3.043537, 3.736845)

biasY <- colMeans(YE.final,na.rm=T) - Y.true
mseY <- apply((YE.final - matrix(Y.true, nrow=200, ncol=5, byrow=T))^2, 2, mean,na.rm=T)

biasM <- colMeans(ME.final,na.rm=T) - M.true
mseM <- apply((ME.final - matrix(M.true, nrow=200, ncol=5, byrow=T))^2, 2, mean,na.rm=T)

