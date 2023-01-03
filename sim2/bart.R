## SIM 2

library(Rcpp); library(MCMCpack); library(rootSolve)
rm(list=ls())
sourceCpp("src/MCMC_bart.cpp", rebuild = T)

num_trial = 200
n=300; P=5; m=150; n.iter=10000; n.iter_ps=5000

rcpp_sep = vector(mode = "numeric", length = num_trial)

DE.final <- AE1.final <- AE2.final <- M1M0 <- NULL
CI.DE <- CI.AE1 <- CI.AE2 <- CI.M1M0 <- NULL

for(test_case in 1:200) {
  cat("Testing ", test_case, "of", num_trial, "at", format(Sys.time(), "%H:%M:%S"), "\n")
  # sep --------------

  source("source/data.R")

  
  rcpp = MCMC(Xpred, Y_trt, M_out, Y_out, seq(0.05,0.95,by=0.05), p.grow, p.prune, p.change, m, nu, lambda, lambda_m, lambda_y, alpha, beta, n.iter_ps, n.iter)
  
  DE <- AE1 <- AE2 <- NULL
  for(i in 1:500){
    DE[i] <- mean(rcpp$predicted_Y1[abs(rcpp$predicted_M1[,i]-rcpp$predicted_M0[,i])<1,i]-rcpp$predicted_Y0[abs(rcpp$predicted_M1[,i]-rcpp$predicted_M0[,i])<1,i])
    AE2[i] <- mean(rcpp$predicted_Y1[(rcpp$predicted_M1[,i]-rcpp$predicted_M0[,i])< -1,i]-rcpp$predicted_Y0[(rcpp$predicted_M1[,i]-rcpp$predicted_M0[,i])< -1,i])
    AE1[i] <- mean(rcpp$predicted_Y1[(rcpp$predicted_M1[,i]-rcpp$predicted_M0[,i])> 1,i]-rcpp$predicted_Y0[(rcpp$predicted_M1[,i]-rcpp$predicted_M0[,i])> 1,i])
  }
  
  DE.final[test_case] <- mean(DE)
  AE1.final[test_case] <- mean(AE1)
  AE2.final[test_case] <- mean(AE2)
  M1M0[test_case] <- mean(rowMeans(rcpp$predicted_M1)-rowMeans(rcpp$predicted_M0))
  CI.de <- quantile(DE, c(0.025, 0.975))
  CI.ae1 <- quantile(AE1, c(0.025, 0.975))
  CI.ae2 <- quantile(AE2, c(0.025, 0.975))
  CI.m1m0 <- quantile(colMeans(rcpp$predicted_M1-rcpp$predicted_M0), c(0.025, 0.975))
  if(CI.de[1] < 1.51311 & CI.de[2] > 1.51311){CI.DE[test_case] <- 1}
  if(CI.ae1[1] < -1.886524 & CI.ae1[2] > -1.886524){CI.AE1[test_case] <- 1}
  if(CI.ae2[1] < 4.962542 & CI.ae2[2] > 4.962542){CI.AE2[test_case] <- 1}
  if(CI.m1m0[1] < -0.099329065 & CI.m1m0[2] > -0.09932906){CI.M1M0[test_case] <- 1}
}

#save.image("sim_bart2.RData")

