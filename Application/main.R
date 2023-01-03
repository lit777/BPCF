#------ required libraries
library(Rcpp)
library(MCMCpack)
library(rootSolve)

#------ load MCMC c++ code
rm(list=ls())
sourceCpp("src/MCMC_main.cpp", rebuild = T)

#------ load the dataset (for a 150km radius analysis)
load("source/Master2014_150.RData")
#load("source/Master2014_100.RData") # for a 100km radius analysis
#load("source/Master2014_50.RData")  # for a 50km radius analysis

#------ Set Treatment (TRT), Outcome (Y), Mediators (M) and Covariates (X)
Data <- Master
Data <- subset(Data, !is.na(PM.2.5))
Y <- Data$PM.2.5
Trt <- Data$SO2.SC
M <- log(Data$SO2_Annual)
X <- cbind(Data$S_n_CR, Data$NumNOxControls, log(Data$Heat_Input), Data$RHUM, Data$Temperature, Data$APCP, log(Data$Population),  Data$PctCapacity/100, Data$Sulfur_Content, Data$Phase2_Indicator, log(Data$Operating_Time))

P <- dim(X)[2] #<--------- Num. of Covariates
n <- dim(X)[1] #<--------- Num. of Observations


#------ MCMC settings
n.iter=10000; n.iter_ps=5000
epsilon <- 0.1 # for sentivity analyses, use 0.25, 0.5, 0.75

nu <- 3    # default setting (nu, q) = (3, 0.90) from Chipman et al. 2010
m <- 200                  # Num. of trees
p.grow <- 0.28            # Prob. of GROW
p.prune <- 0.28           # Prob. of PRUNE
p.change <- 0.44          # Prob. of CHANGE
  
sigma2_m <- var(M)        # Initial value of SD^2
sigma2_y <- var(Y)
sigma2 <- 1

f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2_y)
lambda_y <- rootSolve::uniroot.all(f, c(0.1^5,10))

f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2_m)
lambda_m <- rootSolve::uniroot.all(f, c(0.1^5,10))
  
f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2)
  lambda <- rootSolve::uniroot.all(f, c(0.1^5,10))
  
alpha <- 0.95             # alpha (1+depth)^{-beta} where depth=0,1,2,...
beta <- 2                 # default setting (alpha, beta) = (0.95, 2)
  
  
f <- function(scale) qcauchy(0.75, 0, scale) - 2*sd(M-mean(M))
sigma_mu_m_mu_sigma <- uniroot.all(f, c(0.1^5, 100))
f <- function(sd) qnorm(0.75, 0, sd) - sd(M-mean(M))
sigma_mu_m_tau_sigma <- uniroot.all(f, c(0.1^5, 100))
f <- function(scale) qcauchy(0.75, 0, scale) - 2*sd(Y-mean(Y))
sigma_mu_y_mu_sigma <- uniroot.all(f, c(0.1^5, 100))
f <- function(sd) qnorm(0.75, 0, sd) - sd(Y-mean(Y))
sigma_mu_y_tau_sigma <- uniroot.all(f, c(0.1^5, 100))
  
#------ Main MCMC running
rcpp = MCMC(X, Trt, M, Y, seq(0.05,0.95,by=0.05), p.grow, p.prune, p.change, m, m, 50, m, 50, nu, lambda, lambda_m, lambda_y, alpha, beta, n.iter_ps, n.iter, sigma_mu_m_tau_sigma, sigma_mu_m_mu_sigma, sigma_mu_y_tau_sigma, sigma_mu_y_mu_sigma)

#------ Posterior Summary
DE <- AE1 <- AE2 <- M1M0 <- NULL
DE.len <- AE1.len <- AE2.len <- NULL
for(i in 1:500){
    DE[i] <- mean(rcpp$predicted_Y[abs(rcpp$predicted_S[,i])< epsilon,i])
    AE2[i] <- mean(rcpp$predicted_Y[(rcpp$predicted_S[,i])< -epsilon,i])
    AE1[i] <- mean(rcpp$predicted_Y[(rcpp$predicted_S[,i])> epsilon,i])
    DE.len[i] <- length(which(abs(rcpp$predicted_S[,i])< epsilon))
    AE1.len[i] <- length(which((rcpp$predicted_S[,i])> epsilon))
    AE2.len[i] <- length(which((rcpp$predicted_S[,i])< -epsilon))
}
  
#save.image("150_0.1.RData")


