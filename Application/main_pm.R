#------ required libraries
library(Rcpp)
library(MCMCpack)
library(rootSolve)

#------ load MCMC c++ code
rm(list=ls())
sourceCpp("src/MCMC_main_subset.cpp", rebuild = T)

#------ load the dataset (for a 50km radius analysis)
load("source/Master2014_PM25_50km_annual.RData")

#------ Set Treatment (TRT), Outcome (Y), Mediators (M) and Covariates (X)
Data <- Master
Data <- subset(Data, !is.na(PM25))
Data <- subset(Data, !is.na(RHUM))
Data <- subset(Data, !is.na(Temperature))
Data <- subset(Data, !is.na(APCP))
Data <- subset(Data, totHeatInput!=0)
Data <- subset(Data, totLoad!=0) # this process automatically excludes power plants with totHeatInput==0
Data <- subset(Data, !is.na(Sulfur_Content))
Data <- subset(Data, !is.na(pctCapacity))

Y <- Data$PM25
Trt <- ifelse(Data$SO2.sc < 0.5, 0, 1)
M <- log(Data$totSO2emissions/50)
X <- cbind(log(Data$totHeatInput), Data$Sulfur_Content,Data$nunits,
           Data$RHUM, Data$Temperature, Data$APCP, log(Data$Population),  
           Data$Phase2, log(Data$totNOxemissions_pre), log(Data$totCO2emissions_pre),
           log(Data$totOpTime), log(Data$totLoad),Data$pctCapacity, Data$NumNOxControls,
           Data$C00_1, Data$C01_1, Data$C02_1, Data$C03_1, Data$C04_1, Data$C05_1, Data$C06_1, Data$C07_1, Data$C08_1, Data$C09_1)

#------ PS estimation
PS.fit <- glm(Trt~X, family=binomial())
PS <- predict(PS.fit, type="response")

P <- dim(X)[2] #<--------- Num. of Covariates
n <- dim(X)[1] #<--------- Num. of Observations


#------ MCMC settings
n.iter=100000; 

nu <- 3    # default setting (nu, q) = (3, 0.90) from Chipman et al. 2010
m <- 150                  # Num. of trees
p.grow <- 0.28            # Prob. of GROW
p.prune <- 0.28           # Prob. of PRUNE
p.change <- 0.44          # Prob. of CHANGE

sigma2_m <- var(M-mean(M))        # Initial value of SD^2
sigma2_y <- var(Y-mean(Y))

f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2_y)
lambda_y <- rootSolve::uniroot.all(f, c(0.1^5,10))

f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2_m)
lambda_m <- rootSolve::uniroot.all(f, c(0.1^5,10))

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
rcpp = MCMC(X, X[,c(1:2,8,12:13,15)], X[,c(4:7,15:16)], 
            Trt, M, Y, as.numeric(PS), p.grow, p.prune, p.change, m, 50, m, 50, 
            nu, lambda_m, lambda_y, alpha, beta,  n.iter, sigma_mu_m_tau_sigma, 
            sigma_mu_m_mu_sigma, sigma_mu_y_tau_sigma, sigma_mu_y_mu_sigma)

sd(exp(rowMeans(rcpp$predicted_M1))*50-exp(rowMeans(rcpp$predicted_M0))*50)*0.2
#[1] 159.7534
sd(exp(rowMeans(rcpp$predicted_M1))*50-exp(rowMeans(rcpp$predicted_M0))*50)*0.5
#[1] 798.7671
quant <- c(-Inf,-798, -320, 0, 320, 798, Inf)


temp.ind <- list()
len <- matrix(nrow=6, ncol=5000)
YE <- matrix(nrow=6, ncol=5000)
ME <- matrix(nrow=6, ncol=5000)
for(i in 1:5000){
  for(j in 1:6){
    temp.ind[[j]] <- which(exp(rcpp$predicted_M1[, i])*50-exp(rcpp$predicted_M0[, i])*50 >= quant[j] & exp(rcpp$predicted_M1[, i])*50-exp(rcpp$predicted_M0[, i])*50 < quant[j+1])
    len[j,i] <- length(temp.ind[[j]])
    ME[j,i] <- exp(mean(rcpp$predicted_M1[temp.ind[[j]],i]))*50-exp(mean(rcpp$predicted_M0[temp.ind[[j]],i]))*50
    YE[j,i] <- mean(rcpp$predicted_Y[temp.ind[[j]],i])
  }
}


rowMeans(ME)
apply(ME, 1, function(x) sort(x)[5000*0.025])
apply(ME, 1, function(x) sort(x)[5000*0.975])
rowMeans(YE)
apply(YE, 1, function(x) sort(x)[5000*0.025])
apply(YE, 1, function(x) sort(x)[5000*0.975])

