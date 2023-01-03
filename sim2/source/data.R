P <- 5      # Num. of predictors
n <- 300
cov <- list()
for(i in 1:2){
  cov[[i]] <- runif(n,0,1)
}
cov[[3]] <- rnorm(n, 1, 1) 
cov[[4]] <- rnorm(n, -1, 2) 
cov[[5]] <- rnorm(n, 1, 0.5) 

Xpred <- do.call(cbind, cov)
Xcut <- lapply(1:dim(Xpred)[2], function(t) sort(unique(Xpred[,t]))) # e.g. unique values of predictors


MU_func <- -1+Xpred[,1]-Xpred[,2]+1*Xpred[,3]+Xpred[,4]+Xpred[,5]

Y_trt <- rbinom(n, 1, 0.8*pnorm(0.5*MU_func / (0.1*(2-Xpred[,1]-Xpred[,2])+0.25))+0.1*(Xpred[,1]+Xpred[,2])) # exposure model

M_err <- rnorm(n,0,0.1)
M_1 <- 1.5+0.9*1+1*(Xpred[,1])+1.5*Xpred[,2]+1*abs(Xpred[,3]+1) + 1*1*Xpred[,4] - 0.5*Xpred[,5]+M_err
M_0 <- 1.5+0.9*0+1*(Xpred[,1])+1.5*Xpred[,2]+1*abs(Xpred[,3]+1) + 0*1*Xpred[,4] - 0.5*Xpred[,5]+M_err
M_out <- ifelse(Y_trt==1, M_1, M_0)

S <- M_1-M_0

Y_err <- rnorm(n, 0, 0.3)
Y_1 <- MU_func-1.5*1*(M_1-M_0)+1.5*1*Xpred[,3]+Y_err
Y_0 <- MU_func-1.5*0*(M_1-M_0)+1.5*0*Xpred[,3]+Y_err

Y_out <- ifelse(Y_trt==1, Y_1, Y_0)


### Initial Setup (priors, initial values and hyper-parameters)
p.grow <- 0.28            # Prob. of GROW
p.prune <- 0.28           # Prob. of PRUNE
p.change <- 0.44          # Prob. of CHANGE
#m <- 100                  # Num. of Trees: default setting 100
sigma2_m <- var(Y_out)       # Initial value of SD^2
sigma2_y <- var(Y_out) 

nu <- 3                   # default setting (nu, q) = (3, 0.90) from Chipman et al. 2010
f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2_y)
lambda_y <- rootSolve::uniroot.all(f, c(0.1^5,10))
f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2_m)
lambda_m <- rootSolve::uniroot.all(f, c(0.1^5,10))

sigma2 <- 1
f <- function(lambda) invgamma::qinvgamma(0.90, nu/2, rate = lambda*nu/2, lower.tail = TRUE, log.p = FALSE) - sqrt(sigma2)
lambda <- rootSolve::uniroot.all(f, c(0.1^5,10))

alpha <- 0.95             # alpha (1+depth)^{-beta} where depth=0,1,2,...
beta <- 2                 # default setting (alpha, beta) = (0.95, 2)


