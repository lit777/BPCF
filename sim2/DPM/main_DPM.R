# SIM 2
library(tidyverse)
library(mvtnorm)        # dmvnorm
library(MASS)           # mvrnorm
library(plyr)           # alply
library(mniw)           # riwish
library(LaplacesDemon)  # rinvgamma

### set iterations
iter <- 10000
seq <- seq(iter/2,iter,by=10)
cutoff <- 0.5


for(process in 1:200){
  tryCatch({
ii <- 0
M1M0 <- AE1 <- AE2 <- DE <- list()

### Data Preparation
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

A <- Y_trt <- rbinom(n, 1, 0.8*pnorm(0.5*MU_func / (0.1*(2-Xpred[,1]-Xpred[,2])+0.25))+0.1*(Xpred[,1]+Xpred[,2])) # exposure model

M_err <- rnorm(n,0,0.1)
M_1 <- 1.5+0.9*1+1*(Xpred[,1])+1.5*Xpred[,2]+1*abs(Xpred[,3]+1) + 1*1*Xpred[,4] - 0.5*Xpred[,5]+M_err
M_0 <- 1.5+0.9*0+1*(Xpred[,1])+1.5*Xpred[,2]+1*abs(Xpred[,3]+1) + 0*1*Xpred[,4] - 0.5*Xpred[,5]+M_err
M_out <- ifelse(Y_trt==1, M_1, M_0)

S <- M_1-M_0

Y_err <- rnorm(n, 0, 0.3)
Y_1 <- MU_func-1.5*1*(M_1-M_0)+1.5*Xpred[,3]+Y_err
Y_0 <- MU_func-1.5*0*(M_1-M_0)+1.5*Xpred[,3]+Y_err

Y_out <- ifelse(Y_trt==1, Y_1, Y_0)


### Data set
dat <- cbind(A, Xpred, M_out, Y_out)
dat <- dat %>% as_tibble %>% arrange(A) # arrange data by treatment


#####################################
##          Basic setting          ##
#####################################

mat_split <- function(M, r, c){
  nr <- ceiling(nrow(M)/r)
  nc <- ceiling(ncol(M)/c)
  newM <- matrix(NA, nr*r, nc*c)
  newM[1:nrow(M), 1:ncol(M)] <- M
  
  div_k <- kronecker(matrix(seq_len(nr*nc), nr, byrow = TRUE), matrix(1, r, c))
  matlist <- split(newM, div_k)
  N <- length(matlist)
  mats <- unlist(matlist)
  dim(mats)<-c(r, c, N)
  mats <- lapply(seq(dim(mats)[3]), function(x) mats[ , , x])
  return(mats)
}  # function which converts matrix to list

covariates <- dat[colnames(dat)[grep('V', colnames(dat))]]
num.covariates <- ncol(covariates)

trt <- dat %>% filter(A==1) 
ctrl <- dat %>% filter(A==0) 

num.trt <- trt %>% nrow()
num.ctrl <- ctrl %>% nrow()

trt_M <- trt$M_out
ctrl_M <- ctrl$M_out
names(trt_M) <- NULL
names(ctrl_M) <- NULL

trt_y <- trt$Y_out
ctrl_y <- ctrl$Y_out
names(trt_y) <- NULL
names(ctrl_y) <- NULL

trt_x <- trt[colnames(trt)[grep('V', colnames(trt))]]
ctrl_x <- ctrl[colnames(ctrl)[grep('V', colnames(ctrl))]]

H <- 30  # set the number of cluster H

idx_11_ctrl <- seq(1, 4*num.ctrl, by=4)
idx_12_ctrl <- seq(2, 4*num.ctrl, by=4)
idx_21_ctrl <- seq(3, 4*num.ctrl, by=4)
idx_22_ctrl <- seq(4, 4*num.ctrl, by=4)

idx_11_trt <- seq(1, 4*num.trt, by=4)
idx_12_trt <- seq(2, 4*num.trt, by=4)
idx_21_trt <- seq(3, 4*num.trt, by=4)
idx_22_trt <- seq(4, 4*num.trt, by=4)


xx <- seq(2*n)

I2 <- diag(2)
In <- diag(n)
s <- 20   # large variance for beta_d, beta_y


#####################################
##        set initial value        ##
#####################################

# output
output <- list()
potential_M_1 <- list()
potential_M_0 <- list()


# assign random indicator and cluster
indicator <- sample(seq(1,H), n, replace=TRUE)
cluster <- table(factor(indicator, levels = 1:H))
ctrl_indic <- indicator[1:num.ctrl]
trt_indic <- indicator[-(1:num.ctrl)]


# initial eta and Sigma
eta <- mvrnorm(H, mu=c(0,0), Sigma=10*diag(2))
Sig <- alply(riwish(H, 10*I2, 6), 3)


# initial beta_M
beta_M <- mvrnorm(1, mu=rep(0,2*num.covariates),
                  Sigma=s^2*diag(2*num.covariates))


# initial beta_y
beta_y_0 <- mvrnorm(1, 
                    mu=rep(0,3+num.covariates), 
                    Sigma=s^2*diag(3+num.covariates))
beta_y_1 <- mvrnorm(1, 
                    mu=rep(0,3+num.covariates), 
                    Sigma=s^2*diag(3+num.covariates))


# initial alpha and w(weight)
alpha <- rgamma(1, 1, 1)
w <- rbeta(H-1, 1, alpha)
w_ <- 1-w
w_ <- cumprod(append(w_, 1, after=0))
w <- append(w, 1, after=H-1)
w <- w*w_

# initial sig_0 and sig_1
sig_0 <- rinvgamma(1, shape = 1, scale = 1)
sig_1 <- rinvgamma(1, shape = 1, scale = 1)


#####################################
##          Gibbs Sampling         ##
#####################################

for (i in 1:iter) {

  unlist_Sig_ctrl <- unlist(Sig[ctrl_indic])
  unlist_Sig_trt <- unlist(Sig[trt_indic])
  
  X.beta_M_1 <- covariates %>% 
    as.matrix %>% t %>% 
    crossprod(beta_M[-(1:num.covariates)])
  
  X.beta_M_0 <- covariates %>% 
    as.matrix %>% t %>% 
    crossprod(beta_M[1:num.covariates])
  
  X.beta_y_0 <- ctrl_x %>% 
    as.matrix %>% t %>% 
    crossprod(beta_y_0[4:(3+num.covariates)])
  
  X.beta_y_1 <- trt_x %>% 
    as.matrix %>% t %>% 
    crossprod(beta_y_1[4:(3+num.covariates)])
  
  
  # update potential_M_1 for control group
  # (update all in one)
  v1 <- unlist_Sig_ctrl[idx_22_ctrl] - 
    unlist_Sig_ctrl[idx_21_ctrl]^2 / unlist_Sig_ctrl[idx_11_ctrl]
  
  mu1 <- eta[ctrl_indic,2] + X.beta_M_1[1:num.ctrl,] + 
    unlist_Sig_ctrl[idx_21_ctrl] / unlist_Sig_ctrl[idx_11_ctrl] * 
    (ctrl_M - eta[ctrl_indic,1] - X.beta_M_0[1:num.ctrl,])
  
  v2 <- sig_0 / beta_y_0[3]^2
  mu2 <- (ctrl_y - beta_y_0[1] - beta_y_0[2]*ctrl_M - X.beta_y_0) / beta_y_0[3]
  
  v <- (v1 * v2) / (v1 + v2)
  mu <- v*(mu1/v1 + mu2/v2)
  
  potential_M_1[[i]] <- rnorm(num.ctrl, mu, sd=sqrt(v))
  
  
  # update potential_M_0 for treatment group
  # (update all in one)
  v1 <- unlist_Sig_trt[idx_11_trt] - 
    unlist_Sig_trt[idx_12_trt]^2 / unlist_Sig_trt[idx_22_trt]
  
  mu1 <- eta[trt_indic,1] + X.beta_M_0[-(1:num.ctrl),] + 
    unlist_Sig_trt[idx_12_trt] / unlist_Sig_trt[idx_22_trt] * 
    (trt_M - eta[trt_indic,2] - X.beta_M_1[-(1:num.ctrl),])
  
  v2 <- sig_1 / beta_y_1[2]^2
  mu2 <- (trt_y - beta_y_1[1] - beta_y_1[3]*trt_M - X.beta_y_1) / beta_y_1[2]
  
  v <- (v1 * v2) / (v1 + v2)
  mu <- v*(mu1/v1 + mu2/v2)
  
  potential_M_0[[i]] <- rnorm(num.trt, mu, sd=sqrt(v))
  
  potential_M <- cbind(c(ctrl_M, potential_M_0[[i]]), c(potential_M_1[[i]], trt_M))
  
  
  # update indicator
  # (update by each observations)
  new.indicator <- indicator
  X.beta_M <- cbind(X.beta_M_0, X.beta_M_1)
  
  for (j in 1:n) {
    prop <- NULL
    for (k in 1:H) {
      prop[k] <- dmvnorm(potential_M[j,], 
                         mean=(eta[k,] + X.beta_M[j,]), 
                         sigma=Sig[[k]])
    }
    
    if (sum(prop==0)!=H) {
      prop.w <- prop*w
      if (sum(prop.w==0)!=H) {
        new.indicator[j] <- which(rmultinom(1, 1, prop.w)==1)
      } else {
        prop <- w
        new.indicator[j] <- which(rmultinom(1, 1, prop)==1)
      }
    } else {
      prop <- w
      new.indicator[j] <- which(rmultinom(1, 1, prop)==1)
    }
    
  }
  
  
  indicator <- new.indicator
  cluster <- table(factor(indicator, levels = 1:H))
  ctrl_indic <- indicator[1:num.ctrl]
  trt_indic <- indicator[-(1:num.ctrl)]
  
  
  # update w
  ww <- as.vector(cluster)
  w.prime <- rbeta(H-1, 1 + ww[-H], alpha + rev(cumsum(rev(ww)))[-1])
  w_ <- 1-w.prime
  cumprod.w_ <- cumprod(append(w_, 1, after=0))
  w.prime_ <- append(w.prime, 1, after=H-1)
  w <- w.prime_*cumprod.w_
  
  
  # update alpha
  w_[which(w_==0)] <- 1e-10
  alpha <- rgamma(1, 1 + H - 1, rate = 1-sum(log(w_)))
  
  # update eta and Sigma
  new.eta <- eta
  new.Sig <- Sig
  inv.Sig <- Map('solve', new.Sig)
  
  # 1. update eta and Sigma which contain no observation
  if (!is_empty(which(cluster==0))) {
    
    new.eta[which(cluster==0),] <- mvrnorm(sum(cluster==0), 
                                           mu=c(0,0), 
                                           Sigma=10*diag(2))
    new.Sig[which(cluster==0)] <- alply(riwish(sum(cluster==0), 
                                               10*I2, 6), 3)
  }
  
  V <- Map('solve', Map('+', rep(list(I2/10), H), 
                        Map('*', inv.Sig, as.list(cluster))))
  
  # 2. update eta and Sigma which contain observations
  for (j in which(cluster!=0)) {
    
    X.beta_M <- cbind(X.beta_M_0[which(indicator==j),], 
                      X.beta_M_1[which(indicator==j),])
    
    mu.component <- colSums(potential_M[which(indicator==j),] - X.beta_M)
    
    mu <- V[[j]] %*% (c(0,0)/10 + 
                        inv.Sig[[j]] %*% mu.component)
    
    new.eta[j,] <- mvrnorm(1, mu=mu, Sigma=V[[j]])
    
    scale.component <- potential_M[which(indicator==j),] - 
      matrix(new.eta[j,], ncol=2, nrow=cluster[j], byrow=TRUE) - 
      X.beta_M
    
    new.Sig[[j]] <- mniw::riwish(1, 
                                 10*I2 + crossprod(scale.component), 
                                 6 + cluster[j])
    
  }
  
  eta <- new.eta
  Sig <- new.Sig
  inv.Sig <- Map('solve', Sig)
  
  
  # update beta_M
  V <- diag(2*num.covariates)/s^2
  
  x1 <- matrix(0, 2*n, num.covariates)
  x2 <- matrix(0, 2*n, num.covariates)
  
  x1[xx%%2==1,] <- as.matrix(covariates)
  x2[xx%%2==0,] <- as.matrix(covariates)
  
  X_ <- cbind(x1, x2)
  X_ <- mat_split(X_, 2, 2*num.covariates)
  t_X_ <- Map('t', X_)
  
  V.component <- Reduce('+', 
                        Map('%*%', 
                            Map('%*%', 
                                t_X_, inv.Sig[indicator]), X_))
  V <- solve(V + V.component)
  
  mu.component <- potential_M - eta[indicator,]
  mu.component <- split(t(mu.component), rep(1:n, each=2))
  
  mu <- V%*%Reduce('+', Map('%*%', 
                            Map('%*%', 
                                t_X_, inv.Sig[indicator]), mu.component))
  
  beta_M <- mvrnorm(1, mu=mu, Sigma=V)
  
  
  # update beta_y_0
  ctrl.X_ <- cbind(rep(1, num.ctrl), potential_M[1:num.ctrl,], ctrl_x) %>% as.matrix
  colnames(ctrl.X_) <- NULL
  V <- solve(crossprod(ctrl.X_)/sig_0 + diag(ncol(ctrl.X_))/s^2)
  mu <- V %*% t(ctrl.X_)%*%ctrl_y/sig_0
  
  beta_y_0 <- mvrnorm(1, mu=mu, Sigma=V)

  
  # update beta_y_1
  trt.X_ <- cbind(rep(1, num.trt), potential_M[-(1:num.ctrl),], trt_x) %>% as.matrix
  colnames(trt.X_) <- NULL
  V <- solve(crossprod(trt.X_)/sig_1 + diag(ncol(trt.X_))/s^2)
  mu <- V %*% t(trt.X_)%*%trt_y/sig_1
  
  beta_y_1 <- mvrnorm(1, mu=mu, Sigma=V)

  
  # update sig_0 and sig_1
  aa <- 1 + num.ctrl/2
  bb <- 1 + sum((ctrl_y-ctrl.X_%*% beta_y_0)^2)/2
  sig_0 <- LaplacesDemon::rinvgamma(1, shape = aa, scale = bb)
  
  aa <- 1 + num.trt/2
  bb <- 1 + sum((trt_y-trt.X_%*% beta_y_1)^2)/2
  sig_1 <- LaplacesDemon::rinvgamma(1, shape = aa, scale = bb)
  
  potential_y_1 <- rnorm(num.ctrl, mean=ctrl.X_%*%beta_y_1, sd=sqrt(sig_1))
  potential_y_0 <- rnorm(num.trt, mean=trt.X_%*%beta_y_0, sd=sqrt(sig_0))
  
  potential_Y <- cbind(c(ctrl_y, potential_y_0), c(potential_y_1, trt_y))
  
  if(i %in% seq){
    ii <- ii + 1
    output[[ii]] <- cbind(potential_Y, potential_M)
#    S <- output.temp[,4]-output.temp[,3]
#    M1M0[[ii]] <- quantile(S, seq(0, 1, length=6))
#    DE[[ii]] <- (output.temp[,2]-output.temp[,1])[S.de]
#    AE1[[ii]] <- (output.temp[,2]-output.temp[,1])[S >= cutoff]
#    AE2[[ii]] <- (output.temp[,2]-output.temp[,1])[S <= -cutoff]
    colnames(output[[ii]]) <- c('Y_0', 'Y_1', 'M_0', 'M_1')
  }
  print(i)
}
save(Y_1, Y_0, M_1, M_0, A,output, file=paste0("Sim4_DP_",process,".RData"))
  }, error=function(e){})
}

















