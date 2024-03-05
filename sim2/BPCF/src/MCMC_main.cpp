#include <Rcpp.h>

#include "decision_tree.h"
#include "MCMC_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
List MCMC(
    const NumericMatrix& Xpred,
    const NumericVector& Y_trt,
    NumericVector& M_out,
    NumericVector& Y_out,
    const NumericVector& PS,
    const double p_grow,   // Prob. of GROW
    const double p_prune,  // Prob. of PRUNE
    const double p_change, // Prob. of CHANGE
    const int m1,          // Num. of Trees: default setting 150
    const int m2,          // Num. of Trees: default setting 50
    const int m3,          // Num. of Trees: default setting 150
    const int m4,          // Num. of Trees: default setting 30
    const int nu,
    double lambda_m, double lambda_y,
    double alpha, double beta, 
    const int n_iter,
    const double sigma_mu_m_tau_sigma,
    const double sigma_mu_m_mu_sigma,
    const double sigma_mu_y_tau_sigma,
    const double sigma_mu_y_mu_sigma,
    const bool verbose = false
) {

    // Data preparation
    const int P  = Xpred.ncol(); // number of covariates
    const int n  = Xpred.nrow(); // number of observations

    NumericVector M1 = clone(M_out);
    NumericVector M0 = clone(M_out);
    NumericVector Y1(n), Y0(n);
    
    const double shift = mean(Y_out);
    const double mshift = mean(M_out);
    Y_out = Y_out - shift; // center mean
    M_out = M_out - mshift; // center mean

    NumericVector Xcut_ps[P + 1], Xcut[P], Xcut_y[P + 2]; // e.g. unique value of potential confounders
    for (int j = 0; j < P; j++)
    {
        NumericVector temp;
        temp = unique(Xpred(_, j));
        temp.sort();
        Xcut[j] = clone(temp);
        
        Xcut_ps[j] = clone(temp);
        
        Xcut_y[j] = clone(temp);
    }
    NumericVector tempps;
    tempps = unique(PS);
    tempps.sort();
    Xcut_ps[P] = clone(tempps);
    

    NumericVector temp;
    temp = unique(M_out) + mshift;
    temp.sort();
    Xcut_y[P] = clone(temp);
    Xcut_y[P + 1] = clone(temp);

    // Initial Setup
    // Priors, initial values and hyper-parameters
    NumericVector Z = Rcpp::rnorm(n, R::qnorm(mean(Y_trt), 0, 1, true, false), 1); // latent variable
    NumericVector prob = {p_grow, p_prune, p_change};

//    double sigma2 = 1;
    NumericVector sigma2_m       (n_iter + 1); // create placeholder for sigma2_1
    sigma2_m(0)       = var(M_out);
    NumericVector sigma2_y       (n_iter + 1); // create placeholder for sigma2_1
    sigma2_y(0)       = var(Y_out);
    
    // sigma_mu based on min/max of Z, M (A=1) and Y (A=0)
    // double sigma_mu   = std::max(pow(min(Z)  / (-2 * sqrt(m)), 2), pow(max(Z)  / (2 * sqrt(m)), 2));
    double sigma_mu_m_mu = var(M_out) / m1;
    double sigma_mu_m_tau = var(M_out) / m2;
    double sigma_mu_y_mu = var(Y_out) / m3;
    double sigma_mu_y_tau = var(Y_out) / m4;

    
    // Initial values of R
//    NumericVector R  = clone(Z);
    NumericVector R_M = Rcpp::rnorm(n, 0, 1);
    NumericVector R_Y = Rcpp::rnorm(n, 0, 1);
    
    // Initial values for the selection probabilities (now, we may go back to uniform prob)
  //  NumericVector post_dir_alpha  = rep(1.0, P);
    NumericVector post_dir_alpha1 = rep(1.0, P+1);
    NumericVector post_dir_alpha2 = rep(1.0, P);
    NumericVector post_dir_alpha3 = rep(1.0, P+1);
    NumericVector post_dir_alpha4 = rep(1.0, P+2);
    
    // thin = 10, burn-ins = n_iter/2
    int thin       = 10;
    int burn_in    = n_iter / 2;
//    int burn_in_ps = n_iter_ps / 2;
    int n_post     = (n_iter - burn_in) / thin; // number of post sample
//    int n_post_ps  = (n_iter_ps - burn_in_ps) / thin; // number of post sample
    int thin_count = 1;

    NumericVector Effect (n_post);
    NumericMatrix predicted_M1 (n, n_post);
    NumericMatrix predicted_M0 (n, n_post);
    NumericMatrix predicted_Y1 (n, n_post);
    NumericMatrix predicted_Y0 (n, n_post);
    NumericMatrix predicted_Y (n, n_post);
//    NumericMatrix predicted_PS (n, n_post_ps);
    NumericMatrix predicted_S (n, n_post);
    int post_sample_idx = 0;

//    IntegerMatrix Obs_list  (n,  m); // changed list to matrix
    IntegerMatrix Obs1_list (n, m1);
    IntegerMatrix Obs2_list (n, m2);
    IntegerMatrix Obs3_list (n, m3);
    IntegerMatrix Obs4_list (n, m4);
    

    // Place-holder for the posterior samples
//    NumericMatrix Tree   (n,     m);
    NumericMatrix Tree1  (n,    m1);
    NumericMatrix Tree2  (n,    m2);
    NumericMatrix Tree3  (n,    m3);
    NumericMatrix Tree4  (n,    m4);
    NumericMatrix TreeY  (1,    m4);
    NumericMatrix TreeY_holder  (1,    P+2);
//    NumericMatrix TreePS (n, m);

//    DecisionTree dt_list[m]; // changed to array of trees
    DecisionTree dt1_list[m1];
    DecisionTree dt2_list[m2];
    DecisionTree dt3_list[m3];
    DecisionTree dt4_list[m4];
//    for (int t = 0; t < m; t++)
//    {
//        dt_list[t]  = DecisionTree(n,  t);
//    }
    for (int t = 0; t < m1; t++)
    {
        dt1_list[t] = DecisionTree(n, t);
    }
    for (int t = 0; t < m2; t++)
    {
        dt2_list[t] = DecisionTree(n, t);
    }
    for (int t = 0; t < m3; t++)
    {
      dt3_list[t] = DecisionTree(n, t);
    }
    for (int t = 0; t < m4; t++)
    {
      dt4_list[t] = DecisionTree(n, t);
    }
    
    // Obtaining namespace of MCMCpack package
    Environment MCMCpack = Environment::namespace_env("MCMCpack");

    // Picking up rinvgamma() and rdirichlet() function from MCMCpack package
    Function rinvgamma  = MCMCpack["rinvgamma"];
    Function rdirichlet = MCMCpack["rdirichlet"];

//    NumericVector prop_prob = rdirichlet(1, rep(1, P));
    NumericVector prop_prob1 = rdirichlet(1, rep(1, (P + 1)));
    NumericVector prop_prob2 = rdirichlet(1, rep(1, P));
    NumericVector prop_prob3 = rdirichlet(1, rep(1, (P + 1)));
    NumericVector prop_prob4 = rdirichlet(1, rep(1, (P + 2)));
    
    NumericMatrix xpred_mult(n, P + 1), Xpred1(n, P + 1), Xpred2(n, P + 2), prop_Xpred2(n, P + 2);   // placeholder for bootstrap sample for inference (correct this statement)
    NumericVector prop_M;
    
    for( int i = 1; i < n; i++)
    {
       for (int j = 0; j < P; j++)
       {
           xpred_mult(i, j) = Xpred(i, j);
           Xpred1(i, j) = Xpred(i, j);
           Xpred2(i, j) = Xpred(i, j);
           prop_Xpred2(i, j) = Xpred(i, j);
       }
       xpred_mult(i, P) = i;
       Xpred1(i, P) = PS(i);
       Xpred2(i, P) = Rcpp::rnorm(1, mean(M_out)+mshift, 1)(0);
       Xpred2(i, P + 1) = Rcpp::rnorm(1, mean(M_out)+mshift, 1)(0);
       prop_Xpred2(i, P) = Rcpp::rnorm(1, mean(M_out)+mshift, 1)(0);
       prop_Xpred2(i, P + 1) = Rcpp::rnorm(1, mean(M_out)+mshift, 1)(0);
    }

  
    
    ////////////////////////////////////////
    //////////   Run main MCMC    //////////
    ////////////////////////////////////////


    double mu1, mu0;
    for (int i = 0; i < n; i++)
    {
      Xpred1(i, P) = PS(i);
    }
    
    for (int iter = 1; iter <= n_iter; iter++)
    {
        Rcout << "Rcpp iter : " << iter << " of " << n_iter << std::endl;
 
        // ------ Intermediate Model (mu function)
        for (int t = 0; t < m1; t++)
        {
            update_R_mu(R_M, M_out, Tree1, Tree2, Y_trt, t);

            if (dt1_list[t].length() == 1)
            {
                // tree has no node yet
                dt1_list[t].GROW_first(
                    Xpred1, Xcut_ps, sigma2_m(iter - 1), sigma_mu_m_mu, R_M, Obs1_list,
                    p_prune, p_grow, alpha, beta, prop_prob1);
            }
            else
            {
                int step = sample(3, 1, false, prob)(0);

                switch (step)
                {
                    case 1: // GROW step
                        dt1_list[t].GROW(
                            Xpred1, Xcut_ps, sigma2_m(iter - 1), sigma_mu_m_mu, R_M, Obs1_list,
                           p_prune, p_grow, alpha, beta, prop_prob1
                        );
                        break;

                    case 2: // PRUNE step
                        dt1_list[t].PRUNE(
                            Xpred1, Xcut_ps, sigma2_m(iter - 1), sigma_mu_m_mu, R_M, Obs1_list,
                            p_prune, p_grow, alpha, beta, prop_prob1
                        );
                        break;

                    case 3: // CHANGE step
                        dt1_list[t].CHANGE(
                            Xpred1, Xcut_ps, sigma2_m(iter - 1), sigma_mu_m_mu, R_M, Obs1_list,
                            p_prune, p_grow, alpha, beta, prop_prob1
                        );
                        break;

                    default: {};
                } // end of switch
            }     // end of tree instance
            
            dt1_list[t].Mean_Parameter(Tree1, sigma2_m(iter - 1), sigma_mu_m_mu, R_M, Obs1_list);
        }
        

        NumericVector prop_sigma_mu_m_mu = Rcpp::rgamma(1, 1000 * sigma_mu_m_mu, 0.001);
        double dir_lik_p, dir_lik, ratio;
        double tree_sum = 0, tree_sum1 = 0;
        for (int t = 0; t < m1; t++)
        {
            NumericVector tree_unique = unique(clone(Tree1)(_,t));
            tree_sum += sum(Rcpp::dnorm(tree_unique, 0, prop_sigma_mu_m_mu(0), true));
            tree_sum1 += sum(Rcpp::dnorm(tree_unique, 0, sigma_mu_m_mu, true));
        }
        dir_lik_p = tree_sum + R::dgamma(sigma_mu_m_mu, 1000*prop_sigma_mu_m_mu(0),0.001, true) + R::dcauchy(prop_sigma_mu_m_mu(0),0, sigma_mu_m_mu_sigma, true);
        dir_lik = tree_sum1 + R::dgamma(prop_sigma_mu_m_mu(0), 1000*sigma_mu_m_mu,0.001, true) + R::dcauchy(sigma_mu_m_mu,0, sigma_mu_m_mu_sigma, true);
        ratio = dir_lik_p - dir_lik;
        if (ratio > log(R::runif(0, 1)))
        {
            sigma_mu_m_mu = prop_sigma_mu_m_mu(0);
        }

        // ------ Intermediate Model (tau function)
        for (int t = 0; t < m2; t++)
        {
            update_R_tau(R_M, M_out, Tree1, Tree2, Y_trt, t);
            
            
            if (dt2_list[t].length() == 1)
            {
                // tree has no node yet
                dt2_list[t].GROW_first(
                        Xpred, Xcut, sigma2_m(iter - 1), sigma_mu_m_tau, R_M / (Y_trt-0.5), Obs2_list,
                        p_prune, p_grow, 0.25, 3, prop_prob2);
            }
            else
            {
                int step = sample(3, 1, false, prob)(0);
                //   Rcout << "Rcpp step start: " << step <<  std::endl;
                
                switch (step)
                {
                case 1: // GROW step
                    dt2_list[t].GROW(
                            Xpred, Xcut, sigma2_m(iter - 1), sigma_mu_m_tau, R_M / (Y_trt-0.5), Obs2_list,
                            p_prune, p_grow, 0.25, 3, prop_prob2
                    );
                    break;
                    
                case 2: // PRUNE step
                    dt2_list[t].PRUNE(
                            Xpred, Xcut, sigma2_m(iter - 1), sigma_mu_m_tau, R_M / (Y_trt-0.5), Obs2_list,
                            p_prune, p_grow, 0.25, 3, prop_prob2
                    );
                    break;
                    
                case 3: // CHANGE step
                    dt2_list[t].CHANGE(
                            Xpred, Xcut, sigma2_m(iter - 1), sigma_mu_m_tau, R_M / (Y_trt-0.5), Obs2_list,
                            p_prune, p_grow, 0.25, 3, prop_prob2
                    );
                    break;
                    
                default: {};
                } // end of switch
            }     // end of tree instance
            
            dt2_list[t].Mean_Parameter(Tree2, sigma2_m(iter - 1), sigma_mu_m_tau, R_M / (Y_trt-0.5), Obs2_list);
        }
        
        NumericVector prop_sigma_mu_m_tau = Rcpp::rgamma(1, 1000 * sigma_mu_m_tau, 0.001);
        tree_sum = 0;
        tree_sum1 = 0;
        for (int t = 0; t < m2; t++)
        {
            NumericVector tree_unique1 = unique(clone(Tree2)(_,t));
            tree_sum += sum(Rcpp::dnorm(tree_unique1, 0, prop_sigma_mu_m_tau(0), true));
            tree_sum1 += sum(Rcpp::dnorm(tree_unique1, 0, sigma_mu_m_tau, true));
        }
        dir_lik_p = tree_sum + R::dgamma(sigma_mu_m_tau, 1000*prop_sigma_mu_m_tau(0),0.001, true) + R::dnorm(prop_sigma_mu_m_tau(0),0,  sigma_mu_m_tau_sigma, true);
        dir_lik = tree_sum1 + R::dgamma(prop_sigma_mu_m_tau(0), 1000*sigma_mu_m_tau,0.001, true) + R::dnorm(sigma_mu_m_tau,0,  sigma_mu_m_tau_sigma, true);
        ratio = dir_lik_p - dir_lik;
        if (ratio > log(R::runif(0, 1)))
        {
            sigma_mu_m_tau = prop_sigma_mu_m_tau(0);
        }

        
        NumericVector m_new = clone(rowSums(Tree1)) + clone(rowSums(Tree2)) * (Y_trt - 0.5);
        
        
        if (iter > 1)
        {
          for (int i = 0; i < n; i++)
          {
            mu0 = sum(clone(Tree1)(i,_))+sum(clone(Tree2)(i,_))*(-0.5)+mshift;
            mu1 = sum(clone(Tree1)(i,_))+sum(clone(Tree2)(i,_))*(0.5)+mshift;
            if (Y_trt(i) == 1)
            {
              prop_M = Rcpp::rnorm(1, M0(i), 0.05);
              prop_Xpred2(i, P) = clone(prop_M)(0);
              prop_Xpred2(i, P+1) = M_out(i) + mshift;
            }
            else
            {
              prop_M = Rcpp::rnorm(1, M1(i),  0.05);
              prop_Xpred2(i, P) = M_out(i) + mshift;
              prop_Xpred2(i, P+1) = clone(prop_M)(0);
            }
            
            TreeY_holder(0,_) = clone(prop_Xpred2)(i,_);
            for (int h = 0; h < m4; h++)
            {
              dt4_list[h].Predict(TreeY, Xcut_y, TreeY_holder, 1);
            }
            
            if (Y_trt(i) == 1)
            {
              dir_lik_p = R::dnorm(M0(i), prop_M(0), 0.05, true)+R::dnorm(prop_M(0), mu0, sqrt(sigma2_m(iter - 1)), true )+R::dnorm(Y_out(i), sum(clone(Tree3)(i,_))+sum(clone(TreeY))*(Y_trt(i)-0.5), sqrt(sigma2_y(iter - 1)), true);
              dir_lik = R::dnorm(prop_M(0), M0(i), 0.05, true)+R::dnorm(M0(i), mu0, sqrt(sigma2_m(iter - 1)), true )+R::dnorm(Y_out(i), sum(clone(Tree3)(i,_))+sum(clone(Tree4)(i,_))*(Y_trt(i)-0.5), sqrt(sigma2_y(iter - 1)), true);
              ratio = dir_lik_p - dir_lik;
            }
            else
            {
              dir_lik_p = R::dnorm(M1(i), prop_M(0), 0.05, true)+R::dnorm(prop_M(0), mu1, sqrt(sigma2_m(iter - 1)), true )+R::dnorm(Y_out(i), sum(clone(Tree3)(i,_))+sum(clone(TreeY))*(Y_trt(i)-0.5), sqrt(sigma2_y(iter - 1)), true);
              dir_lik = R::dnorm(prop_M(0), M1(i), 0.05, true)+R::dnorm(M1(i), mu1, sqrt(sigma2_m(iter - 1)), true )+R::dnorm(Y_out(i), sum(clone(Tree3)(i,_))+sum(clone(Tree4)(i,_))*(Y_trt(i)-0.5), sqrt(sigma2_y(iter - 1)), true);
              ratio = dir_lik_p - dir_lik;
              
            }
            if (ratio > log(R::runif(0, 1)))
            {
              M1(i) = clone(prop_Xpred2)(i, P+1);
              M0(i) = clone(prop_Xpred2)(i, P);
            //  Rcout << "Rcpp accept: " << "yes" <<  std::endl;
            }
            Xpred2(i, P) = clone(M0)(i);
            Xpred2(i, P+1) = clone(M1)(i);
          }
          
        }
        
        //  Sample variance parameter
        {
          NumericVector sigma2_m_temp = rinvgamma(1, nu / 2 + n / 2, nu * lambda_m / 2 + sum(pow(M_out - m_new, 2)) / 2);
           sigma2_m(iter) = sigma2_m_temp(0);
        }
        
        
        // ------ Outcome Model (mu function)
        for (int t = 0; t < m3; t++)
        {
          update_R_mu(R_Y, Y_out, Tree3, Tree4, Y_trt, t);
          
          if (dt3_list[t].length() == 1)
          {
            // tree has no node yet
            dt3_list[t].GROW_first(
                Xpred1, Xcut_ps, sigma2_y(iter - 1), sigma_mu_y_mu, R_Y, Obs3_list,
                p_prune, p_grow, alpha, beta, prop_prob3);
          }
          else
          {
            int step = sample(3, 1, false, prob)(0);
            //   Rcout << "Rcpp step start: " << step <<  std::endl;
            
            switch (step)
            {
            case 1: // GROW step
              dt3_list[t].GROW(
                  Xpred1, Xcut_ps, sigma2_y(iter - 1), sigma_mu_y_mu, R_Y, Obs3_list,
                  p_prune, p_grow, alpha, beta, prop_prob3
              );
              break;
              
            case 2: // PRUNE step
              dt3_list[t].PRUNE(
                  Xpred1, Xcut_ps, sigma2_y(iter - 1), sigma_mu_y_mu, R_Y, Obs3_list,
                  p_prune, p_grow, alpha, beta, prop_prob3
              );
              break;
              
            case 3: // CHANGE step
              dt3_list[t].CHANGE(
                  Xpred1, Xcut_ps, sigma2_y(iter - 1), sigma_mu_y_mu, R_Y, Obs3_list,
                  p_prune, p_grow, alpha, beta, prop_prob3
              );
              break;
              
            default: {};
            } // end of switch
          }     // end of tree instance
          
          dt3_list[t].Mean_Parameter(Tree3, sigma2_y(iter - 1), sigma_mu_y_mu, R_Y, Obs3_list);
        }
        
        
        NumericVector prop_sigma_mu_y_mu = Rcpp::rgamma(1, 1000 * sigma_mu_y_mu, 0.001);
        tree_sum = 0;
        tree_sum1 = 0;
        for (int t = 0; t < m3; t++)
        {
          NumericVector tree_unique = unique(clone(Tree3)(_,t));
          tree_sum += sum(Rcpp::dnorm(tree_unique, 0, prop_sigma_mu_y_mu(0), true));
          tree_sum1 += sum(Rcpp::dnorm(tree_unique, 0, sigma_mu_y_mu, true));
        }
        dir_lik_p = tree_sum + R::dgamma(sigma_mu_y_mu, 1000*prop_sigma_mu_y_mu(0),0.001, true) + R::dcauchy(prop_sigma_mu_y_mu(0),0, sigma_mu_y_mu_sigma, true);

        dir_lik = tree_sum1 + R::dgamma(prop_sigma_mu_y_mu(0), 1000*sigma_mu_y_mu,0.001, true) + R::dcauchy(sigma_mu_y_mu,0, sigma_mu_y_mu_sigma, true);

        ratio = dir_lik_p - dir_lik;
        if (ratio > log(R::runif(0, 1)))
        {
          sigma_mu_y_mu = prop_sigma_mu_y_mu(0);
        }
        

        
        
        // ------ Outcome Model (tau function)
        for (int t = 0; t < m4; t++)
        {
          update_R_tau(R_Y, Y_out, Tree3, Tree4, Y_trt, t);
          
          
          if (dt4_list[t].length() == 1)
          {
            // tree has no node yet
            dt4_list[t].GROW_first(
                Xpred2, Xcut_y, sigma2_y(iter - 1), sigma_mu_y_tau, R_Y / (Y_trt-0.5), Obs4_list,
                p_prune, p_grow, 0.25, 3, prop_prob4);
          }
          else
          {
            int step = sample(3, 1, false, prob)(0);
            
            switch (step)
            {
            case 1: // GROW step
              dt4_list[t].GROW(
                  Xpred2, Xcut_y, sigma2_y(iter - 1), sigma_mu_y_tau, R_Y / (Y_trt-0.5), Obs4_list,
                  p_prune, p_grow, 0.25, 3, prop_prob4
              );
              break;
              
            case 2: // PRUNE step
              dt4_list[t].PRUNE(
                  Xpred2, Xcut_y, sigma2_y(iter - 1), sigma_mu_y_tau, R_Y / (Y_trt-0.5), Obs4_list,
                  p_prune, p_grow, 0.25, 3, prop_prob4
              );
              break;
              
            case 3: // CHANGE step
              dt4_list[t].CHANGE(
                  Xpred2, Xcut_y, sigma2_y(iter - 1), sigma_mu_y_tau, R_Y / (Y_trt-0.5), Obs4_list,
                  p_prune, p_grow, 0.25, 3, prop_prob4
              );
              break;
              
            default: {};
            } // end of switch
          }     // end of tree instance
          
          dt4_list[t].Mean_Parameter(Tree4, sigma2_y(iter - 1), sigma_mu_y_tau, R_Y / (Y_trt-0.5), Obs4_list);
        }

        NumericVector prop_sigma_mu_y_tau = Rcpp::rgamma(1, 1000 * sigma_mu_y_tau, 0.001);
        tree_sum = 0;
        tree_sum1 = 0;
        for (int t = 0; t < m4; t++)
        {
          NumericVector tree_unique1 = unique(clone(Tree4)(_,t));
          tree_sum += sum(Rcpp::dnorm(tree_unique1, 0, prop_sigma_mu_y_tau(0), true));
          tree_sum1 += sum(Rcpp::dnorm(tree_unique1, 0, sigma_mu_y_tau, true));
        }
        dir_lik_p = tree_sum + R::dgamma(sigma_mu_y_tau, 1000*prop_sigma_mu_y_tau(0),0.001, true) + R::dnorm(prop_sigma_mu_y_tau(0),0,  sigma_mu_y_tau_sigma, true);

        dir_lik = tree_sum1 + R::dgamma(prop_sigma_mu_y_tau(0), 1000*sigma_mu_y_tau,0.001, true) + R::dnorm(sigma_mu_y_tau,0,  sigma_mu_y_tau_sigma, true);

        ratio = dir_lik_p - dir_lik;
        if (ratio > log(R::runif(0, 1)))
        {
          sigma_mu_y_tau = prop_sigma_mu_y_tau(0);
        }

        for (int i = 0; i < n; i++)
        {
        Y1(i) = Y_trt(i) * Y_out(i) + (1 - Y_trt(i)) * (clone(rowSums(Tree4))(i) + Y_out(i) ) + shift;
        Y0(i) = (1 - Y_trt(i)) * Y_out(i) + Y_trt(i) * (Y_out(i) - clone(rowSums(Tree4))(i) ) + shift;
        }
        
        NumericVector y_new = clone(rowSums(Tree3)) + clone(rowSums(Tree4)) * (Y_trt - 0.5);

        
        //  Sample variance parameter
        {
          NumericVector sigma2_y_temp = rinvgamma(1, nu / 2 + n / 2, nu * lambda_y / 2 + sum(pow(Y_out - y_new, 2)) / 2);
          sigma2_y(iter) = sigma2_y_temp(0);
        }
        
        

        // Num. of inclusion of each potential confounder
        NumericVector add1(P+1), add2(P), add3(P+1), add4(P+2);

        
        for (int t = 0; t < m1; t++)
        {
          add1 += dt1_list[t].num_included(P+1);
        }
        
        for (int t = 0; t < m2; t++)
        {
          add2 += dt2_list[t].num_included(P);
        }
        
        for (int t = 0; t < m3; t++)
        {
          add3 += dt3_list[t].num_included(P+1);
        }
        for (int t = 0; t < m4; t++)
        {
          add4 += dt4_list[t].num_included(P+2);
        }
        post_dir_alpha1 = rep(1.0, P+1) + add1;
        post_dir_alpha2 = rep(1.0, P) + add2;
        post_dir_alpha3 = rep(1.0, P+1) + add3;
        post_dir_alpha4 = rep(1.0, P+2) + add4;
        prop_prob1 = rdirichlet(1, post_dir_alpha1);
        prop_prob2 = rdirichlet(1, post_dir_alpha2);
        prop_prob3 = rdirichlet(1, post_dir_alpha3);
        prop_prob4 = rdirichlet(1, post_dir_alpha4);

        
        // Sampling E[Y(1)-Y(0)]
        if (iter > burn_in)
        {
            if (thin_count < thin)
            {
                thin_count++;
            }
            else
            {
                thin_count = 1;
                predicted_M1 (_, post_sample_idx) = M1;
                predicted_M0 (_, post_sample_idx) = M0;
                predicted_Y1 (_, post_sample_idx) = Y1;
                predicted_Y0 (_, post_sample_idx) = Y0;
                predicted_Y (_, post_sample_idx) = Y1-Y0;
                predicted_S (_, post_sample_idx) = M1-M0;
                post_sample_idx++;
            }
        }

        Rcpp::checkUserInterrupt(); // check for break in R
    } // end of MCMC iterations

    List L = List::create(
        Named("predicted_S") = predicted_S,
        Named("predicted_Y") = predicted_Y,
        Named("predicted_Y0") = predicted_Y0,
        Named("predicted_Y1") = predicted_Y1,
        Named("predicted_M0") = predicted_M0,
        Named("predicted_M1") = predicted_M1
    );

    return L;
}
