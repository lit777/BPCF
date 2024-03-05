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
    const int m,           // Num. of Trees: default setting 150
    const int nu,
    double lambda_m, double lambda_y,
    double alpha, double beta,
    const int n_iter,
    const bool verbose = false
) {

    // Data preparation
    const int P  = Xpred.ncol(); // number of covariates
    const int n  = Xpred.nrow(); // number of observations
    const int n1 = sum(Y_trt);   // number of treated sample
    const int n0 = n - n1;       // number of control sample

    const double shift = mean(Y_out);
    const double mshift = mean(M_out);
    Y_out = Y_out - shift; // center mean
    M_out = M_out - mshift; // center mean

    NumericVector M1(n1), M0(n0), Y1(n1), Y0(n0);
    NumericMatrix Xpred1(n1, P), Xpred0(n0, P);
 
        IntegerVector Y_trt_int = as<IntegerVector>(Y_trt);
        IntegerVector idx1(n1), idx0(n0);
        int count1 = 0, count0 = 0;
        for (int i = 0; i < n; i++)
        {
            if (Y_trt(i) == 1.0)
            {
                idx1(count1) = i; // idx of Y_trt = 1
                count1++;
            }
            else
            {
                idx0(count0) = i; // idx of Y_trt = 0
                count0++;
            }
        }
        

    for (int i = 0; i < n1; i++)
    {
      Xpred1(i, _) = Xpred(idx1(i), _); // potential confounders (A=1)
      Y1(i) = Y_out(idx1(i));           // Outcome (A=1)
      M1(i) = M_out(idx1(i));           // Intermediate (A=1)
    }
    
    for (int i = 0; i < n0; i++)
    {
      Xpred0(i, _) = Xpred(idx0(i), _); // potential confounders (A=0)
      Y0(i) = Y_out(idx0(i));           // Outcome (A=0)
      M0(i) = M_out(idx0(i));           // Intermediate (A=0)
    }
  
    

    // Xcut <- lappy(1:dim(Xpred)[2], function(t) sort(unique(Xpred[,t])))
    NumericVector Xcut_m1[P + 1], Xcut_m0[P + 1], Xcut_y1[P + 3], Xcut_y0[P + 3]; // e.g. unique value of potential confounders
    for (int j = 0; j < P; j++)
    {
        NumericVector temp;
        temp = unique(Xpred1(_, j));
        temp.sort();
        Xcut_m1[j] = clone(temp);
        Xcut_y1[j] = clone(temp);
        
        temp = unique(Xpred0(_, j));
        temp.sort();
        Xcut_m0[j] = clone(temp);
        Xcut_y0[j] = clone(temp);
    }

    
    NumericVector temp1(n1);
    for (int j = 0; j < n1; j++)
    {
      temp1(j) = PS(idx1(j));
    }
    temp1.sort();
    Xcut_m1[P] = clone(temp1);
    Xcut_y1[P] = clone(temp1);
    
    
    NumericVector temp0(n0);
    for (int j = 0; j < n0; j++)
    {
      temp0(j) = PS(idx0(j));
    }
    temp0.sort();
    Xcut_m0[P] = clone(temp0);
    Xcut_y0[P] = clone(temp0);
    

    
    NumericVector temp;
    temp = unique(M_out);
    temp.sort();
    Xcut_y1[P + 1] = clone(temp);
    Xcut_y1[P + 2] = clone(temp);
    Xcut_y0[P + 1] = clone(temp);
    Xcut_y0[P + 2] = clone(temp);
    
    // Initial Setup
    // Priors, initial values and hyper-parameters
    // NumericVector Z = Rcpp::rnorm(n, R::qnorm(mean(Y_trt), 0, 1, true, false), 1); // latent variable
    NumericVector prob = {p_grow, p_prune, p_change};

    // double sigma2 = 1;
    NumericVector sigma2_m1       (n_iter + 1); // create placeholder for sigma2_1
    sigma2_m1(0)       = var(M_out);
    NumericVector sigma2_m0       (n_iter + 1); // create placeholder for sigma2_1
    sigma2_m0(0)       = var(M_out);
    
    NumericVector sigma2_y1       (n_iter + 1); // create placeholder for sigma2_1
    sigma2_y1(0)       = var(Y_out);
    NumericVector sigma2_y0       (n_iter + 1); // create placeholder for sigma2_1
    sigma2_y0(0)       = var(Y_out);
    

    
    // sigma_mu based on min/max of Z, M (A=1) and Y (A=0)
    // double sigma_mu   = std::max(pow(min(Z)  / (-2 * sqrt(m)), 2), pow(max(Z)  / (2 * sqrt(m)), 2));
    double sigma_mu_m1 = std::max(pow(min(M1) / (-2 * sqrt(m)), 2), pow(max(M1) / (2 * sqrt(m)), 2));
    double sigma_mu_m0 = std::max(pow(min(M0) / (-2 * sqrt(m)), 2), pow(max(M0) / (2 * sqrt(m)), 2));
    double sigma_mu_y1 = std::max(pow(min(Y1) / (-2 * sqrt(m)), 2), pow(max(Y1) / (2 * sqrt(m)), 2));
    double sigma_mu_y0 = std::max(pow(min(Y0) / (-2 * sqrt(m)), 2), pow(max(Y0) / (2 * sqrt(m)), 2));
    // Need to specify sigma_mu_mu_sigma and sigma_mu_tau_sigma
    // f <- function(scale) qcauchy(0.75, 0, scale) - 2*sd(M_out.s)   # first
    // sigma_mu_m1.sigma <- uniroot.all(f, c(0.1^5, 100))

    // f <- function(sd) qnorm(0.75, 0, sd) - sd(M_out.s)             # second
    // sigma_mu_m0.sigma <- uniroot.all(f, c(0.1^5, 100))


    // Initial values of R
    // NumericVector R  = clone(Z);
    NumericVector R_M1 = Rcpp::rnorm(n1, 0, 1);
    NumericVector R_Y1 = Rcpp::rnorm(n1, 0, 1);
    NumericVector R_M0 = Rcpp::rnorm(n0, 0, 1);
    NumericVector R_Y0 = Rcpp::rnorm(n0, 0, 1);
    
    // Initial values for the selection probabilities (now, we may go back to uniform prob)
    // NumericVector post_dir_alpha  = rep(1.0, P);
    NumericVector post_dir_alpha_m1 = rep(1.0, P+1);
    NumericVector post_dir_alpha_m0 = rep(1.0, P+1);
    NumericVector post_dir_alpha_y1 = rep(1.0, P+3);
    NumericVector post_dir_alpha_y0 = rep(1.0, P+3);
    
    // thin = 10, burn-ins = n_iter/2
    int thin       = 10;
    int burn_in    = n_iter / 2;
    int n_post     = (n_iter - burn_in) / thin; // number of post sample
    int thin_count = 1;
    // IntegerVector seq = init_seq(n_iter, thin, burn_in);
    // IntegerVector::iterator seq_pt = seq.begin();

    NumericVector Effect (n_post);
    // NumericVector PO_Y1  (n_post);
    // NumericVector PO_Y0  (n_post);
    NumericMatrix predicted_M1 (n, n_post);
    NumericMatrix predicted_M0 (n, n_post);
    NumericMatrix predicted_Y1 (n, n_post);
    NumericMatrix predicted_Y0 (n, n_post);
    NumericMatrix predicted_Y (n, n_post); 
    // NumericMatrix predicted_PS (n, n_post);
    NumericMatrix predicted_S (n, n_post);
    // IntegerMatrix ind    (n_post, P);
    int post_sample_idx = 0;

    // IntegerMatrix Obs_list  (n,  m); // changed list to matrix
    IntegerMatrix Obs_m1_list (n1, m);
    IntegerMatrix Obs_m0_list (n0, m);
    IntegerMatrix Obs_y1_list (n1, m);
    IntegerMatrix Obs_y0_list (n0, m);
    

    // Place-holder for the posterior samples
    // NumericMatrix Tree   (n,     m);
    NumericMatrix Tree_m1  (n1,    m);
    NumericMatrix Tree_m0  (n0,    m);
    NumericMatrix Tree_y1  (n1,    m);
    NumericMatrix Tree_y0  (n0,    m);
    NumericMatrix TreeY_holder  (1,    P+3);
    // NumericMatrix TreePS (n, m);
    NumericMatrix Tree_m11 (n0, m);
    NumericMatrix Tree_m00 (n1, m);
    NumericMatrix Tree_y11 (n0, m);
    NumericMatrix Tree_y00 (n1, m);
    
    NumericMatrix TreeM_holder  (1,    P+1);
    // NumericMatrix Tree11 (n, m1);
    // NumericMatrix Tree00 (n, m2);

    // DecisionTree dt_list[m]; // changed to array of trees
    DecisionTree dt_m1_list[m];
    DecisionTree dt_m0_list[m];
    DecisionTree dt_y1_list[m];
    DecisionTree dt_y0_list[m];
    for (int t = 0; t < m; t++)
    {
        // dt_list[t]  = DecisionTree(n,  t);
        dt_m1_list[t]  = DecisionTree(n1,  t);
        dt_m0_list[t]  = DecisionTree(n0,  t);
        dt_y1_list[t]  = DecisionTree(n1,  t);
        dt_y0_list[t]  = DecisionTree(n0,  t);
    }

    // Obtaining namespace of MCMCpack package
    Environment MCMCpack = Environment::namespace_env("MCMCpack");

    // Picking up rinvgamma() and rdirichlet() function from MCMCpack package
    Function rinvgamma  = MCMCpack["rinvgamma"];
    Function rdirichlet = MCMCpack["rdirichlet"];

    // NumericVector prop_prob = rdirichlet(1, rep(1, P));
    NumericVector prop_prob_m1 = rdirichlet(1, rep(1, (P + 1)));
    NumericVector prop_prob_m0 = rdirichlet(1, rep(1, (P + 1)));
    NumericVector prop_prob_y1 = rdirichlet(1, rep(1, (P + 3)));
    NumericVector prop_prob_y0 = rdirichlet(1, rep(1, (P + 3)));
    
    NumericMatrix xpred_mult(n, P + 1), Xpred_m1(n1, P + 1), Xpred_m0(n0, P + 1), Xpred_y1(n1, P + 3), Xpred_y0(n0, P + 3), prop_Xpred_y1(n1, P + 3), prop_Xpred_y0(n0, P + 3);   // placeholder for bootstrap sample for inference (correct this statement)
    NumericVector prop_M;

    for( int i = 1; i < n; i++)
    {
       for (int j = 0; j < P; j++)
       {
           xpred_mult(i, j) = Xpred(i, j);
       }
       xpred_mult(i, P) = i;
    }
    for( int i = 1; i < n1; i++)
    {
      for (int j = 0; j < P; j++)
      {
        Xpred_m1(i, j) = Xpred(idx1(i), j);
        Xpred_y1(i, j) = Xpred(idx1(i), j);
        prop_Xpred_y1(i, j) = Xpred(idx1(i), j);
      }
      Xpred_m1(i, P) = Rcpp::runif(1, 0, 1)(0);
      Xpred_y1(i, P) = Rcpp::runif(1, 0, 1)(0);
      Xpred_y1(i, P + 1) = Rcpp::rnorm(1, mean(M_out), 1)(0);
      Xpred_y1(i, P + 2) = Rcpp::rnorm(1, mean(M_out), 1)(0);
    }
    for( int i = 1; i < n0; i++)
    {
      for (int j = 0; j < P; j++)
      {
        Xpred_m0(i, j) = Xpred(idx0(i), j);
        Xpred_y0(i, j) = Xpred(idx0(i), j);
        prop_Xpred_y1(i, j) = Xpred(idx0(i), j);
      }
      Xpred_m0(i, P) = Rcpp::runif(1, 0, 1)(0);
      Xpred_y0(i, P) = Rcpp::runif(1, 0, 1)(0);
      Xpred_y0(i, P + 1) = Rcpp::rnorm(1, mean(M_out), 1)(0);
      Xpred_y0(i, P + 2) = Rcpp::rnorm(1, mean(M_out), 1)(0);
    }
    


    ////////////////////////////////////////
    //////////   Run main MCMC    //////////
    ////////////////////////////////////////

    // Run MCMC
    post_sample_idx = 0;
    thin_count = 1;
    for (int i = 0; i < n0; i++)
    {
      Xpred_m0(i, P) = PS(idx0(i));
      Xpred_y0(i, P) = PS(idx0(i));
    }
    for (int i = 0; i < n1; i++)
    {
      Xpred_m1(i, P) = PS(idx1(i));
      Xpred_y1(i, P) = PS(idx1(i));
    }
    
    for (int iter = 1; iter <= n_iter; iter++)
    {
        Rcout << "Rcpp iter : " << iter << " of " << n_iter << std::endl;
 
        
        // ------ Intermediate Model (M1)
        for (int t = 0; t < m; t++)
        {
            update_R(R_M1, M1, Tree_m1, t);

            if (dt_m1_list[t].length() == 1)
            {
                // tree has no node yet
                dt_m1_list[t].GROW_first(
                    Xpred_m1, Xcut_m1, sigma2_m1(iter - 1), sigma_mu_m1, R_M1, Obs_m1_list,
                    p_prune, p_grow, alpha, beta, prop_prob_m1);
            }
            else
            {
                int step = sample(3, 1, false, prob)(0);
             //   Rcout << "Rcpp step start: " << step <<  std::endl;

                switch (step)
                {
                    case 1: // GROW step
                      dt_m1_list[t].GROW(
                          Xpred_m1, Xcut_m1, sigma2_m1(iter - 1), sigma_mu_m1, R_M1, Obs_m1_list,
                          p_prune, p_grow, alpha, beta, prop_prob_m1
                        );
                        break;

                    case 2: // PRUNE step
                      dt_m1_list[t].PRUNE(
                          Xpred_m1, Xcut_m1, sigma2_m1(iter - 1), sigma_mu_m1, R_M1, Obs_m1_list,
                          p_prune, p_grow, alpha, beta, prop_prob_m1
                        );
                        break;

                    case 3: // CHANGE step
                      dt_m1_list[t].CHANGE(
                          Xpred_m1, Xcut_m1, sigma2_m1(iter - 1), sigma_mu_m1, R_M1, Obs_m1_list,
                          p_prune, p_grow, alpha, beta, prop_prob_m1
                        );
                        break;

                    default: {};
                } // end of switch
            }     // end of tree instance
            
            dt_m1_list[t].Mean_Parameter(Tree_m1, sigma2_m1(iter - 1), sigma_mu_m1, R_M1, Obs_m1_list);
        }
        
        // ------ Intermediate Model (M0)
        for (int t = 0; t < m; t++)
        {
          update_R(R_M0, M0, Tree_m0, t);
          
          if (dt_m0_list[t].length() == 1)
          {
            // tree has no node yet
            dt_m0_list[t].GROW_first(
                Xpred_m0, Xcut_m0, sigma2_m0(iter - 1), sigma_mu_m0, R_M0, Obs_m0_list,
                p_prune, p_grow, alpha, beta, prop_prob_m0);
          }
          else
          {
            int step = sample(3, 1, false, prob)(0);
            //   Rcout << "Rcpp step start: " << step <<  std::endl;
            
            switch (step)
            {
            case 1: // GROW step
              dt_m0_list[t].GROW(
                  Xpred_m0, Xcut_m0, sigma2_m0(iter - 1), sigma_mu_m0, R_M0, Obs_m0_list,
                  p_prune, p_grow, alpha, beta, prop_prob_m0
              );
              break;
              
            case 2: // PRUNE step
              dt_m0_list[t].PRUNE(
                  Xpred_m0, Xcut_m0, sigma2_m0(iter - 1), sigma_mu_m0, R_M0, Obs_m0_list,
                  p_prune, p_grow, alpha, beta, prop_prob_m0
              );
              break;
              
            case 3: // CHANGE step
              dt_m0_list[t].CHANGE(
                  Xpred_m0, Xcut_m0, sigma2_m0(iter - 1), sigma_mu_m0, R_M0, Obs_m0_list,
                  p_prune, p_grow, alpha, beta, prop_prob_m0
              );
              break;
              
            default: {};
            } // end of switch
          }     // end of tree instance
          
          dt_m0_list[t].Mean_Parameter(Tree_m0, sigma2_m0(iter - 1), sigma_mu_m0, R_M0, Obs_m0_list);
        }
        
        //  Sample variance parameter
        {
          NumericVector m_new0 = clone(rowSums(Tree_m0));
          NumericVector sigma2_m0_temp = rinvgamma(1, nu / 2 + n0 / 2, nu * lambda_m / 2 + sum(pow(M0 - m_new0, 2)) / 2);
          sigma2_m0(iter) = sigma2_m0_temp(0);
          NumericVector m_new1 = clone(rowSums(Tree_m1));
          NumericVector sigma2_m1_temp = rinvgamma(1, nu / 2 + n1 / 2, nu * lambda_m / 2 + sum(pow(M1 - m_new1, 2)) / 2);
          sigma2_m1(iter) = sigma2_m1_temp(0);
        }
        
        for (int i = 0; i < m; i++)
        {
          dt_m1_list[i].Predict(Tree_m11,  Xcut_m1, Xpred_m0, n0);
        }
        for (int i = 0; i < n0; i++)
        {
          Xpred_y0(i, P + 1) = clone(M0)(i);
          Xpred_y0(i, P + 2) = clone(rowSums(Tree_m11))(i);
        }
        
        for (int i = 0; i < m; i++)
        {
          dt_m0_list[i].Predict(Tree_m00,  Xcut_m0, Xpred_m1, n1);
        }
        for (int i = 0; i < n1; i++)
        {
          Xpred_y1(i, P + 1) = clone(rowSums(Tree_m00))(i);
          Xpred_y1(i, P + 2) = clone(M1)(i);
        }
        
        // ------ Outcome Model (Y1)
        for (int t = 0; t < m; t++)
        {
          update_R(R_Y1, Y1, Tree_y1, t);
          
          if (dt_y1_list[t].length() == 1)
          {
            // tree has no node yet
            dt_y1_list[t].GROW_first(
                Xpred_y1, Xcut_y1, sigma2_y1(iter - 1), sigma_mu_y1, R_Y1, Obs_y1_list,
                p_prune, p_grow, alpha, beta, prop_prob_y1);
          }
          else
          {
            int step = sample(3, 1, false, prob)(0);
            //   Rcout << "Rcpp step start: " << step <<  std::endl;
            
            switch (step)
            {
            case 1: // GROW step
              dt_y1_list[t].GROW(
                  Xpred_y1, Xcut_y1, sigma2_y1(iter - 1), sigma_mu_y1, R_Y1, Obs_y1_list,
                  p_prune, p_grow, alpha, beta, prop_prob_y1
              );
              break;
              
            case 2: // PRUNE step
              dt_y1_list[t].PRUNE(
                  Xpred_y1, Xcut_y1, sigma2_y1(iter - 1), sigma_mu_y1, R_Y1, Obs_y1_list,
                  p_prune, p_grow, alpha, beta, prop_prob_y1
              );
              break;
              
            case 3: // CHANGE step
              dt_y1_list[t].CHANGE(
                  Xpred_y1, Xcut_y1, sigma2_y1(iter - 1), sigma_mu_y1, R_Y1, Obs_y1_list,
                  p_prune, p_grow, alpha, beta, prop_prob_y1
              );
              break;
              
            default: {};
            } // end of switch
          }     // end of tree instance
          
          dt_y1_list[t].Mean_Parameter(Tree_y1, sigma2_y1(iter - 1), sigma_mu_y1, R_Y1, Obs_y1_list);
        }
        

        
        // ------ Outcome Model (Y1)
        for (int t = 0; t < m; t++)
        {
          update_R(R_Y0, Y0, Tree_y0, t);
          
          
          if (dt_y0_list[t].length() == 1)
          {
            // tree has no node yet
            dt_y0_list[t].GROW_first(
                Xpred_y0, Xcut_y0, sigma2_y0(iter - 1), sigma_mu_y0, R_Y0, Obs_y0_list,
                p_prune, p_grow, alpha, beta, prop_prob_y0);
          }
          else
          {
            int step = sample(3, 1, false, prob)(0);
            //   Rcout << "Rcpp step start: " << step <<  std::endl;
            
            switch (step)
            {
            case 1: // GROW step
              dt_y0_list[t].GROW(
                  Xpred_y0, Xcut_y0, sigma2_y0(iter - 1), sigma_mu_y0, R_Y0, Obs_y0_list,
                  p_prune, p_grow, alpha, beta, prop_prob_y0
              );
              break;
              
            case 2: // PRUNE step
              dt_y0_list[t].PRUNE(
                  Xpred_y0, Xcut_y0, sigma2_y0(iter - 1), sigma_mu_y0, R_Y0, Obs_y0_list,
                  p_prune, p_grow, alpha, beta, prop_prob_y0
              );
              break;
              
            case 3: // CHANGE step
              dt_y0_list[t].CHANGE(
                  Xpred_y0, Xcut_y0, sigma2_y0(iter - 1), sigma_mu_y0, R_Y0, Obs_y0_list,
                  p_prune, p_grow, alpha, beta, prop_prob_y0
              );
              break;
              
            default: {};
            } // end of switch
          }     // end of tree instance
          
          dt_y0_list[t].Mean_Parameter(Tree_y0, sigma2_y0(iter - 1), sigma_mu_y0, R_Y0, Obs_y0_list);
        }
        //Rcout << "Rcpp Tree1: " << rowSums(Tree1) <<  std::endl;
        

        NumericVector y_new1 = clone(rowSums(Tree_y1));
        NumericVector y_new0 = clone(rowSums(Tree_y0));
        
        
        //  Sample variance parameter
        {
          NumericVector y_new1 = clone(rowSums(Tree_y1));
          NumericVector sigma2_y1_temp = rinvgamma(1, nu / 2 + n1 / 2, nu * lambda_y / 2 + sum(pow(Y1 - y_new1, 2)) / 2);
          sigma2_y1(iter) = sigma2_y1_temp(0);
          NumericVector y_new0 = clone(rowSums(Tree_y0));
          NumericVector sigma2_y0_temp = rinvgamma(1, nu / 2 + n0 / 2, nu * lambda_y / 2 + sum(pow(Y0 - y_new0, 2)) / 2);
          sigma2_y0(iter) = sigma2_y0_temp(0);
        }
        
        

        // Num. of inclusion of each potential confounder
        NumericVector add_m1(P+1), add_m0(P+1), add_y1(P+3), add_y0(P+3);

        
        for (int t = 0; t < m; t++)
        {
          add_m1 += dt_m1_list[t].num_included(P+1);
        }
        
        for (int t = 0; t < m; t++)
        {
          add_m0 += dt_m0_list[t].num_included(P+1);
        }
        
        for (int t = 0; t < m; t++)
        {
          add_y1 += dt_y1_list[t].num_included(P+3);
        }
        for (int t = 0; t < m; t++)
        {
          add_y0 += dt_y0_list[t].num_included(P+3);
        }
        // M.H. algorithm for the alpha parameter in the dirichlet distribution (after some warm-ups)
        // if (iter < n_iter / 10)
        //{
            post_dir_alpha_m1 = rep(1.0, P+1) + add_m1;
            post_dir_alpha_m0 = rep(1.0, P+1) + add_m0;
            post_dir_alpha_y1 = rep(1.0, P+3) + add_y1;
            post_dir_alpha_y0 = rep(1.0, P+3) + add_y0;
            
        // }
        // else
        // {
        //    double p_dir_alpha = std::max(R::rnorm(dir_alpha, 0.1), pow(0.1, 10));

        //    NumericVector SumS(P);
        //    log_with_LB(SumS, prop_prob);

        //    double dir_lik_p, dir_lik, ratio;

        //    dir_lik_p =
        //        sum(SumS * (rep(p_dir_alpha / P, P) - 1)) + lgamma(sum(rep(p_dir_alpha / P, P))) - sum(lgamma(rep(p_dir_alpha / P, P)));

        //    dir_lik =
        //        sum(SumS * (rep(dir_alpha / P, P) - 1)) + lgamma(sum(rep(dir_alpha / P, P))) - sum(lgamma(rep(dir_alpha / P, P)));

        //    ratio =
        //       dir_lik_p + log(pow(p_dir_alpha / (p_dir_alpha + P), 0.5 - 1) * pow(P / (p_dir_alpha + P), 1 - 1) * abs(1 / (p_dir_alpha + P) - p_dir_alpha / pow(p_dir_alpha + P, 2))) + R::dnorm(dir_alpha, p_dir_alpha, 0.1, true) - dir_lik - log(pow(dir_alpha / (dir_alpha + P), 0.5 - 1) * pow(P / (dir_alpha + P), 1 - 1) * abs(1 / (dir_alpha + P) - dir_alpha / pow(dir_alpha + P, 2))) - R::dnorm(p_dir_alpha, dir_alpha, 0.1, true);

        //    if (ratio > log(R::runif(0, 1)))
        //    {
        //        dir_alpha = p_dir_alpha;
        //    }

        //    post_dir_alpha = rep(dir_alpha / P, P) + add + add1 + add0;
        //} // end of M.H. algorithm

        // dir_alpha_hist(iter) = dir_alpha;

        prop_prob_m1 = rdirichlet(1, post_dir_alpha_m1);
        prop_prob_m0 = rdirichlet(1, post_dir_alpha_m0);
        prop_prob_y1 = rdirichlet(1, post_dir_alpha_y1);
        prop_prob_y0 = rdirichlet(1, post_dir_alpha_y0);

        
        // Sampling E[Y(1)-Y(0)]
        if (iter > burn_in)
        {
            if (thin_count < thin)
            {
                thin_count++;
              //Rcout << "Rcpp R: " << thin_count <<  std::endl;
            }
            else
            {
                thin_count = 1;
                for (int i = 0; i < m; i++)
                {
                  //Rcout << "Rcpp R: " << 0 <<  std::endl;
                  dt_y0_list[i].Predict(Tree_y00, Xcut_y0, Xpred_y1, n1);
                  dt_y1_list[i].Predict(Tree_y11, Xcut_y1, Xpred_y0, n0);
                  //Rcout << "Rcpp R: " << 1 <<  std::endl;
                }

                // Effect(post_sample_idx) = mean(rowSums(Tree11) - rowSums(Tree00));
                // predicted_PS (_, post_sample_idx) = Rcpp::pnorm(clone(rowSums(TreePS)),0,1,true,false);
                int count_f1 = 0, count_f0 = 0;
                for (int i = 0; i < n; i++)
                {
                  if (Y_trt(i) == 1)
                  {
                    predicted_Y1 (i, post_sample_idx) = Y1(count_f1) + shift;
                    predicted_Y0 (i, post_sample_idx) = clone(rowSums(Tree_y00))(count_f1) + shift;
                    predicted_M0 (i, post_sample_idx) = Xpred_y1(count_f1,P+1) + mshift;
                    predicted_M1 (i, post_sample_idx) = Xpred_y1(count_f1,P+2) + mshift;
                    count_f1++;
                    //Rcout << "Rcpp R: " << count_f1 <<  std::endl;
                  }
                  else
                  {
                    predicted_Y1 (i, post_sample_idx) = clone(rowSums(Tree_y11))(count_f0) + shift;
                    predicted_Y0 (i, post_sample_idx) = Y0(count_f0) + shift;
                    predicted_M0 (i, post_sample_idx) = Xpred_y0(count_f0,P+1) + mshift;
                    predicted_M1 (i, post_sample_idx) = Xpred_y0(count_f0,P+2) + mshift;
                    count_f0++;
                  }
                }
                // PO_Y1  (post_sample_idx)          = mean(predicted_Y1(_, post_sample_idx));
                // PO_Y0  (post_sample_idx)          = mean(predicted_Y0(_, post_sample_idx));
                // Effect (post_sample_idx)          = PO_Y1(post_sample_idx) - PO_Y0(post_sample_idx);

                // IntegerVector ind_temp  = ifelse(add1 + add0 > 0.0, 1, 0);
                // ind(post_sample_idx, _) = clone(ind_temp); // indicator of whether confounders are included
                post_sample_idx++;
            }
        }

        Rcpp::checkUserInterrupt(); // check for break in R
    } // end of MCMC iterations

    List L = List::create(
//        Named("Effect")       = Effect,
//        Named("PO_Y1")        = PO_Y1,
//        Named("PO_Y0")        = PO_Y0,
//        Named("predicted_PS") = predicted_PS,
        Named("predicted_Y0") = predicted_Y0,
        Named("predicted_Y1") = predicted_Y1,
        Named("predicted_M0") = predicted_M0,
        Named("predicted_M1") = predicted_M1
//        Named("xpred_mult")   = xpred_mult,
//        Named("ind")          = ind,
//        Named("sigma2_1")     = sigma2_1,
//        Named("sigma2_0")     = sigma2_0,
//        Named("dir_alpha")    = dir_alpha_hist
    );

    return L;
}
