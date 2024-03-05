// GROW.h
#pragma once

#include "decision_tree.h"
#include "functions.h"
#include "subset.h"

using namespace Rcpp;

// Fun. of grow alteration
void DecisionTree::GROW(
    const NumericMatrix& xpred,
    const NumericVector xcut[],
    double sigma2, double sigma_mu,
    const NumericVector& R,
    IntegerMatrix& Obs_list,

    double p_prune, double p_grow,
    double alpha, double beta,
    const NumericVector& prop_prob
) {

    // GROW()
    const int P = xpred.ncol();
    IntegerVector col_idx = Rcpp::Range(0, P - 1); // Create vector s.t. [0, 1, ..., P - 1]

    // equivalent to R code `which(dt$Terminal==1)`
    IntegerVector tnode_idx = this->terminal_nodes(); // Vector with idx of terminal nodes
    int prop_tnode = sample(tnode_idx, 1)(0); // randomly sample terminal node

    // find begin and end for chosen terminal node
    int begin = this->begin(prop_tnode); // begin value of proposed node
    int end   = this->end(prop_tnode);     // end value of proposed node
    if (end - begin + 1 < 2)             // end - begin < 1 means node with only one observation
    {
        // return the current tree if there is no covariate with enough unique values
        return;
    }

    IntegerVector Obs_ind(end - begin + 1);                   // create empty vector of size = (end - begin + 1)
    subset_by_range(Obs_ind, Obs_list, begin, end, this->id); // subset Obs_list[begin:end]

    // find covariate with enough unique values
    // LogicalVector flag = pred_enough_unique(xpred, Obs_ind);  // Boolean Vector with 1 if variable has enough unique values and 0 otherwise
    // IntegerVector enough_unique = col_idx[flag];              // idx of column idx with enough unique values
    // NumericVector enough_unique_prob = prop_prob[flag];       // subset prop_prob w.r.t. flag

    // pick a predictor -> already index
    int prop_pred = sample(P, 1, false, prop_prob)(0) - 1; // sample prop_pred w.r.t. enough_unique_prob
    NumericVector xpred_prop_pred(Obs_ind.length());                        // create empty vector of size = Obs_ind.length()
    subset_by_idx(xpred_prop_pred, xpred, Obs_ind, prop_pred);              // xpred_prop_pred = xpred[Obs_ind, prop_pred]
   
    
    double value = sample(xcut[prop_pred], 1)(0);
    double min_xpred = min(xpred_prop_pred);
    double max_xpred = max(xpred_prop_pred);
    if (min_xpred >= value || max_xpred < value)
    {
        return; // return the current tree
    }

    // NumericVector unique_xpred = unique(xpred_prop_pred); // unique values of xpred_prop_pred

    // int unique_len = unique_xpred.length(); // number of unique values
    // if (unique_len == 1)
    // {
    //    return; // return the current tree if there is no covariate with enough unique values
    //}

    // sample from  unique values of xpred_prop_pred and it should not be minimum value
    // double min_xpred = min(xpred_prop_pred);
    // double value;
    // do
    // {
    //    value = sample(unique_xpred, 1)(0);
    // } while (value == min_xpred);
    int unique_len = 0;
    for (int i = 0; i < xcut[prop_pred].length(); i++)
    {
        if (xcut[prop_pred](i) > min_xpred &&  xcut[prop_pred](i) <= max_xpred)
        {
            unique_len += 1;
        }
    }
            
    int prop_rule = which(xcut[prop_pred], value); // index of xcut[prop_pred] with value

    IntegerVector R_L = Obs_ind[xpred_prop_pred <  value];
    IntegerVector R_R = Obs_ind[xpred_prop_pred >= value];
    int nlL = R_L.length();
    int nlR = R_R.length();

    // Temporary new tree structure
    DecisionTree dt_new;   // create new tree
    this->copy_to(dt_new); // copy structure of this tree to temporary tree
    // update temporary tree as if new node is accepted
    int new_position = dt_new.position(prop_tnode);
    dt_new.Terminal(prop_tnode) = 0;

    dt_new.position = append(dt_new.position, 2 * new_position, 2 * new_position + 1);
    dt_new.parent   = append(dt_new.parent, new_position, new_position);
    dt_new.Terminal = append(dt_new.Terminal, 1, 1);
    dt_new.Split    = append(dt_new.Split, NA_INTEGER, NA_INTEGER);
    dt_new.Value    = append(dt_new.Value, NA_INTEGER, NA_INTEGER);
    dt_new.MU       = append(dt_new.MU, NA_REAL, NA_REAL);
    dt_new.begin    = append(dt_new.begin, begin, begin + nlL);
    dt_new.end      = append(dt_new.end, begin + nlL - 1, end);

    // Find internal nodes with two terminal child nodes 
    // equivalent expression in R is
    // count <- length(which(table(dt.new$parent[which(dt.new$Terminal==1)])==2))
    IntegerVector tnode_idx_new = dt_new.terminal_nodes(); // terminal nodes of new tree
    int count = dt_new.terminal_parents(tnode_idx_new);    // count updated number of singly nodes

    // Transition ratio
    double sum_prop_prob = sum(prop_prob);
    double sum_R_L     = sum_by_idx(R, R_L);
    double sum_R_R     = sum_by_idx(R, R_R);
    double sum_R_union = sum_union(R, R_L, R_R);

    double TRANS = log(p_prune) + log(tnode_idx.length()) - log(std::max(prop_prob(prop_pred) / sum_prop_prob, 0.0)) + log(unique_len - 1) - log(p_grow) - log(count);

    double LH =
        log(sqrt(sigma2 * (sigma2 + (nlL + nlR) * sigma_mu)) / sqrt((sigma2 + nlL * sigma_mu) * (sigma2 + nlR * sigma_mu))) + (sigma_mu / (2 * sigma2) * (pow(sum_R_L, 2) / (sigma2 + sigma_mu * nlL) + pow(sum_R_R, 2) / (sigma2 + sigma_mu * nlR) - pow(sum_R_union, 2) / (sigma2 + sigma_mu * (nlR + nlL))));

    // Structure ratio
    int d = 1;
    while (this->position(prop_tnode) >= pow(2, d))
        d += 1;
    d -= 1;
    double STR = log(alpha) + 2 * log(1 - alpha / pow(2 + d, beta)) - log(pow(1 + d, beta) - alpha) + log(std::max(prop_prob(prop_pred) / sum_prop_prob, 0.0)) - log(unique_len - 1);

    double r = TRANS + LH + STR;

    if (r > log(R::runif(0, 1)))
    {
        this->copy_from(dt_new); // copy temporary tree to current tree
        this->Split(prop_tnode) = prop_pred;
        this->Value(prop_tnode) = prop_rule;

        // update obs
        R_L.sort();
        R_R.sort();
        for (int i = 0; i < nlL; i++)
            Obs_list(i + begin, this->id) = R_L(i);
        for (int i = 0; i < nlR; i++)
            Obs_list(i + begin + nlL, this->id) = R_R(i);
    }
} // end of GROW()
