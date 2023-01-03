// CHANGE.h
#pragma once

#include "decision_tree.h"
#include "functions.h"
#include "subset.h"

using namespace Rcpp;

// Fun. of CHANGE alteration
void DecisionTree::CHANGE(
    const NumericMatrix& xpred,
    const NumericVector xcut[],
    double sigma2, double sigma_mu,
    const NumericVector& R,
    IntegerMatrix& Obs_list,

    double p_prune, double p_grow,
    double alpha, double beta,
    const NumericVector& prop_prob
) {

    // CHANGE()
    const int P = xpred.ncol();
    IntegerVector col_idx = Rcpp::Range(0, P - 1); // Create vector s.t. [0, 1, ..., P - 1]

    // find nodes with two terminal child nodes (singly internal parent nodes)
    IntegerVector singly_position = this->singly_position(); // vector with idx of singly positions
    int singly_inode;                                        // pick a singly node
    switch (singly_position.length())
    {
        case 0:
            // no singly nodes -> CHANGE is not possible
            return;
        case 1:
            // 1 singly node -> no need for sampling
            singly_inode = singly_position(0);
            break;
        default:
            // sample idx of singly node
            singly_inode = sample(singly_position, 1)(0);
    }

    int subset_ind = which(this->position, singly_inode);     // get position of singly node
    int begin = this->begin(subset_ind);                      // use position to get begin and end
    int end   = this->end(subset_ind);
    IntegerVector Obs_ind(end - begin + 1);                   // Create empty vector of size = (end - begin + 1)
    subset_by_range(Obs_ind, Obs_list, begin, end, this->id); // Obs_ind = Obs_list[begin:end, id]

    // LogicalVector flag = pred_enough_unique(xpred, Obs_ind);  // Boolean Vector with 1 if variable has enough unique values and 0 otherwise
    // IntegerVector enough_unique = col_idx[flag];              // idx of column idx with enough unique values
    // NumericVector enough_unique_prob = prop_prob[flag];       // subset prop_prob w.r.t. flag

    // pick a predictor -> already index
    int prop_pred = sample(P, 1, false, prop_prob)(0) - 1; // sample prop_pred
    // int prop_pred = sample(enough_unique, 1, false, enough_unique_prob)(0); // sample prop_pred w.r.t. enough_unique_prob

    NumericVector xpred_prop_pred(Obs_ind.length());           // Create empty vector of size = Obs_ind.length()
    subset_by_idx(xpred_prop_pred, xpred, Obs_ind, prop_pred); // xpred_prop_pred = xpred[Obs_ind, prop_pred]

    double value = sample(xcut[prop_pred], 1)(0);
    double min_xpred = min(xpred_prop_pred);
    double max_xpred = max(xpred_prop_pred);
    if (min_xpred >= value || max_xpred < value)
    {
        return; // return the current tree
    }

    
    // NumericVector unique_xpred = unique(xpred_prop_pred);      // unique values of xpred_prop_pred
    // int unique_len = unique_xpred.length();                    // number of unique values
    // if (unique_len == 1)
    // {
    //    return; // return the current tree if there is no covariate with enough unique values
    // }

    // sample from  unique values of xpred_prop_pred and it should not be minimum value
    // double min_xpred = min(xpred_prop_pred);
    // double value;
    // do
    // {
    //     value = sample(unique_xpred, 1)(0);
    // } while (value == min_xpred);
    // int prop_rule = which(xcut[prop_pred], value);
    int unique_len = 0;
    for (int i = 0; i < xcut[prop_pred].length(); i++)
    {
        if (xcut[prop_pred](i) > min_xpred &&  xcut[prop_pred](i) <= max_xpred)
        {
            unique_len += 1;
        }
    }
            
    int prop_rule = which(xcut[prop_pred], value); // index of xcut[prop_pred] with value


    IntegerVector RL_star = Obs_ind[xpred_prop_pred < value];
    IntegerVector RR_star = Obs_ind[xpred_prop_pred >= value];

    // find terminal nodes from proposed singly node
    // then find begin and end to subset Obs_list
    int begin_idx, end_idx;
    begin_idx = this->begin[which(this->position, 2 * singly_inode)];
    end_idx   = this->end[which(this->position,   2 * singly_inode)];
    IntegerVector RL(end_idx - begin_idx + 1);
    subset_by_range(RL, Obs_list, begin_idx, end_idx, this->id);

    // repeat again for other node
    begin_idx = this->begin[which(this->position, 2 * singly_inode + 1)];
    end_idx   = this->end[which(this->position,   2 * singly_inode + 1)];
    IntegerVector RR(end_idx - begin_idx + 1);
    subset_by_range(RR, Obs_list, begin_idx, end_idx, this->id);

    // Likelihood ratio
    int nlL = RL.length();
    int nlR = RR.length();
    int nlL_star = RL_star.length();
    int nlR_star = RR_star.length();
    double sum_R_RL_star = sum_by_idx(R, RL_star);
    double sum_R_RR_star = sum_by_idx(R, RR_star);
    double sum_R_RL = sum_by_idx(R, RL);
    double sum_R_RR = sum_by_idx(R, RR);

    double LH = log(sqrt((sigma2 / sigma_mu + nlL) * (sigma2 / sigma_mu + nlR))) - log(sqrt((sigma2 / sigma_mu + nlL_star) * (sigma2 / sigma_mu + nlR_star))) + (0.5 / sigma2 * (pow(sum_R_RL_star, 2) / (sigma2 / sigma_mu + nlL_star) + pow(sum_R_RR_star, 2) / (sigma2 / sigma_mu + nlR_star) - pow(sum_R_RL, 2) / (sigma2 / sigma_mu + nlL) - pow(sum_R_RR, 2) / (sigma2 / sigma_mu + nlR)));

    double r = LH;

    if (r > log(R::runif(0, 1)))
    {
        // New tree structure
        subset_ind = which(this->position, singly_inode);
        this->Terminal(subset_ind) = 0;
        this->Split(subset_ind) = prop_pred;
        this->Value(subset_ind) = prop_rule;

        subset_ind = which(this->position, 2 * singly_inode);
        this->begin(subset_ind) = begin;
        this->end(subset_ind) = begin + nlL_star - 1;

        subset_ind = which(this->position, 2 * singly_inode + 1);
        this->begin(subset_ind) = begin + nlL_star;
        this->end(subset_ind) = end;

        RL_star.sort();
        RR_star.sort();
        for (int i = 0; i < nlL_star; i++)
            Obs_list(i + begin, this->id) = RL_star(i);
        for (int i = 0; i < nlR_star; i++)
            Obs_list(i + begin + nlL_star, this->id) = RR_star(i);
    }
}
