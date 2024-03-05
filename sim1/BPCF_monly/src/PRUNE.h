// PRUNE.h
#pragma once

#include "decision_tree.h"
#include "functions.h"
#include "subset.h"

using namespace Rcpp;

// Fun. of PRUNE alteration
void DecisionTree::PRUNE(
    const NumericMatrix& xpred,
    const NumericVector xcut[],
    double sigma2, double sigma_mu,
    const NumericVector& R,
    IntegerMatrix& Obs_list,

    double p_prune, double p_grow,
    double alpha, double beta,
    const NumericVector& prop_prob
) {

    // PRUNE()
    const int P = xpred.ncol();
    IntegerVector col_idx = Rcpp::Range(0, P - 1); // Create vector s.t. [0, 1, ..., P - 1]

    // find nodes with two terminal child nodes (singly internal parent nodes)
    IntegerVector singly_position = this->singly_position(); // vector with idx of singly positions
    int singly_inode;                                        // pick a singly node
    switch (singly_position.length())
    {
        case 0:
            // no singly nodes -> PRUNE is not possible
            return;
        case 1:
            // 1 singly node -> no need for sampling
            singly_inode = singly_position(0);
            break;
        default:
            // sample idx of singly node
            singly_inode = sample(singly_position, 1)(0);
    }

    // find indices of Obs_list of proposed singly node
    int subset_ind = which(this->position, singly_inode);           // get position of singly node
    int prop_pred  = this->Split(subset_ind);                       // use position to get begin and end
    int begin      = this->begin(subset_ind);
    int end        = this->end(subset_ind);
    IntegerVector Obs_begin_end(end - begin + 1);                   // Create empty vector of size = (end - begin + 1)
    subset_by_range(Obs_begin_end, Obs_list, begin, end, this->id); // Obs_begin_end = Obs_list[begin:end, id]

    // repeat for left child node
    subset_ind  = which(this->position, 2 * singly_inode);
    int begin_L = this->begin(subset_ind);
    int end_L   = this->end(subset_ind);
    IntegerVector temp_L(end_L - begin_L + 1);
    subset_by_range(temp_L, Obs_list, begin_L, end_L, this->id);

    // repeat for right child node
    subset_ind  = which(this->position, 2 * singly_inode + 1);
    int begin_R = this->begin(subset_ind);
    int end_R   = this->end(subset_ind);
    IntegerVector temp_R(end_R - begin_R + 1);
    subset_by_range(temp_R, Obs_list, begin_R, end_R, this->id);

    // LogicalVector flag = pred_enough_unique(xpred, Obs_begin_end); // Boolean Vector with 1 if variable has enough unique values and 0 otherwise
    // IntegerVector enough_unique = col_idx[flag];                   // idx of column idx with enough unique values
    // NumericVector enough_unique_prob = prop_prob[flag];            // subset prop_prob w.r.t. flag

    double prop_pred_prob = prop_prob(prop_pred) / sum(prop_prob);
    NumericVector xpred_prop_pred(Obs_begin_end.length());           // Create empty vector of size = Obs_begin_end.length()
    subset_by_idx(xpred_prop_pred, xpred, Obs_begin_end, prop_pred); // xpred_prop_pred = xpred[Obs_begin_end, prop_pred]
    
    double min_xpred = min(xpred_prop_pred);
    double max_xpred = max(xpred_prop_pred);

    int unique_len = 0;
    for (int i = 0; i < xcut[prop_pred].length(); i++)
    {
        if (xcut[prop_pred](i) > min_xpred &&  xcut[prop_pred](i) <= max_xpred)
        {
            unique_len += 1;
        }
    }

    // int unique_len = unique(xpred_prop_pred).length();

    // transition ratio
    double TRANS = log(p_grow) - (log(this->terminal_nodes().length() - 1)) + log(std::max(prop_pred_prob, 0.0)) - log(unique_len) - log(p_prune) + log(singly_position.length());

    // likelihood ratio
    int nlL = temp_L.length();
    int nlR = temp_R.length();

    double sum_R_temp_L = sum_by_idx(R, temp_L);
    double sum_R_temp_R = sum_by_idx(R, temp_R);
    double sum_R_union  = sum_union(R, temp_L, temp_R);

    double LH =
        log(sqrt((sigma2 + nlL * sigma_mu) * (sigma2 + nlR * sigma_mu)) / sqrt(sigma2 * (sigma2 + (nlL + nlR) * sigma_mu))) + (sigma_mu / (2 * sigma2) * (-pow(sum_R_temp_L, 2) / (sigma2 + nlL * sigma_mu) - pow(sum_R_temp_R, 2) / (sigma2 + nlR * sigma_mu) + pow(sum_R_union, 2) / (sigma2 + (nlR + nlL) * sigma_mu)));

    // structure ratio
    int d = 1;
    while (singly_inode >= pow(2, d))
        d += 1;
    d -= 1;

    double STR = -log(alpha) - 2 * log(1 - alpha / pow(2 + d, beta)) + log(pow(1 + d, beta) - alpha) - log(std::max(prop_pred_prob, 0.0)) + log(unique_len);

    double r = TRANS + LH + STR;

    if (r > log(R::runif(0, 1)))
    {

        int subset_inds = which(this->parent, singly_inode);

        this->position.erase(subset_inds, 2 + subset_inds);
        this->parent.erase(subset_inds, 2 + subset_inds);
        this->Terminal.erase(subset_inds, 2 + subset_inds);
        this->Split.erase(subset_inds, 2 + subset_inds);
        this->Value.erase(subset_inds, 2 + subset_inds);
        this->MU.erase(subset_inds, 2 + subset_inds);
        this->begin.erase(subset_inds, 2 + subset_inds);
        this->end.erase(subset_inds, 2 + subset_inds);

        subset_ind = which(this->position, singly_inode);
        this->Split(subset_ind) = NA_INTEGER;
        this->Value(subset_ind) = NA_INTEGER;
        this->Terminal(subset_ind) = 1;

        Obs_begin_end.sort();
        for (int i = 0; i < Obs_begin_end.length(); i++)
            Obs_list(begin + i, this->id) = Obs_begin_end(i);
    }
}
