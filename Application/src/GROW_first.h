// GROW_first.h
#pragma once

#include "decision_tree.h"
#include "functions.h"
#include "subset.h"

using namespace Rcpp;

// Fun. of grow (first) alteration
void DecisionTree::GROW_first(
    const NumericMatrix& xpred,
    const NumericVector xcut[],
    double sigma2, double sigma_mu,
    const NumericVector& R,
    IntegerMatrix& Obs_list,

    double p_prune, double p_grow,
    double alpha, double beta,
    const NumericVector& prop_prob
) {
    // GROW FIRST
    const int n = xpred.nrow();
    const int P = xpred.ncol();
    const IntegerVector row_idx = Rcpp::Range(0, n - 1); // Create vector s.t. [0, 1, ..., n - 1]
    int prop_pred, prop_rule;
    double value;
    prop_pred = sample(P, 1, false, prop_prob)(0) - 1;   // pick a predictor and match index with cpp
    prop_rule = sample(xcut[prop_pred].length() - 1, 1)(0); // sample from U(2, length(xcut(prop_pred)))
    value = xcut[prop_pred](prop_rule);                     // value for separation
    
    IntegerVector R_L = row_idx[xpred(_, prop_pred) <  value];
    IntegerVector R_R = row_idx[xpred(_, prop_pred) >= value];

    NumericVector xpred_prop_pred = xpred(_, prop_pred);

    // Transition ratio (log scale)
    double TRANS = log(p_prune) - log(std::max(prop_prob(prop_pred), 0.0)) + log(xcut[prop_pred].length() - 1) - log(p_grow);

    // Likelihood ratio (log scale)
    int nlL = R_L.length();
    int nlR = R_R.length();
    double sum_R_L = sum_by_idx(R, R_L);
    double sum_R_R = sum_by_idx(R, R_R);

    double LH =
        log(sqrt(sigma2 * (sigma2 + (nlL + nlR) * sigma_mu)) / sqrt((sigma2 + nlL * sigma_mu) * (sigma2 + nlR * sigma_mu))) + (sigma_mu / (2 * sigma2) * (pow(sum_R_L, 2) / (sigma2 + sigma_mu * nlL) + pow(sum_R_R, 2) / (sigma2 + sigma_mu * nlR) - pow(sum(R), 2) / (sigma2 + sigma_mu * (nlR + nlL))));

    // Structure ratio (log scale)
    int d = 0;
    double STR = log(alpha) + 2 * log((1 - alpha / pow(2 + d, beta))) - log(pow(1 + d, beta) - alpha) + log(std::max(prop_prob(prop_pred), 0.0)) - log(xcut[prop_pred].length() - 1);

    double r = TRANS + LH + STR;

    if (r > log(R::runif(0, 1)))
    {
        // New tree structure
        this->Split    = prop_pred;
        this->Terminal = rep(0, 1);
        this->Value    = prop_rule;
        this->position = append(this->position, 2, 3);
        this->parent   = append(this->parent, 1, 1);
        this->Terminal = append(this->Terminal, 1, 1);
        this->Split    = append(this->Split, NA_INTEGER, NA_INTEGER);
        this->Value    = append(this->Value, NA_INTEGER, NA_INTEGER);
        this->MU       = append(this->MU, NA_REAL, NA_REAL);
        this->begin    = append(this->begin, 0, nlL);
        this->end      = append(this->end, nlL - 1, n - 1);

        // update obs
        R_L.sort();
        R_R.sort();
        for (int i = 0; i < nlL; i++)
            Obs_list(i, this->id) = R_L(i);
        for (int i = 0; i < nlR; i++)
            Obs_list(i + nlL, this->id) = R_R(i);
    }
} // end of GROW_first()
