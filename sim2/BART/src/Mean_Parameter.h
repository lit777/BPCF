// Mean_Parameter.h
#pragma once

#include "decision_tree.h"
#include "functions.h"
#include "subset.h"

using namespace Rcpp;

// Sampling mean parameters
void DecisionTree::Mean_Parameter(
    NumericMatrix& Tree,
    double sigma2,
    double sigma_mu,
    const NumericVector& R,
    const IntegerMatrix& Obs_list
) {

    NumericVector T(this->n);
    IntegerVector tnode_idx = this->terminal_nodes();
    for (int i = 0; i < tnode_idx.length(); i++)
    {
        int begin = this->begin(tnode_idx(i)); // match index
        int end   = this->end(tnode_idx(i));

        IntegerVector Obs_ind(end - begin + 1);
        if (begin == end)
            Obs_ind = Obs_list(begin, this->id);
        else
            subset_by_range(Obs_ind, Obs_list, begin, end, this->id); // Obs_ind = Obs_list[begin:end, ]

        double Var  = 1 / (1/sigma_mu + Obs_ind.length()/sigma2);
        double Mean = Var * (sum_by_idx(R, Obs_ind) / sigma2);

        this->MU(tnode_idx(i)) = R::rnorm(Mean, sqrt(Var));
        T[Obs_ind] = this->MU(tnode_idx(i));
    }
    Tree(_, this->id) = clone(T);
}
