// Predict.h
#pragma once

#include "decision_tree.h"
#include "functions.h"
#include "subset.h"

using namespace Rcpp;

// Prediction (binary exposure)
void DecisionTree::Predict
(
    NumericMatrix& Tree,
    const NumericVector xcut[],
    const NumericMatrix& xpred_mult,
    const int n
) {
    // predict for separate model
    for (int i = 0; i < xpred_mult.nrow(); i++)
    {
        // start from root node
        int pos = 1; // position of root node
        int idx = 0; // index of root node
        while (Terminal(idx) != 1)
        {
            int    split = Split(idx);
            int    value = Value(idx);
            double rule  = xcut[split](value);
            double xpred = xpred_mult(i, split);
            if (xpred < rule)
            {
                // move to left child node
                // then update pos and idx
                pos = 2 * pos;
                idx = which(position, pos);
            }
            else
            {
                // move to right child node
                // then update pos and idx
                pos = 2 * pos + 1;
                idx = which(position, pos);
            }
        }
        Tree(i, this->id) = MU(idx);
    }
}

