// decision_tree.h
#pragma once

#include <Rcpp.h>

using namespace Rcpp;

class DecisionTree
{

private:
    IntegerVector position = rep(1, 1);          // position of each node
    IntegerVector parent   = rep(NA_INTEGER, 1); // position of parent node
    IntegerVector Terminal = rep(1, 1);          // 1 if terminal node, 0 otherwise
    IntegerVector Split    = rep(NA_INTEGER, 1); // idx of split variable
    IntegerVector Value    = rep(NA_INTEGER, 1); // idx of xcut
    NumericVector MU       = rep(NA_REAL, 1);    // leaf value of terminal node
    IntegerVector begin    = rep(0, 1);          // start idx for Obs_list
    IntegerVector end      = rep(0, 1);          // start idx for Obs_list
    int n;                                       // Number of rows of X
    int id;                                      // ID of each tree

public:
    DecisionTree(){}; // default constructor
   //  DecisionTree(int n, int id, bool mar_exp = false, bool mar_out = false)
    DecisionTree(int n, int id)
    {
        this->n       = n;
        this->id      = id;
        this->end     = rep(n - 1, 1);
    //    this->mar_exp = mar_exp;
    //    this->mar_out = mar_out;
    }
    void copy_from(const DecisionTree& dt)
    {
        this->position = clone(dt.position);
        this->parent   = clone(dt.parent);
        this->Terminal = clone(dt.Terminal);
        this->Split    = clone(dt.Split);
        this->Value    = clone(dt.Value);
        this->MU       = clone(dt.MU);
        this->begin    = clone(dt.begin);
        this->end      = clone(dt.end);
        this->n        = dt.n;
        this->id       = dt.id;
    //    this->mar_exp  = dt.mar_exp;
    //    this->mar_out  = dt.mar_out;
    }
    void copy_to(DecisionTree& dt)
    {
        dt.position = clone(this->position);
        dt.parent   = clone(this->parent);
        dt.Terminal = clone(this->Terminal);
        dt.Split    = clone(this->Split);
        dt.Value    = clone(this->Value);
        dt.MU       = clone(this->MU);
        dt.begin    = clone(this->begin);
        dt.end      = clone(this->end);
        dt.n        = this->n;
        dt.id       = this->id;
    //    dt.mar_exp  = this->mar_exp;
    //    dt.mar_out  = this->mar_out;
    }

    inline int  length()          { return this->position.length(); };
   //  inline bool is_mar()          { return this->mar_exp || this->mar_out; };
   //  inline bool is_mar_exposure() { return this->mar_exp; };
   //  inline bool is_mar_outcome()  { return this->mar_out; };

    NumericVector num_included(const int P);
    IntegerVector singly_position();
    IntegerVector terminal_nodes();
    int terminal_parents(IntegerVector tnode_idx);

    void GROW_first(
        const NumericMatrix& xpred,
        const NumericVector xcut[],
        double sigma2, double sigma_mu,
        const NumericVector& R,
        IntegerMatrix& Obs_list,

        double p_prune, double p_grow,
        double alpha, double beta,
        const NumericVector& prop_prob
    );
    void GROW(
        const NumericMatrix& xpred,
        const NumericVector xcut[],
        double sigma2, double sigma_mu,
        const NumericVector& R,
        IntegerMatrix& Obs_list,

        double p_prune, double p_grow,
        double alpha, double beta,
        const NumericVector& prop_prob
    );
    void PRUNE(
        const NumericMatrix& xpred,
        const NumericVector xcut[],
        double sigma2, double sigma_mu,
        const NumericVector& R,
        IntegerMatrix& Obs_list,

        double p_prune, double p_grow,
        double alpha, double beta,
        const NumericVector& prop_prob
    );
    void CHANGE(
        const NumericMatrix& xpred,
        const NumericVector xcut[],
        double sigma2, double sigma_mu,
        const NumericVector& R,
        IntegerMatrix& Obs_list,

        double p_prune, double p_grow,
        double alpha, double beta,
        const NumericVector& prop_prob
    );
    void Mean_Parameter(
        NumericMatrix& Tree,
        double sigma2, double sigma_mu,
        const NumericVector& R,
        const IntegerMatrix& Obs_list
    );
    void Predict(
        NumericMatrix& Tree,
        const NumericVector xcut[],
        const NumericMatrix& xpred_mult,
        const int n
    );

};
