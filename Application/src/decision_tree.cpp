// decision_tree.cpp

#include "decision_tree.h"
#include "GROW_first.h"
#include "GROW.h"
#include "PRUNE.h"
#include "CHANGE.h"
#include "Mean_Parameter.h"
#include "Predict.h"

using namespace Rcpp;
using namespace std;

NumericVector DecisionTree::num_included(const int P)
{
    // count number of included variables
    NumericVector res(P);
    for (auto i = this->Split.begin(); i != this->Split.end(); i++)
    {
        if (*i != NA_INTEGER)
            res(*i) += 1.0;
    }
    return res;
}

IntegerVector DecisionTree::terminal_nodes()
{
    // method to find idx of terminal nodes
    IntegerVector tnode_idx = Rcpp::Range(0, this->Terminal.length() - 1);
    tnode_idx = tnode_idx[this->Terminal == 1];
    return tnode_idx;
};

int DecisionTree::terminal_parents(IntegerVector tnode_idx)
{
    // method to count internal nodes with two terminal child nodes
    int count = 0;
    map<int, int> table; // use map to count parent
    for (int i = 0; i < tnode_idx.length(); i++)
    { 
        // equivalent to table() in R
        table[this->parent[tnode_idx(i)]]++;
    }
    for (auto i = table.begin(); i != table.end(); i++)
    {
        if (i->second == 2)
            count += 1; // i->first : key, i->second : value
    }                   // count value with 2
    return count;
};

IntegerVector DecisionTree::singly_position()
{
    // return index of singly positions
    IntegerVector singly_position;
    map<int, int> table;
    for (int i = 0; i < this->Terminal.length(); i++)
    { 
        // equivalent to table() in R
        if (this->Terminal(i) == 1)
        {
            table[this->parent(i)]++;
        }
    }
    for (auto i = table.begin(); i != table.end(); i++)
    {
        if (i->second == 2)
        {                                        // count value with 2
            singly_position.push_back(i->first); // i->first : key, i->second : value
        }
    }
    return singly_position;
}
