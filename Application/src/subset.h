// subset.h
#pragma once

#include <Rcpp.h>

using namespace Rcpp;

// this header includes functions related to subsetting

void subset_by_idx(IntegerVector& res, const IntegerMatrix& src, const IntegerVector& idx, const int& col)
{
    for (int i = 0; i < idx.length(); i++)
        res(i) = src(idx(i), col);
};

void subset_by_idx(NumericVector& res, const NumericMatrix& src, const IntegerVector& idx, const int& col)
{
    for (int i = 0; i < idx.length(); i++)
        res(i) = src(idx(i), col);
};

void subset_by_range(IntegerVector& res, const IntegerMatrix& src, const int start, const int end, const int col)
{
    for (int i = 0; i < end - start + 1; i++)
        res(i) = src(start + i, col);
}

void subset_by_range(NumericVector& res, const NumericMatrix& src, const int start, const int end, const int col)
{
    for (int i = 0; i < end - start + 1; i++)
        res(i) = src(start + i, col);
}

NumericMatrix subset_by_row(const NumericMatrix& src, const LogicalVector& idx)
{
    NumericMatrix res(sum(idx), src.ncol());
    IntegerVector idx_int = Rcpp::Range(0, idx.length() - 1);
    idx_int = idx_int[idx];

    for (int i = 0; i < idx_int.length(); i++)
        res(i, _) = src(idx_int(i), _);
    return res;
}
