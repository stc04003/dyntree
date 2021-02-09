
#include <RcppArmadillo.h>
#include <armadillo>
#include "Forest.h"
#include "globals.h"

int Forest::trainRF(std::vector<std::shared_ptr<Tree> >& trees,
		    const arma::umat& X0,
		    const arma::umat& r0,
                    const arma::umat& ids,
		    const arma::uvec& e) {
  for(size_t i = 0; i != NUM_TREE; i++) {
    trees.push_back(train(X0.rows(ids.col(i)), r0, e(ids.col(i))));
  }
  return 1;
}

std::shared_ptr<Tree> Forest::train(const arma::umat& mat1Z,
				    const arma::umat& range0,
				    const arma::uvec& e) const
{
  int n = mat1Z.n_rows;
  int P = mat1Z.n_cols;
  arma::ucube ranges = arma::zeros<arma::ucube>(MAX_NODE, P, 2);
  arma::uvec left_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec right_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec split_vars = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec split_values = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec isLeaf = arma::zeros<arma::uvec>(MAX_NODE);
  ranges.row(0) = range0.t();
  arma::field<arma::uvec> nodeSampleY(MAX_NODE);
  nodeSampleY(0) = arma::regspace<arma::uvec>(0, n-1);
  arma::umat fmat = arma::zeros<arma::umat>(1, MAX_NODE);
  arma::umat Smat = arma::zeros<arma::umat>(1, MAX_NODE);
  size_t ndcount = 0;
  size_t countsp = 0;
  int end = 0;
  while(end == 0) {
    end = split_logrank(mat1Z, left_childs, right_childs,
		     split_vars, split_values, isLeaf,
		     fmat, Smat, ranges, nodeSampleY, 
		     countsp, ndcount, e);    
    if(ndcount >= MAX_NODE - 2) {
      isLeaf( arma::find(left_childs == 0) ).ones();
      break;
    }
  }
  arma::uvec nonEmpty = arma::regspace<arma::uvec>(0, ndcount);
  //Rcpp::Rcout << "2";
  std::shared_ptr<Tree> tr(new Tree(left_childs(nonEmpty),
				    right_childs(nonEmpty),
				    split_vars(nonEmpty),
				    split_values(nonEmpty),
				    isLeaf(nonEmpty)));
  return tr;  
}

int Forest::split_logrank(const arma::umat& mat1Z,
			  arma::uvec& left_childs,
			  arma::uvec& right_childs,
			  arma::uvec& split_vars,
			  arma::uvec& split_values,
			  arma::uvec& isLeaf,
			  arma::umat& fmat,
			  arma::umat& Smat,
			  arma::ucube& ranges,
			  arma::field<arma::uvec>& nodeSampleY,
			  size_t& countsp,
			  size_t& ndcount,
			  const arma::uvec& e) const {
  int end = 0;
  int varsp = -1;
  int cutsp = 0;
  size_t nd = countsp;
  while(varsp == -1 && countsp <= ndcount) {
    nd = countsp;
    arma::ivec bestSp(3);
    bestSp = find_split_logrank(nd, mat1Z, isLeaf,
				ranges, nodeSampleY,
				fmat, Smat, ndcount, e);
    varsp = bestSp(1);
    cutsp = bestSp(2);
    if(varsp == -1) {
      isLeaf(nd) = 1;
      while(countsp <= ndcount) {
        countsp++;
        if(isLeaf(countsp) == 0) break;
      }
    }
  }
  int n = mat1Z.n_rows;
  if(varsp != -1) {
    split_vars(nd) = varsp;
    split_values(nd) = cutsp;
    arma::uword ndc1 = ndcount + 1;
    arma::uword ndc2 = ndcount + 2;
    left_childs(nd) = ndc1;
    right_childs(nd) = ndc2;
    arma::uvec nodeSampleYnd = std::move(nodeSampleY(nd));
    arma::uvec zvarspsub = mat1Z( varsp*n + nodeSampleYnd );
    nodeSampleY(ndc1) = nodeSampleYnd( arma::find(zvarspsub <= cutsp) );
    nodeSampleY(ndc2) = nodeSampleYnd( arma::find(zvarspsub > cutsp) );    
    ranges.row(ndc1) = ranges.row(nd);
    ranges.row(ndc2) = ranges.row(nd);
    ranges(ndc2,varsp,0) = cutsp+1;
    ranges(ndc1,varsp,1) = cutsp;
    ndcount = ndcount + 2;
    while(countsp <= ndcount) {
      countsp++;
      if(isLeaf(countsp) == 0) break;
    }
  } else {
    end = 1;
  }
  return end;
}

arma::ivec Forest::find_split_logrank(size_t nd,
				      const arma::umat& mat1Z,
				      const arma::uvec& isLeaf,
				      const arma::ucube& ranges,
				      const arma::field<arma::uvec>& nodeSampleY,
				      arma::umat& fmat,
				      arma::umat& Smat,
				      int ndcount,
				      const arma::uvec& e) const {
  int P = mat1Z.n_cols;
  int n = mat1Z.n_rows;
  int varsp = -1;
  int cutsp = 0;
  double LGmax = 0;
  double LGTemp = 0;
  arma::uvec spSet = arma::shuffle( arma::regspace<arma::uvec>(0,P-1) );
  for(auto p : spSet.head(mtry)) {
    arma::uvec indY = nodeSampleY(nd);
    int nj = indY.size();
    arma::uvec fLSum = arma::zeros<arma::uvec>(nj);
    arma::uvec fRSum = arma::zeros<arma::uvec>(nj);
    fRSum = e(indY);
    arma::vec SRSum = arma::regspace<arma::vec>(nj, 1);
    arma::vec SLSum = arma::zeros<arma::vec>(nj);
    arma::uvec rangeCut = arma::regspace<arma::uvec>(ranges(nd, p, 0), ranges(nd, p, 1));

    for(auto cu : rangeCut) {
      for (size_t j = 0; j < nj; j++) {
	int indYj = indY(j);
	size_t z = mat1Z(indYj, p);
	if(z == cu) {
	  for (int k = 0; k <= j; k++) {
	    SLSum(k) = SLSum(k) + 1;
	    SRSum(k) = SRSum(k) - 1;
	  }
	  fLSum(j) = fLSum(j) + fRSum(j);
	  fRSum(j) = fRSum(j) - fRSum(j);
	}
      }
      if(sum(fLSum) < MIN_NODE1 || sum(fRSum) < MIN_NODE1) {
        LGTemp = 0;
      } else {
	double w1 = 0;
	double w2 = 0;
	arma::vec vec1 = arma::zeros<arma::vec>(nj);
	arma::vec vec2 = arma::zeros<arma::vec>(nj);
	vec1 = fLSum - (fLSum + fRSum) % SLSum / ( SLSum + SRSum );
	vec2 = SRSum % SLSum % (fLSum + fRSum) % (SLSum + SRSum - fLSum - fRSum) /
	  (SLSum + SRSum) / (SLSum + SRSum) / (SLSum + SRSum - 1);
	w1 = arma::sum(vec1.elem(find_finite(vec1)));
	w2 = arma::sum(vec2.elem(find_finite(vec2)));
	LGTemp = w1 * w1 / w2;
      }
      if(LGTemp > LGmax) {
        LGmax = LGTemp;
        varsp = p;
        cutsp = cu;
        // fmat.col(ndcount + 1) = fLSum;
        // fmat.col(ndcount + 2) = fRSum;
        // Smat.col(ndcount + 1) = SLSum;
        // Smat.col(ndcount + 2) = SRSum;
      }
    }
  }
  arma::ivec vecsp(3);
  if(varsp == -1) {
    vecsp(0) = 0;
    vecsp(1) = -1;
    vecsp(2) = 0;
  } else {
    vecsp(0) = 1;
    vecsp(1) = varsp;
    vecsp(2) = cutsp;
  }
  return vecsp;
}
