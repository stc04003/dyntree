#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include "globals.h"
#include "Tree.h"
#include "Forest.h"
#include "ForestPrediction.h"
#include "TreeGrow.h"
#include "TreePrediction.h"

#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#endif

//' @noRd
// [[Rcpp::export]]
SEXP dynforest_C(const arma::umat& X0,
		 const arma::uvec& D0,
		 const arma::umat& r0,
		 int numTree,
		 int minNode1,
		 int minSplit1,
		 int maxNode,
		 int mtry) {
  int n = X0.n_rows;
  std::vector<std::shared_ptr<Tree> > trees;
  trees.reserve(numTree);
  Forest ff(numTree, maxNode, minNode1, minSplit1, mtry);
  arma::umat ids0(n, numTree);
  ff.sampleWithReplacementSplit(n, n, ids0);
  // Bootstrapping
  arma::umat id1 = ids0;//ids.rows( arma::regspace<arma::uvec>(0, s-1)  );
  ff.trainRF(trees, X0, r0, id1, D0);

  arma::field<arma::umat> treeList(numTree);
  std::vector<std::shared_ptr<Tree> >::const_iterator it;
  int i = 0;
  for(it = trees.begin(); it != trees.end(); it++, i++) {
    std::shared_ptr<Tree> tt = *it;
    const arma::uvec& vars0 = tt->get_split_vars();
    arma::umat treeMat(vars0.n_elem,5);
    treeMat.col(0) = vars0;
    treeMat.col(1) = tt->get_split_values();
    treeMat.col(2) = tt->get_left_childs();
    treeMat.col(3) = tt->get_right_childs();
    treeMat.col(4) = tt->get_isLeaf();
    treeList(i) = treeMat;
  }
  // use bootstrap observations
  arma::umat id2 = ids0;
  //ids.rows( arma::regspace<arma::uvec>(0, n-1)  );
  ForestPrediction fp(X0, id2, trees, n);
  return Rcpp::List::create(Rcpp::Named("trees") = treeList,
                            Rcpp::Named("nodeLabel") = fp.get_nodeLabel(),
                            Rcpp::Named("nodeSize") = fp.get_nodeSize(),
                            Rcpp::Named("nodeMap") = fp.get_nodeMap(),
                            Rcpp::Named("boot.id") = id2);
}

// [[Rcpp::export]]
SEXP predict_dynforest_C(const arma::mat& zraw0,
			 const arma::vec& y0,
			 const arma::uvec& e0,
			 const Rcpp::List& forestobj,
			 const arma::umat& matX) {
  arma::umat z0(zraw0.n_rows, arma::sum(e0));
  ForestPrediction::transformZ(zraw0, z0, matX, e0);
  arma::vec sy = ForestPrediction::getSurvival(z0,
                                               y0,
                                               e0,
                                               forestobj[2],
                                               forestobj[1],
                                               forestobj[3],
                                               forestobj[4],
                                               forestobj[0]);
  return Rcpp::wrap(sy);
}

//' Main function for tree.
//' Data are prepared outside of the function.
//' 
//' @param mat1f0 is a K x n matrix; K is the number of cutpoint
//'               for i = 1, ..., n, k = 1, ..., K,
//'               the transpose of it has the (i, j)th element of f_i(t_j) 
//' @param mat1Z0 is a n x p matrix; p is the number of covariates
//'               instead of covaraites values, this gives the 'order' that indicates which
//'               time intervals the covaraite values lie in.
//' @param mat2Z0 is a list of length K. The lth list consists of a r x p matrix that
//'               specified the order of covariates for subjects at risk at tk;
//'               r is the number of subjects at risk at t_l.
//' @param r0     is a 2 x p matrix. Each column specified the smallest and the largest intervals
//'               the pth covariates lie in.
//' @param zt0    is a list of length equal to the numbers of events (\sum\Delta).
//'               This is created similar to `mat2Z` but it stores the order of covariates for
//'               subjects at risk at the event times (Y0)
//' @param zy0    is a transpose of a subset of `mat1Z0` on these event times,
//'               e.g., .zy <- t(.mat1Z[.E0 == 1,])
//' @param e0     censoring indicator; 1 = event and 0 = control
//'
//'
//' Tunning parameters
//' @param spCriterion specifies the splitting criterion; 1 = DICON and 0 = ICON
//' @param numFold is the number of fold used in cross-validation
//' @param minNode1 is the minimum number of baseline observations in each terminal node; default = 15
//'                 we used time-invariant tree, so the node size criterion is on the baseline obs.
//' @param minSplit1 is the minimum number of baseline observations in each splitable node; default 30
//' @param maxNode is the maximum number of terminal nodes
//'
//' Returns a list with the following elements
//'   *treeMat - a xx by 5 matrix that consists of 5 columns;
//'     the number of rows depends on how depth the tree is
//'      Column1: Which covariate to split? Index corresponds to the columns in .X
//'      Column2: What value to split? (this is the ranked value; corresponding to which interval.
//'      Column3: Left child node (<= the cut value)
//'      Column4: Right child node (> the cut value)
//'      Column5: Node indicator; 1 = terminal node, 0 = non-terminal node
//'
//'   *nodeLabel - a n-dimensional vector, where n is the number of unique uncensored survival times.
//'
//'   *nodeSize - a xx by n matrix;
//'     the number of rows = number of terminal nodes
//'     the number of columns = number of unique ordered uncensored survival times  (Y0)
//'       This gives node size at each of the terminal nodes at each unique Y0
//' 
//'   *nodeMap - a xx-dimensional vector, where xx is the number of rows in treeMat.
//'     The elements indicates the element in 'nodeSize' to be used as risk set size in survival pred.
//'     Only the elements correspond to terminal nodes (treeMat[,5] == 1) will be used.
//' 
//' @noRd
// [[Rcpp::export]]
SEXP dyntree_C(const arma::umat& X0,
	       const arma::uvec& D0,
	       const arma::umat& r0,
	       int numFold,
	       int minNode1,
	       int minSplit1,
	       int maxNode) {
  TreeGrow tg(numFold, maxNode, minNode1, minSplit1);
  std::shared_ptr<Tree> tr2 = tg.trainCV(X0, r0, D0);
  const arma::uvec& vars0 = tr2->get_split_vars();
  arma::mat treeMat(vars0.n_elem, 6);
  arma::uvec tM1;
  arma::uvec tM2;
  arma::uvec tM3;
  arma::uvec tM4;  
  tM1 = tr2->get_split_values();
  tM2 = tr2->get_left_childs();
  tM3 = tr2->get_right_childs();
  tM4 = tr2->get_isLeaf();
  treeMat.col(0) = arma::conv_to<arma::vec>::from(vars0);
  treeMat.col(1) = arma::conv_to<arma::vec>::from(tM1);
  treeMat.col(2) = arma::conv_to<arma::vec>::from(tM2);
  treeMat.col(3) = arma::conv_to<arma::vec>::from(tM3);
  treeMat.col(4) = arma::conv_to<arma::vec>::from(tM4);
  treeMat.col(5) = tr2->get_lr_score();
  TreePrediction tp( X0, vars0, tM1, tM2, tM3, tM4 );
  return Rcpp::List::create(Rcpp::Named("treeMat") = treeMat,
                            Rcpp::Named("nodeLabel") = tp.get_nodeLabel(),
                            Rcpp::Named("nodeSize") = tp.get_nodeSize(),
                            Rcpp::Named("nodeMap") = tp.get_nodeMap());
}


//' Main function for survival prediction from a dyntree object
//'
//' @param zraw0 is a matrix consists of new covariates.
//'             The number of columns equal to the total number of unique survival times (n).
//'             The number of rows equal to the number of covariates
//' @param y0 is a n-dimensional vector consists of all survival times 
//' @param e0 is the censoring indicator
//' @param treeobj is the dyntree object
//' @param matX is the original covariate matrix,
//'             the number of rows is equal to the total observed survival times
//' @param disc is a vector with length equal to the number of covariates.
//'             This indicates whether a covariate is continuous (0) or not (1)
//' @param breaks is the cutoff points, e.g., cutoff <- (1:nc) / (nc+1)
//'             This gives the boundaries of intervals
//'
//' @noRd
// [[Rcpp::export]]
arma::vec predict_dyntree_C(const arma::mat& zraw0,
			    const arma::vec& y0,
			    const arma::uvec& e0,
			    const Rcpp::List& treeobj,
			    const arma::umat& matX) {
  arma::umat z0(zraw0.n_rows, arma::sum(e0));
  ForestPrediction::transformZ(zraw0, z0, matX, e0);
  // Rcpp::Rcout << "3" << std::endl;
  arma::umat nodeSize2 = treeobj[2];
  arma::uvec nodeLabel2 = treeobj[1];
  arma::uvec tnd32 = treeobj[3];
  arma::umat treeMat2 = treeobj[0];
  arma::vec sy = TreePrediction::getSurvival(z0, y0, e0, nodeSize2, nodeLabel2, tnd32, treeMat2);
  return sy;
}
