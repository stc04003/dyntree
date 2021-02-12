#include "Tree.h"
#include "globals.h"

const arma::uvec& Tree::get_split_vars() const
{
  return split_vars;
}

const arma::uvec& Tree::get_split_values() const
{
  return split_values;
}

const arma::uvec& Tree::get_left_childs() const
{
  return left_childs;
}

const arma::uvec& Tree::get_right_childs() const
{
  return right_childs;
}

const arma::uvec& Tree::get_isLeaf() const
{
  return isLeaf;
}

const arma::uvec& Tree::get_parents() const
{
  return parents;
}

const arma::vec& Tree::get_lr_score() const
{
  return lr_score;
}

const arma::vec& Tree::get_lr_score2() const
{
  return lr_score2;
}

void Tree::setzero(uint i, uint ndcount) {
  uint lid = left_childs(i);
  uint rid = right_childs(i);
  if(lid <= ndcount && isLeaf(lid) == 0) {
    setzero(lid,ndcount);
  }
  if(rid <= ndcount && isLeaf(rid) == 0) {
    setzero(rid,ndcount);
  }
  left_childs(i) = 0;
  right_childs(i) = 0;
  split_values(i) = 0;
  split_vars(i) = 0;
}

void Tree::cut(arma::uvec& nodeTerminal)
{
  arma::uvec isLeaf2 = isLeaf;
  isLeaf2.zeros();
  isLeaf2( nodeTerminal ).ones();
  int ndcount = isLeaf.n_elem-1;
  int ndcount2 = nodeTerminal(nodeTerminal.n_elem-1);
  for(int i = 0; i <= ndcount2; i++) {
    if(isLeaf2(i) == 1 && isLeaf(i) == 0) {
      setzero(i, ndcount);
    }
  }
  arma::uvec nonEmpty = arma::regspace<arma::uvec>(0, ndcount2);
  right_childs = right_childs.elem(nonEmpty);
  left_childs = left_childs.elem(nonEmpty);
  split_values = split_values.elem(nonEmpty);
  split_vars = split_vars(nonEmpty);
  isLeaf = isLeaf2(nonEmpty);
}

double Tree::get_LRTrain(const arma::uvec& isLeafTemp,
                           const arma::vec& lr)
{
  arma::vec lrTemp = lr( arma::find(isLeafTemp == 1) );
  int numLeafTemp = arma::sum(isLeafTemp == 1);
  double lrSum = 0;
  for(int i = 0; i < numLeafTemp; i++) {
    lrSum = lrSum + lrTemp(i);
  }
  return lrSum;
}


// calling "ICON" but it is really calculating based on log-rank
double Tree::get_ICONTrain(const arma::uvec& isLeafTemp,
                           const arma::umat& fmat,
                           const arma::umat& Smat)
{
  arma::umat fmatTemp = fmat.cols( arma::find(isLeafTemp == 1) );
  arma::umat SmatTemp = Smat.cols( arma::find(isLeafTemp == 1) );
  int numLeafTemp = arma::sum(isLeafTemp == 1);
  uint K = fmat.n_rows;
  //SmatTemp.print("a");
  arma::vec icon = arma::zeros<arma::vec>(K);
  for(size_t k = 0; k != K; k++) {
    double w1 = 0;
    double w2 = 0;
    for(int i = 0; i < numLeafTemp; i++) {
      for(int j = 0; j < i; j++) {
        double fi = fmatTemp(k,i);
        double fj = fmatTemp(k,j);
        int Sj = SmatTemp(k,j);
        int Si = SmatTemp(k,i);
	w1 += fi - (fi + fj) * Si / (Si + Sj);
	w2 += Si * Sj * (fi + fj) * (Si + Sj - fi - fj) / (Si + Sj) / (Si + Sj) / (Si + Sj - 1);
      }
      // int Si = SmatTemp(k,i);
      // double maxSi = SmatTemp.row(k).max();
      // icon(k) += -1 * log((double)Si / maxSi);
      // Rcpp::Rcout << icon(k) << std::endl;
    }
    if (w2 > 0) icon(k) += w1 * w1 / w2;
  }
  // icon = icon / (arma::sum(fmatTemp, 1) % arma::sum(SmatTemp, 1));
  return arma::sum(icon.elem(find_finite(icon))) / K;
}

void Tree::findOptimalSizekSubtree(arma::umat& fmat, arma::umat& Smat,
                                   arma::vec& iconAll, arma::field<arma::uvec>& nodeSetList,
				   uint numLeaf)
{
  arma::uvec nodeID = arma::regspace<arma::uvec>(0, isLeaf.n_elem-1);
  arma::uvec isLeafTemp = arma::zeros<arma::uvec>(isLeaf.n_elem);
  isLeafTemp(0) = 1;
  iconAll(0) = get_ICONTrain(isLeafTemp, fmat, Smat);
  nodeSetList(0) = nodeID(arma::find(isLeafTemp == 1) );
  arma::uvec isLeafTemp2 = isLeafTemp;
  arma::uvec isLeafTemp3 = isLeafTemp;
  size_t i = 1;
  while(i < numLeaf - 1) {
    double iconMax = 0;
    double iconl = 0;
    isLeafTemp2 = isLeafTemp;
    isLeafTemp3 = isLeafTemp;
    arma::uvec nodeTermTemp = nodeID(arma::find(isLeafTemp == 1) );
    for(size_t l = 0; l < nodeTermTemp.n_elem; l++ ) {
      int spNd = nodeTermTemp(l);
      if(isLeaf(spNd) == 0) {
        isLeafTemp2 = isLeafTemp3;
        int lid = left_childs(spNd);
        int rid = right_childs(spNd);
        isLeafTemp2(spNd) = 0;
        isLeafTemp2(lid) = 1;
        isLeafTemp2(rid) = 1;
        iconl = get_ICONTrain(isLeafTemp2, fmat, Smat);
	// Rcpp::Rcout << iconl << std::endl;
        if(iconl > iconMax) {
          iconMax = iconl;
          nodeSetList(i) = nodeID(arma::find(isLeafTemp2 == 1) );
          isLeafTemp = isLeafTemp2;
        }
      }
    }
    iconAll(i) = iconMax;
    i++;
  }
  iconAll(i) = get_ICONTrain(isLeaf, fmat, Smat);
  nodeSetList(i) = nodeID(arma::find(isLeaf == 1) );
}


void Tree::giveNode(arma::vec& lrAll, arma::vec lrs,
		    arma::field<arma::uvec>& nodeSetList, uint numLeaf)
{
  arma::uvec nodeID = arma::regspace<arma::uvec>(0, isLeaf.n_elem-1);
  arma::uvec isLeafTemp = arma::zeros<arma::uvec>(isLeaf.n_elem);
  isLeafTemp(0) = 1;

  // Rcpp::Rcout << "lrAll-before" << lrAll << std::endl;
    
  lrAll(0) = get_LRTrain(isLeafTemp, lrs);
  nodeSetList(0) = nodeID(arma::find(isLeafTemp == 1) );
  arma::uvec isLeafTemp2 = isLeafTemp;
  arma::uvec isLeafTemp3 = isLeafTemp;
  size_t i = 1;
  while(i < numLeaf - 1) {
    isLeafTemp2 = isLeafTemp;
    isLeafTemp3 = isLeafTemp;
    arma::uvec nodeTermTemp = nodeID(arma::find(isLeafTemp == 1) );
    for(size_t l = 0; l < nodeTermTemp.n_elem; l++ ) {
      int spNd = nodeTermTemp(l);
      if(isLeaf(spNd) == 0) {
        isLeafTemp2 = isLeafTemp3;
        int lid = left_childs(spNd);
        int rid = right_childs(spNd);
        isLeafTemp2(spNd) = 0;
        isLeafTemp2(lid) = 1;
        isLeafTemp2(rid) = 1;
	lrAll(i) = get_LRTrain(isLeafTemp2, lrs);
	nodeSetList(i) = nodeID(arma::find(isLeafTemp2 == 1) );
	isLeafTemp = isLeafTemp2;
      }
    }
    i++;
  }
  // Rcpp::Rcout << "lrs: " << lrs << std::endl;
  // Rcpp::Rcout << "lrAll-after: " << lrAll << std::endl;
  lrAll(i) = get_LRTrain(isLeaf, lrs);
  nodeSetList(i) = nodeID(arma::find(isLeaf == 1) );
}


void Tree::findBeta(arma::vec& lrAll, arma::vec& beta, arma::uvec& sizeTree)
{
  arma::vec alpha(lrAll.n_elem);
  int L = lrAll.n_elem;
  size_t q = 1;
  alpha(0) = 0;
  sizeTree(0) = L;
  while( L > 1 ) {
    arma::vec lrSmallerTree = lrAll.head( L-1 );
    arma::vec LL = arma::regspace(L - 1, 1);
    arma::vec alphaTT = (lrAll(L-1) - lrSmallerTree) / LL;
    // Rcpp::Rcout << "alphaTT" << std::endl;
    // Rcpp::Rcout << alphaTT << std::endl; 
    alpha(q) = alphaTT.min();
    sizeTree(q) = alphaTT.index_min() + 1;
    L = sizeTree(q);
    q++;
  }
  if(q < lrAll.n_elem) {
    sizeTree.shed_rows(q, lrAll.n_elem-1);
    alpha.shed_rows(q, lrAll.n_elem-1);
    beta.shed_rows(q, lrAll.n_elem-1);
  }
  for(size_t i = 0; i < alpha.n_elem; i++) {
    if(i < alpha.n_elem - 1) {
      beta(i) = sqrt( alpha(i)*alpha(i+1)  );
    } else {
      beta(i) = alpha(i);
    }
  }
}
