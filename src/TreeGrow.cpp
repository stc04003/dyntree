#include <memory>
#include <RcppArmadillo.h>

#include "TreeGrow.h"
#include "globals.h"


double TreeGrow::get_ICONTest(const arma::uvec& isLeafTemp,
                              const arma::umat& fmat, const arma::umat& Smat,
                              const arma::umat& fmat2, const arma::umat& Smat2) const
{
  arma::uvec leafTemp = arma::find(isLeafTemp==1);
  int numLeafTemp = leafTemp.n_elem;
  arma::vec icon = arma::zeros<arma::vec>(1);
  arma::vec fSum = arma::zeros<arma::vec>(1);
  arma::vec SSum = arma::zeros<arma::vec>(1);
  double w1 = 0;
  double w2 = 0;
  for(int i = 0; i < numLeafTemp; i++) {
    int li = leafTemp(i);
    double fi2 = fmat2(0,li);
    double Si2 = Smat2(0,li);
    for(int j = 0; j < i; j++) {
      int lj = leafTemp(j);
      // w1 += fmat2(0, lj) - (fi + fmat(0, lj)) * Si / (Si + Smat(0, lj));
      // w2 += Si * Smat(0, lj) * (fi + fmat(0, lj)) * (Si + Smat(0, lj) - fi - fmat(0, lj)) /
      //   (Si + Smat(0, lj)) * (Si + Smat(0, lj)) * (Si + Smat(0, lj) - 1);
      w1 += fmat(0, lj) - (fi2 + fmat2(0, lj)) * Si2 / (Si2 + Smat2(0, lj));
      w2 += Si2 * Smat2(0, lj) * (fi2 + fmat2(0, lj)) * (Si2 + Smat2(0, lj) - fi2 - fmat2(0, lj)) /
	(Si2 + Smat2(0, lj)) * (Si2 + Smat2(0, lj)) * (Si2 + Smat2(0, lj) - 1);
    }      
    // Rcpp::Rcout << log((double)Si / maxSi) << std::endl;
    // if (Si > 0) icon(0) += log((double)Si / maxSi);
    // if (Si2 > 0) icon(0) += -log((double)Si2 / maxSi2);
  }
  if (w2 > 0) icon(0) += w1 * w1 / w2;
  // Rcpp::Rcout << "K is" << K << std::endl;
  return arma::sum(icon);
}

std::shared_ptr<Tree> TreeGrow::trainCV(const arma::umat& X0,
                                        const arma::umat& range0,
                                        const arma::uvec& e) const
{
  arma::umat fmat = arma::zeros<arma::umat>(1, MAX_NODE);
  arma::umat Smat = arma::zeros<arma::umat>(1, MAX_NODE);
  std::shared_ptr<Tree> tr = grow(X0, range0, fmat, Smat, e);
  
  // Rcpp::Rcout << "3" << std::endl;
  const arma::uvec& isl = tr->get_isLeaf();
  uint numLeaf = arma::sum(isl);
  if (NUM_FOLD > 1 & numLeaf > 1) {
    arma::field<arma::uvec> nodeSetList(numLeaf);
    arma::vec iconAll(numLeaf);
    tr->findOptimalSizekSubtree(fmat, Smat, iconAll, nodeSetList, numLeaf);
    // Initialization
    arma::uvec sizeTree = arma::regspace<arma::uvec>(1,iconAll.n_elem);
    arma::vec beta(iconAll.n_elem);
    Tree::findBeta(iconAll, beta, sizeTree); 
    arma::vec iconBeta = prune(beta, X0, range0, e);
    uint qo = iconBeta.index_max();
    // Rcpp::Rcout << "iconBeta" << iconBeta << std::endl;
    // Rcpp::Rcout << "qo" << qo << std::endl;
    arma::uvec nodeSetFinal = nodeSetList(sizeTree(qo)-1);
    tr->cut(nodeSetFinal);
  }
  return tr;
}

std::shared_ptr<Tree> TreeGrow::grow(const arma::umat& X0,
                                     const arma::umat& range0,
                                     arma::umat& fmat,
                                     arma::umat& Smat,
                                     const arma::uvec& e) const
{
  int n = X0.n_rows;
  int P = X0.n_cols;
  arma::ucube ranges = arma::zeros<arma::ucube>(MAX_NODE, P, 2);
  arma::uvec left_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec right_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec split_vars = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec split_values = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec isLeaf = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec parents = arma::zeros<arma::uvec>(MAX_NODE);
  ranges.row(0) = range0.t();
  arma::field<arma::uvec> nodeSampleY(MAX_NODE);
  nodeSampleY(0) = arma::regspace<arma::uvec>(0, n-1);
  size_t ndcount = 0;
  size_t countsp = 0;
  int end = 0;
  while(end == 0) {
    end = split(X0, left_childs, right_childs,
                split_vars, split_values, isLeaf,
                parents, fmat, Smat, 
                ranges, nodeSampleY, 
                countsp, ndcount, e);
    if(ndcount >= MAX_NODE - 2) {
      isLeaf( arma::find(left_childs == 0) ).ones();
      break;
    }
  }
  arma::uvec nonEmpty = arma::regspace<arma::uvec>(0, ndcount);
  std::shared_ptr<Tree> tr(new Tree(left_childs(nonEmpty),
                                    right_childs(nonEmpty),
                                    split_vars(nonEmpty),
                                    split_values(nonEmpty),
                                    isLeaf(nonEmpty),
                                    parents(nonEmpty)  ));
  if( ndcount + 1 <=  MAX_NODE-1 ) {
    arma::uvec Empty = arma::regspace<arma::uvec>(ndcount+1, MAX_NODE-1);
    fmat.shed_cols(Empty);
    Smat.shed_cols(Empty);
  }
  if(ndcount > 1) {
    fmat.col(0) = fmat.col(1) + fmat.col(2);
    Smat.col(0) = Smat.col(1) + Smat.col(2);
  } else {
    // no need to prune if there is only one node
  }
  return tr;
}

std::shared_ptr<Tree> TreeGrow::grow(const arma::umat& mat1Z,
				     const arma::umat& mat1ZVal,
				     const arma::umat& range0,
                                     arma::umat& fmat,
                                     arma::umat& Smat,
                                     arma::umat& fmat2,
                                     arma::umat& Smat2,
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
  arma::uvec parents = arma::zeros<arma::uvec>(MAX_NODE);
  ranges.row(0) = range0.t();
  arma::field<arma::uvec> nodeSampleY(MAX_NODE);
  int nVal = mat1ZVal.n_rows;
  arma::field<arma::uvec> nodeSampleYVal(MAX_NODE);
  nodeSampleYVal(0) = arma::regspace<arma::uvec>(0, nVal-1);
  size_t ndcount = 0;
  size_t countsp = 0;
  int end = 0;
  while(end == 0) {
    end = split(mat1Z,mat1ZVal,
                left_childs, right_childs,
                split_vars, split_values, isLeaf,
                parents, fmat, Smat, fmat2, Smat2, 
                ranges, nodeSampleY, nodeSampleYVal,
                countsp, ndcount, e);
    if(ndcount >= MAX_NODE - 2) {
      isLeaf( arma::find(left_childs == 0) ).ones();
      break;
    }
  }
  arma::uvec nonEmpty = arma::regspace<arma::uvec>(0, ndcount);
  std::shared_ptr<Tree> tr(new Tree(left_childs(nonEmpty),
                                    right_childs(nonEmpty),
                                    split_vars(nonEmpty),
                                    split_values(nonEmpty),
                                    isLeaf(nonEmpty),
                                    parents(nonEmpty)  ));
  if( ndcount + 1 <=  MAX_NODE-1 ) {
    arma::uvec Empty = arma::regspace<arma::uvec>(ndcount+1, MAX_NODE-1);
    fmat.shed_cols(Empty);
    Smat.shed_cols(Empty);
    fmat2.shed_cols(Empty);
    Smat2.shed_cols(Empty);
  }
  if(ndcount > 1)
  {
    fmat.col(0) = fmat.col(1) + fmat.col(2);
    Smat.col(0) = Smat.col(1) + Smat.col(2);
    fmat2.col(0) = fmat2.col(1) + fmat2.col(2);
    Smat2.col(0) = Smat2.col(1) + Smat2.col(2);
  }else{
    // no need to prune if there is only one node
  }
  return tr;
}

arma::vec TreeGrow::prune(arma::vec& beta,
                          const arma::umat& mat1Z,
                          const arma::umat& range0,
                          const arma::uvec& e) const
{
  // create folds
  int n = mat1Z.n_rows;
  arma::uvec s = arma::shuffle(arma::regspace<arma::uvec>(0,n-1));
  int quotient = n / NUM_FOLD;
  int remainder = n % NUM_FOLD;
  arma::uvec foldstart = arma::regspace<arma::uvec>(0, quotient, n-1);
  arma::uvec foldend = arma::regspace<arma::uvec>(quotient-1, quotient, n-1);
  arma::uvec::iterator it = foldstart.end();
  arma::uvec::iterator it2 = foldend.end();
  it--;
  it2--;
  while(remainder > 0 ) {
    (*it) = *it + remainder;
    (*it2) = *it2 + remainder;
    it--;
    it2--;
    remainder--;
  }
  // create folds end
  arma::mat iconVal(beta.n_elem, NUM_FOLD);
  for(size_t l = 0; l != NUM_FOLD; l++) {
    // get training/validation set
    arma::uvec valid = s(arma::regspace<arma::uvec>(foldstart(l), foldend(l)));
    arma::uvec trainid;
    if(l == 0) {
      trainid = s(arma::regspace<arma::uvec>(foldend(l) + 1, n - 1));
    }
    else if(l == NUM_FOLD-1) {
      trainid = s(arma::regspace<arma::uvec>(0, foldstart(l) - 1));
    } else {
      arma::uvec trainid1 = arma::regspace<arma::uvec>(0, foldstart(l) - 1);
      arma::uvec trainid2 = arma::regspace<arma::uvec>(foldend(l) + 1, n - 1);
      trainid = s(join_cols(trainid1, trainid2));
    }
    arma::umat fmat = arma::zeros<arma::umat>(1, MAX_NODE);
    arma::umat Smat = arma::zeros<arma::umat>(1, MAX_NODE);
    arma::umat fmat2 = arma::zeros<arma::umat>(1, MAX_NODE);
    arma::umat Smat2 = arma::zeros<arma::umat>(1, MAX_NODE);
    TreeGrow tg(MAX_NODE, MIN_NODE1, MIN_SPLIT1);
    std::shared_ptr<Tree> trl = tg.grow(mat1Z.rows( trainid ),
                                        mat1Z.rows( valid ),
                                        range0, fmat, Smat, fmat2, Smat2, e(trainid));
    const arma::uvec& il = trl->get_isLeaf();
    uint numLeaf = arma::sum(il);
    arma::field<arma::uvec> nodeSetList(numLeaf);
    arma::vec iconAll(numLeaf);
    if (numLeaf > 1) 
      trl->findOptimalSizekSubtree(fmat, Smat, iconAll, nodeSetList, numLeaf);
    arma::vec sizeTree = arma::regspace<arma::vec>(1,iconAll.n_elem);
    arma::uvec isLeafTemp(il.n_elem);
    for(size_t j = 0; j < beta.n_elem; j++) {
      arma::vec iconbetajAll = iconAll -  beta(j) * arma::pow(sizeTree,1);
      uint opt = iconbetajAll.index_max();
      isLeafTemp.zeros();
      isLeafTemp( nodeSetList(opt) ).ones();
      iconVal(j, l) = TreeGrow::get_ICONTest(isLeafTemp, fmat, Smat, fmat2, Smat2);
    }
  }
  return arma::sum(iconVal, 1)/NUM_FOLD;
}



int TreeGrow::split(const arma::umat& X0,
                    arma::uvec& left_childs,
                    arma::uvec& right_childs,
                    arma::uvec& split_vars,
                    arma::uvec& split_values,
                    arma::uvec& isLeaf,
                    arma::uvec& parents,
                    arma::umat& fmat,
                    arma::umat& Smat,// tree
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
    bestSp = find_split_logrank(nd, X0, isLeaf, 
                                ranges, nodeSampleY, fmat, Smat, ndcount, e);
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
  if(varsp != -1) {
    split_vars(nd) = varsp;
    split_values(nd) = cutsp;
    arma::uword ndc1 = ndcount + 1;
    arma::uword ndc2 = ndcount + 2;
    left_childs(nd) = ndc1;
    right_childs(nd) = ndc2;
    parents(ndc1) = nd;
    parents(ndc2) = nd;
    arma::uvec nodeSampleYnd = std::move(nodeSampleY(nd));
    arma::uvec zvarspsub = mat1Z( varsp*mat1Z.n_rows + nodeSampleYnd );
    nodeSampleY(ndc1) = nodeSampleYnd( arma::find(zvarspsub <=cutsp) );
    nodeSampleY(ndc2) = nodeSampleYnd( arma::find(zvarspsub >cutsp) );
    ranges.row(ndc1) = ranges.row(nd);
    ranges.row(ndc2) = ranges.row(nd);
    ranges(ndc2,varsp,0) = cutsp+1;
    ranges(ndc1,varsp,1) = cutsp;
    ndcount += 2;
    while(countsp <= ndcount) {
      countsp++;
      if(isLeaf(countsp) == 0) break;
    }
  } else {
    end = 1;
  }
  return end;
}

int TreeGrow::split(const arma::umat& mat1Z,
		    const arma::umat& mat1ZVal,
		    arma::uvec& left_childs,
                    arma::uvec& right_childs,
                    arma::uvec& split_vars,
                    arma::uvec& split_values,
                    arma::uvec& isLeaf,
                    arma::uvec& parents,
                    arma::umat& fmat,
                    arma::umat& Smat,
                    arma::umat& fmat2,
                    arma::umat& Smat2,// tree
                    arma::ucube& ranges,
                    arma::field<arma::uvec>& nodeSampleY,
                    arma::field<arma::uvec>& nodeSampleYVal,
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
    bestSp = find_split_logrank(nd, mat1Z, isLeaf, ranges, nodeSampleY, 
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
    parents(ndc1) = nd;
    parents(ndc2) = nd;
    arma::uvec nodeSampleYnd = std::move(nodeSampleY(nd));
    arma::uvec zvarspsub = mat1Z( varsp*n + nodeSampleYnd );
    nodeSampleY(ndc1) = nodeSampleYnd( arma::find(zvarspsub <= cutsp) );
    nodeSampleY(ndc2) = nodeSampleYnd( arma::find(zvarspsub > cutsp) );
    arma::uvec nodeSampleYndVal = std::move(nodeSampleYVal(nd));
    arma::uvec zvarspsubVal = mat1ZVal( varsp*mat1ZVal.n_rows + nodeSampleYndVal );
    nodeSampleYVal(ndc1) = nodeSampleYndVal( arma::find(zvarspsubVal <=cutsp) );
    nodeSampleYVal(ndc2) = nodeSampleYndVal( arma::find(zvarspsubVal >cutsp) );
    fmat2.col(ndc1) = arma::sum( mat1fVal.cols( nodeSampleYVal(ndc1) ), 1);
    fmat2.col(ndc2) = arma::sum( mat1fVal.cols( nodeSampleYVal(ndc2) ), 1);    
    ranges.row(ndc1) = ranges.row(nd);
    ranges.row(ndc2) = ranges.row(nd);
    ranges(ndc2,varsp,0) = cutsp+1;
    ranges(ndc1,varsp,1) = cutsp;
    ndcount += 2;
    while(countsp <= ndcount) {
      countsp++;
      if(isLeaf(countsp) == 0)
      {break;}
    }
  } else {
    end = 1;
  }
  return end;
}

arma::ivec TreeGrow::find_split_logrank(size_t nd,
					const arma::umat& mat1Z,
					const arma::uvec& isLeaf,
					arma::ucube& ranges,
					arma::field<arma::uvec>& nodeSampleY,
					arma::umat& fmat,
					arma::umat& Smat,
					size_t ndcount,
					const arma::uvec& e) const {
  int P = mat1Z.n_cols;
  int n = mat1Z.n_rows;
  int varsp = -1;
  int cutsp = 0;
  double LGmax = 0;
  double LGTemp = 0;
  arma::mat fmatTerm = fmat.cols(arma::find(isLeaf == 1));
  arma::umat SmatTerm = Smat.cols(arma::find(isLeaf == 1));
  // Rcpp::Rcout << "isLeaf:" << isLeaf << std::endl;
  for(int p = 0; p < P; p++) {
    arma::uvec indY = nodeSampleY(nd)( sort_index( mat1Z(p*n + nodeSampleY(nd)) ));
    arma::uvec fLSum = arma::zeros<arma::uvec>(n);
    arma::uvec SLSum = arma::zeros<arma::uvec>(n);
    arma::uvec fRSum = cumsum(e);
    arma::uvec SRSum = arma::regspace<arma::uvec>(n,1);
    int j = 0;
    arma::uvec jv = arma::zeros<arma::uvec>(1);
    int nj = indY.size();
    arma::uvec rangeCut = arma::regspace<arma::uvec>(ranges(nd, p, 0), ranges(nd, p, 1));
    for(auto cu : rangeCut) {
      while(j < nj) {
        int indYj = indY(j);
        size_t z = mat1Z(indYj, p);
        if(z == cu) {
	  fLSum(indYj) = fLSum(indYj) + e(indYj);
	  fRSum(indYj) = fRSum(indYj) - e(indYj);
	  SLSum(indYj) = SLSum(indYj) + 1;
	  SRSum(indYj) = SRSum(indYj) - 1;
	  j++;
        } else {
          break;
        }
      }
      if((SLSum(0) < MIN_NODE1 || SRSum(0) < MIN_NODE1)) {
        LGTemp = 0;
      } else {
	double w1 = 0;
	double w2 = 0;
	arma::uvec vec1 = arma::zeros<arma::uvec>(n);
	arma::uvec vec2 = arma::zeros<arma::uvec>(n);
	vec1 = fLSum - (fLSum + fRSum) % SLSum / ( SLSum + SRSum );
	vec2 = (SLSum + SRSum - 1) % SRSum % SLSum % (fLSum + fRSum) %
	  (SLSum + SRSum - fLSum - fRSum) /
	  (SLSum + SRSum) / (SLSum + SRSum);
	w1 = arma::sum(vec1.elem(find_finite(vec1)));
	w2 = arma::sum(vec2.elem(find_finite(vec2)));
	LGTemp = w1 * w1 / w2;
      }
      // if (LGTemp < 0) Rcpp::Rcout << "LGTemp:" << LGTemp << std::endl;    	
      if (LGTemp > LGmax) {
        LGmax = LGTemp;
        varsp = p;
        cutsp = cu;
        fmat.col(ndcount + 1) = fLSum;
        fmat.col(ndcount + 2) = fRSum;
        Smat.col(ndcount + 1) = SLSum;
        Smat.col(ndcount + 2) = SRSum;
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
