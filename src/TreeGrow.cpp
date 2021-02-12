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
  const arma::uvec& isl = tr->get_isLeaf();
  const arma::vec& lrs = tr->get_lr_score2();
  uint numLeaf = arma::sum(isl);
  if ((NUM_FOLD > 1) & (numLeaf > 1)) {
    arma::field<arma::uvec> nodeSetList(numLeaf);
    arma::vec lrAll(numLeaf);
    // Rcpp::Rcout << "lrs:" << lrs << std::endl;
    tr->giveNode(lrAll, lrs, nodeSetList, numLeaf);
    // Rcpp::Rcout << "lrAll-after-2:" << lrAll << std::endl;
    
    arma::uvec sizeTree = arma::regspace<arma::uvec>(1,lrAll.n_elem);
    arma::vec beta(lrAll.n_elem);

    // Rcpp::Rcout << "beta-before" << beta << std::endl;
    Tree::findBeta(lrAll, beta, sizeTree);

    // Rcpp::Rcout << "beta-after" << beta << std::endl;
    arma::vec lrBeta = prune(beta, X0, range0, e);
    
    // Rcpp::Rcout << "lrBeta" << lrBeta << std::endl;
    uint qo = lrBeta.index_max();
    Rcpp::Rcout << "qo" << qo << std::endl;
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
  arma::vec lr_score = arma::zeros<arma::vec>(MAX_NODE);
  arma::vec lr_score2 = arma::zeros<arma::vec>(MAX_NODE);
  ranges.row(0) = range0.t();
  arma::field<arma::uvec> nodeSampleY(MAX_NODE);
  nodeSampleY(0) = arma::regspace<arma::uvec>(0, n-1);
  size_t ndcount = 0;
  size_t countsp = 0;
  int end = 0;
  while(end == 0) {
    end = split(X0, left_childs, right_childs,
                split_vars, split_values, isLeaf,
                parents, fmat, Smat, lr_score, lr_score2, 
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
                                    parents(nonEmpty),
				    lr_score(nonEmpty),
				    lr_score2(nonEmpty)));
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
  arma::vec lr_score = arma::zeros<arma::vec>(MAX_NODE);
  arma::vec lr_score2 = arma::zeros<arma::vec>(MAX_NODE);
  ranges.row(0) = range0.t();
  arma::field<arma::uvec> nodeSampleY(MAX_NODE);
  nodeSampleY(0) = arma::regspace<arma::uvec>(0, n-1);
  int nVal = mat1ZVal.n_rows;
  arma::field<arma::uvec> nodeSampleYVal(MAX_NODE);
  nodeSampleYVal(0) = arma::regspace<arma::uvec>(0, nVal-1);
  size_t ndcount = 0;
  size_t countsp = 0;
  int end = 0;
  while(end == 0) {
    end = split(mat1Z, mat1ZVal,
                left_childs, right_childs,
                split_vars, split_values, isLeaf,
                parents, fmat, Smat, fmat2, Smat2, 
                lr_score, lr_score2,
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
                                    parents(nonEmpty),
				    lr_score(nonEmpty),
				    lr_score2(nonEmpty)));
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
  arma::mat lrVal(beta.n_elem, NUM_FOLD);
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
    const arma::uvec& il = trl-> get_isLeaf();  
    const arma::vec& lrsl = trl-> get_lr_score2();

    uint numLeaf = arma::sum(il);
    arma::field<arma::uvec> nodeSetList(numLeaf);
    arma::vec lrAll = lrsl(arma::find(il == 1));
    
    if (numLeaf > 1) 
      trl->giveNode(lrAll, lrsl, nodeSetList, numLeaf);

    // Rcpp::Rcout << "lrAll: " << lrAll << std::endl; //
    
    arma::vec sizeTree = arma::regspace<arma::vec>(1,lrAll.n_elem);
    arma::uvec isLeafTemp(il.n_elem);

    // Rcpp::Rcout << "lrAll: " << lrAll << std::endl; // 
    // Rcpp::Rcout << "beta: " << beta << std::endl; // 
    for(size_t j = 0; j < beta.n_elem; j++) {
      arma::vec lrbetajAll = lrAll -  beta(j) * arma::pow(sizeTree,1);
      uint opt = lrbetajAll.index_max();
      isLeafTemp.zeros();
      isLeafTemp( nodeSetList(opt) ).ones();
      // Rcpp::Rcout << "isLeafTemp: " << isLeafTemp << std::endl; //
      // Rcpp::Rcout << "lrAll: " << lrAll << std::endl; //
      Rcpp::Rcout << "lrbetajAll: " << lrbetajAll << std::endl; //
      lrVal(j, l) = Tree::get_LRTrain(isLeafTemp, lrsl);
      // TreeGrow::get_LRTest(isLeafTemp, fmat, Smat, fmat2, Smat2);
    }
  }
  return arma::sum(lrVal, 1)/NUM_FOLD;
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
		    arma::vec& lr_score,
		    arma::vec& lr_score2,
		    arma::ucube& ranges,
                    arma::field<arma::uvec>& nodeSampleY,
                    size_t& countsp,
                    size_t& ndcount,
                    const arma::uvec& e) const {
  int end = 0;
  int varsp = -1;
  int cutsp = 0;
  double LGmax;
  size_t nd = countsp;
  while(varsp == -1 && countsp <= ndcount) {
    nd = countsp;
    Rcpp::List bestSp = find_split_logrank(nd, X0, isLeaf, ranges, nodeSampleY,
					   fmat, Smat, ndcount, e);
    varsp = bestSp(1);
    cutsp = bestSp(2);
    LGmax = bestSp(3);
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
    lr_score(nd) = LGmax;
    arma::uword ndc1 = ndcount + 1;
    arma::uword ndc2 = ndcount + 2;
    lr_score2(nd) = LGmax;
    lr_score2(ndc1) = LGmax;
    lr_score2(ndc2) = LGmax;
    left_childs(nd) = ndc1;
    right_childs(nd) = ndc2;
    parents(ndc1) = nd;
    parents(ndc2) = nd;
    arma::uvec nodeSampleYnd = std::move(nodeSampleY(nd));
    arma::uvec zvarspsub = X0( varsp*X0.n_rows + nodeSampleYnd );
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
  // Rcpp::Rcout << "lr_score: " << lr_score << std::endl;
  // Rcpp::Rcout << "lr_score2: " << lr_score2 << std::endl;
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
		    arma::vec& lr_score,
                    arma::vec& lr_score2,
                    arma::ucube& ranges,
                    arma::field<arma::uvec>& nodeSampleY,
                    arma::field<arma::uvec>& nodeSampleYVal,
                    size_t& countsp,
                    size_t& ndcount,
                    const arma::uvec& e) const {
  int end = 0;
  int varsp = -1;
  int cutsp = 0;
  double LGmax;
  size_t nd = countsp;
  while(varsp == -1 && countsp <= ndcount) {
    nd = countsp;
    // arma::ivec bestSp(3);

    // Rcpp::Rcout << "isLeaf: " << isLeaf << std::endl;
    // Rcpp::Rcout << "nodeSampleY: " << nodeSampleY << std::endl;
      
    Rcpp::List bestSp = find_split_logrank(nd, mat1Z, isLeaf, ranges, nodeSampleY, 
				fmat, Smat, ndcount, e);
    varsp = bestSp(1);
    cutsp = bestSp(2);
    LGmax = bestSp(3);

    // Rcpp::Rcout << "varsp: " << varsp << "cutsp: " << cutsp << std::endl;
    
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
    lr_score(nd) = LGmax;
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

    lr_score2(nd) = LGmax * nodeSampleYVal(nd).n_elem;
    lr_score2(ndc1) = LGmax * nodeSampleYVal(ndc1).n_elem;
    lr_score2(ndc2) = LGmax * nodeSampleYVal(ndc2).n_elem;

    
    fmat2.col(ndc1) = arma::cumsum(e(ndc1));
    fmat2.col(ndc2) = arma::cumsum(e(ndc2));
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

// Rcpp::Rcout << cutsp << std::endl;  
// arma::ivec
Rcpp::List TreeGrow::find_split_logrank(size_t nd,
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
  for(int p = 0; p < P; p++) {
    // arma::uvec indY = nodeSampleY(nd)( sort_index( mat1Z(p*n + nodeSampleY(nd)) ));
    arma::uvec indY = nodeSampleY(nd);
    int nj = indY.size();
    arma::uvec fLSum = arma::zeros<arma::uvec>(nj);
    arma::uvec fRSum = arma::zeros<arma::uvec>(nj);
    // arma::uvec SLSum = arma::zeros<arma::uvec>(nj);
    // arma::uvec SRSum = arma::zeros<arma::uvec>(nj);
    fRSum = e(indY);
    arma::vec SRSum = arma::regspace<arma::vec>(nj, 1);
    arma::vec SLSum = arma::zeros<arma::vec>(nj);
    arma::uvec rangeCut = arma::regspace<arma::uvec>(ranges(nd, p, 0), ranges(nd, p, 1));
    // Rcpp::Rcout << "nj: " << nj << std::endl;
    // Rcpp::Rcout << "indY: " << indY << std::endl;
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
	// Rcpp::Rcout << "j: " << j << std::endl;
      } // end while
      // arma::uvec n1 = find(fLSum > 0);
      // arma::uvec n2 = find(fRSum > 0);
      // Rcpp::Rcout << "sum(fLSum): " << sum(fLSum) << "fLSum: " << fLSum << std::endl;
      // Rcpp::Rcout << "fLSum: " << fLSum << "fRSum" << fRSum << std::endl;
      // Rcpp::Rcout << "SLSum: " << SLSum << "SRSum" << SRSum << std::endl;
      // if((SLSum(0) < MIN_NODE1 || SRSum(0) < MIN_NODE1)) {
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
	// Rcpp::Rcout << "w1: " << w1 << std::endl;
	// Rcpp::Rcout << "w2: " << w2 << std::endl;
        LGTemp = w1 * w1 / w2;
	// Rcpp::Rcout << "P: " << p << " LGTemp: " << LGTemp << std::endl;
      }
      if (LGTemp > LGmax) {
        LGmax = LGTemp;
        varsp = p;
        cutsp = cu;
	// Rcpp::Rcout << "varsp" << varsp << std::endl;
	// fmat.col(ndcount + 1) = fLSum;
	// fmat.col(ndcount + 2) = fRSum;
	// Smat.col(ndcount + 1) = SLSum;
	// Smat.col(ndcount + 2) = SRSum;
      }
    }
  }
  // Rcpp::Rcout << "fmat: " << fmat << std::endl;
  // arma::ivec vecsp(4);
  // arma::ivec vecsp(3);
  if(varsp == -1) {
    return Rcpp::List::create(0, -1, 0, 0);
    // vecsp(0) = 0;
    // vecsp(1) = -1;
    // vecsp(2) = 0;
  } else {
    return Rcpp::List::create(1, varsp, cutsp, LGmax);
    // vecsp(0) = 1;
    // vecsp(1) = varsp;
    // vecsp(2) = cutsp;
  }
  // return vecsp;
}
