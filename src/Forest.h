#ifndef Forest_h
#define Forest_h

#include "Tree.h"
#include <memory>

class Forest {
public:
   Forest(uint nt,
          uint mn,
          uint min1,
          uint minsp1,
          uint mt) : NUM_TREE(nt), MAX_NODE(mn),
    MIN_NODE1(min1), MIN_SPLIT1(minsp1), mtry(mt){};

   void sampleWithoutReplacementSplit(arma::uword n, arma::uword n1,
                                      arma::umat& ids) {
      arma::uvec s0 = arma::regspace<arma::uvec>(0,n-1);
      for(size_t col_it = 0 ; col_it != NUM_TREE; ++col_it) {
         arma::uvec s = arma::shuffle(s0);
         ids.col(col_it) = s.subvec(0, (n1-1));
      }
   };

   void sampleWithReplacementSplit(arma::uword n, arma::uword n1, arma::umat& ids) {
      for(size_t col_it = 0 ; col_it != NUM_TREE; ++col_it) {
         ids.col(col_it) = arma::sort(arma::randi<arma::uvec>(n1, arma::distr_param(0,n1-1)));
      }
   };

   // GROW A FOREST
   int trainRF(std::vector<std::shared_ptr<Tree> >& trees,
               const arma::umat& X0,
               const arma::umat& r0,
               const arma::umat& ids,
               const arma::uvec& e);

   // GROW A TREE IN THE FOREST, WITHOUT PRUNING
   std::shared_ptr<Tree> train(const arma::umat& mat1Z,
                               const arma::umat& range0,
                               const arma::uvec& e) const;

   // We need to use fmat and Smat when calculating ICON
   int split_logrank(const arma::umat& mat1Z,
                  arma::uvec& left_childs,
                  arma::uvec& right_childs,
                  arma::uvec& split_vars,
                  arma::uvec& split_values,
                  arma::uvec& isLeaf,
                  arma::umat& fmat,
                  arma::umat& Smat,
                  // tree
                  arma::ucube& ranges,
                  arma::field<arma::uvec>& nodeSampleY,
        		  size_t& countsp,
                  size_t& ndcount,
                  const arma::uvec& e) const;

   arma::ivec find_split_logrank(size_t nd,
				 const arma::umat& mat1Z,
				 const arma::uvec& isLeaf,
                                 const arma::ucube& ranges,
                                 const arma::field<arma::uvec>& nodeSampleY,
                                 arma::umat& fmat,
                                 arma::umat& Smat,
                                 int ndcount,
                                 const arma::uvec& e) const;

private:
   uint spCriterion;
   uint NUM_TREE;
   uint MAX_NODE;
   uint MIN_NODE1;
   uint MIN_SPLIT1;
   uint mtry;
};

#endif /* Forest_h */
