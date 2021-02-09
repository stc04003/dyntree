#ifndef TreeGrow_h
#define TreeGrow_h

#include <memory>
#include <armadillo>
#include "Tree.h"

class TreeGrow {
public:
  TreeGrow(uint nf, uint mn, uint min1, uint minsp1) :
  NUM_FOLD(nf), MAX_NODE(mn), MIN_NODE1(min1), MIN_SPLIT1(minsp1){};

  TreeGrow(uint mn, uint min1, uint minsp1) :
  MAX_NODE(mn), MIN_NODE1(min1), MIN_SPLIT1(minsp1){};

  double get_ICONTest(const arma::uvec& isLeafTemp,
                      const arma::umat& fmat, const arma::umat& Smat,
                      const arma::umat& fmat2, const arma::umat& Smat2) const;

  // This is the main function to grow a tree
  // return a tree
  std::shared_ptr<Tree> trainCV(const arma::umat& X0, 
                                const arma::umat& range0,
                                const arma::uvec& e) const;

  // GROW A LARGE TREE ON TRAINING DATA
  std::shared_ptr<Tree> grow(const arma::umat& X0,
                             const arma::umat& range0,
                             arma::umat& fmat,
                             arma::umat& Smat,
                             const arma::uvec& e) const;

  // GROW A TREE ON THE TRAINING SET AND UPDATE FMAT AND SMAT FOR THE VALIDATION SET
  std::shared_ptr<Tree> grow(const arma::umat& mat1Z,
                             const arma::umat& mat1ZVal,
                             const arma::umat& range0,
                             arma::umat& fmat,
                             arma::umat& Smat,
                             arma::umat& fmat2,
                             arma::umat& Smat2,
                             const arma::uvec& e) const;

  // PRUNING BASED ON CV
  arma::vec prune(arma::vec& beta,
                  const arma::umat& mat1Z,
                  const arma::umat& range0,
                  const arma::uvec& e) const;


  // USED IN THE GROW FUNCTION: FIND A SPLIT
  // USED WHEN GROW A LARGE TREE
  int split(const arma::umat& X0,
            arma::uvec& left_childs,
            arma::uvec& right_childs,
            arma::uvec& split_vars,
            arma::uvec& split_values,
            arma::uvec& isLeaf,
            arma::uvec& parents,
            arma::umat& fmat,
            arma::umat& Smat,
            // tree
            arma::ucube& ranges,
            arma::field<arma::uvec>& nodeSampleY,
            size_t& countsp,
            size_t& ndcount,
            const arma::uvec& e) const;

  // TRAINING + VALIDATION
  int split(const arma::umat& mat1Z,
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
            const arma::uvec& e) const;
 
  // logrank
  arma::ivec find_split_logrank(size_t nd,
				const arma::umat& mat1Z,
				const arma::uvec& isLeaf,
				arma::ucube& ranges,
				arma::field<arma::uvec>& nodeSampleY,
				arma::umat& fmat,
				arma::umat& Smat,
				size_t ndcount,
				const arma::uvec& e) const;

private:
  // control
  uint NUM_FOLD;
  uint MAX_NODE;
  uint MIN_NODE1;
  uint MIN_SPLIT1;
};

#endif /* TreeGrow_h */
