#' dynTree: Landmark tree and ensemble
#'
#' The \code{dynTree} package grow/prune survival trees and ensemble.
#' 
#'
#' @aliases dynTree-package
#' @section Introduction:
#' The \code{dynTree} package provides implementations to a unified framework for
#' tree-structured analysis with censored survival outcomes.
#' 
#' @section Methods:
#' The package contains functions to construct survival trees and ensemble through
#' the main function \code{\link{dynTree}}.
#' 
#' @seealso \code{\link{dynTree}}
#' @docType package
#' 
#' @importFrom stats model.extract model.matrix model.response
#' @importFrom utils tail
#' @importFrom survival survfit Surv
#' @importFrom parallel detectCores makeCluster setDefaultCluster clusterExport stopCluster
#' @importFrom parallel parSapply parLapply
#' @importFrom graphics legend lines plot
#' @importFrom data.tree Node ToDataFrameTree ToDiagrammeRGraph SetGraphStyle SetNodeStyle
#' @importFrom DiagrammeR render_graph %>% export_graph
#' @importFrom Rcpp sourceCpp
#' 
#' @useDynLib dynTree 
"_PACKAGE"
NULL

