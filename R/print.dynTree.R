#' Printing an \code{dynTree} object
#'
#' The function prints an \code{dynTree} object. It is a method for the generic function print of class "\code{dynTree}".
#'
#' @param x an \code{dynTree} object.
#' @param digits the number of digits of numbers to print.
#' @param tree an optional integer specifying the \eqn{n^{th}} tree in the forest to print.
#' The function prints the contents of an \code{lforest} object by default,
#' if a tree is not specified.
#' @param ... for future development.
#'
#' @importFrom data.tree Node ToDataFrameTree 
#' @export
#' @example inst/examples/ex_dynTree.R
print.dynTree <- function(x, digits = 5, tree = NULL, ...) {
    if (!is.dynTree(x)) stop("Response must be a \"dynTree\" object.")
    ## digits = getOption("digits")
    if (!x$ensemble) {
        printTree(x$Frame, x$vNames, digits)
    } else {
        if (!is.null(tree)) {
            if (!is.wholenumber(tree)) stop("Tree number must be an integer.")
            if (tree > length(x$trees)) stop("Tree number exceeded the number of trees in forest.")
            printTree(x$Frame[[tree]], vNames = x$vNames, digits = digits)
        } else {
            cat("Ensembles\n\n")
            cat("Call:\n", deparse(x$call), "\n\n")
            cat("Sample size:                                       ", ncol(x$xlist[[1]]), "\n")
            cat("Number of independent variables:                   ", length(unique(x$data$.id2)),"\n")
            cat("Number of trees:                                   ", x$control$numTree, "\n")
            cat("Number of variables tried at each split:           ", x$control$mtry, "\n")
            ## cat("Size of subsample:                           ", x$parm@fsz, "\n")
            cat("Number of time points to evaluate CON:             ", x$control$K, "\n")
            cat("Min. number of baseline obs. in a splittable node: ", x$control$minSplitNode, "\n")
            cat("Min. number of baseline obs. in a terminal node:   ", x$control$minSplitTerm, "\n")
        }
    }
}

tree.split.names <- function(nd0, nd, p, cut, xname, digits = getOption("digits")) {
    if (nd0 == 1) return("root")
    ind <- which(nd == nd0 %/% 2)
    if (nd0 %% 2 == 0) {
        return(paste(xname[p[ind]], "<=", formatC(cut[ind], digits = digits, flag = "#")))
    } else {
        return(paste(xname[p[ind]], ">", formatC(cut[ind], digits = digits, flag = "#")))
    }
}

#' Function to print a tree, this is called by print.dynTree()
#' 
#' @param Frame is the treeMat produced by `dynTree()`
#' @param vNames is a vector consists of variable's names entered in the formula.
#' The length of this vector must equal to p, the total number of covariates.
#' 
#' @keywords internal
#' @noRd
printTree <- function(Frame, vNames, digits) {
    ## create data.tree
    root <- Node$new("Root", type = "root", decision = "", nd = 1)
    if (nrow(Frame) > 1) {
        for (i in 2:nrow(Frame)) {
            if (i <= 3) parent <- "root"
            if (i > 3) parent <- paste0("Node", Frame$nd[i] %/% 2)
            if (Frame$is.terminal[i] > 0) {
                type <- "terminal"
                display <- with(Frame, paste0(nd[i], ") ", tree.split.names(nd[i], nd, p, cutVal, vNames, digits), "*"))
            } else {
                type <- "interior"
                display <- with(Frame, paste0(nd[i], ") ", tree.split.names(nd[i], nd, p, cutVal, vNames, digits)))
            }
            eval(parse(text = paste0("Node", Frame$nd[i], "<-", parent,
                                     "$AddChild(display, type = type, nd = Frame$nd[i])")))
        }
    }
    toPrint <- ToDataFrameTree(root)[[1]]
    cat(" Survival tree\n")
    if (nrow(Frame) > 1) {        
        cat("\n")
        cat(" node), split\n")
        cat("   * denotes terminal node\n")
        cat("  ", toPrint, sep = "\n")
    } else {
        cat(" Decision tree found no splits.")
    }
    cat("\n")
}

#' Print a predicted object
#'
#' @param x is an \code{dynTree} object
#' @noRd
#' 
#' @exportS3Method print predict.dynTree
print.predict.dynTree <- function(x, ...) {
    if (!is.predict.dynTree(x)) stop("Response must be a 'predict.dynTree' object")
    if (names(x$pred)[[2]] == "Survival") {
        cat(" Fitted survival probabilities:\n")
    }
    if (names(x$pred)[[2]] == "hazard") {
        cat(" Fitted cumulative hazard:\n")
    }
    print(head(x$pred, 5))
    cat("\n")
}
