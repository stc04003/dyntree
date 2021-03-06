#' Survival trees for landmark analysis
#'
#' Fits a "\code{dynTree}" model.
#'
#' The argument "control" defaults to a list with the following values:
#' \describe{
#'   \item{\code{tau}}{is the maximum follow-up time; default value is the 90th percentile of the unique observed survival times.}
#'   \item{\code{maxNode}}{is the maximum node number allowed to be in the tree; the default value is 500.}
#'   \item{\code{numFold}}{is the number of folds used in the cross-validation. When \code{numFold > 0}, the survival tree will be pruned;
#' when \code{numFold = 0}, the unpruned survival tree will be presented. The default value is 10.}
#'   \item{\code{h}}{is the smoothing parameter used in the Kernel; the default value is \code{tau / 20}.}
#'   \item{\code{minSplitTerm}}{is the minimum number of baseline observations in each terminal node; the default value is 15.}
#'   \item{\code{minSplitNode}}{is the minimum number of baseline observations in each splitable node; the default value is 30.}
#'   \item{\code{disc}}{is a logical vector specifying whether the covariates in \code{formula} are discrete (\code{TRUE}) or continuous (\code{FALSE}).
#' The length of \code{disc} should be the same as the number of covariates in \code{formula}. When not specified, the \code{dynTree()} function assumes continuous covariates for all.}
#'   \item{\code{K}}{is the number of time points on which the concordance measure is computed.
#' A less refined time grids (smaller \code{K}) generally yields faster speed but a very small \code{K} is not recommended. The default value is 20.}
#' }
#' 
#' @param formula is a formula object, with the response on the left of a '~' operator,
#' and the terms on the right. The response must be a survival object returned by the
#' 'Surv' function.
#' @param data is an optional data frame in which to interpret the variables occurring
#' in the 'formula'.
#' @param id is an optional vector used to identify the longitudinal observations of subject's id.
#' The length of 'id' should be the same as the total number of observations.
#' If 'id' is missing, each row of `data` represents a distinct observation from a subject and
#' all covariates are treated as a baseline covariate. 
#' @param subset is an optional vector specifying a subset of observations to be used in
#' the fitting process.
#' @param ensemble is an optional logical value.
#' If \code{TRUE} (default), ensemble methods will be fitted.
#' Otherwise, the survival tree will be fitted.
#' @param groups is an optional vector with the length equal to the number of covariates.
#' This vector indicates the covariates to be splitted together in the emsemble method.
#' @param control a list of control parameters. See 'details' for important special
#' features of control parameters.
#' @param trans indicates whether to perform time-dependent transformation;
#' only makes sense when all covariates are time-independent.
#'
#' @export
#' @return An object of S4 class "\code{dynTree}" representig the fit, with the following components:
#'
#' @keywords dynTree
#' @seealso See
#'
#' @example inst/examples/ex_dynTree.R
#' @importFrom survival Surv
#' @importFrom stats aggregate predict stepfun
dynTree <- function(formula, data, id, subset, ensemble = TRUE,
                  groups = NULL, trans = TRUE,
                  control = list()) {
    control <- dynTree.control(control)
    Call <- match.call()
    if (missing(formula)) stop("Argument 'formula' is required.")
    if (missing(data))
        data <- environment(formula)
    ## take care of subset individual for possible non-numeric ID
    if (! missing(subset)) {
        sSubset <- substitute(subset)
        subIdx <- eval(sSubset, data, parent.frame())
        if (!is.logical(subIdx)) stop("'subset' must be logical")
        subIdx <- subIdx & ! is.na(subIdx)
        data <- data[subIdx, ]
    }
    ## extract information
    Call <- match.call(expand.dots = FALSE)
    callName <- match(c("formula", "data", "id"), names(Call), nomatch = 0L)
    mcall <- Call[c(1L, callName)]
    mcall[[1L]] <- quote(stats::model.frame)
    mf <- eval(mcall, parent.frame())
    mt <- attr(mf, "terms")
    .Y <- model.response(mf)[,1]
    .D <- model.response(mf)[,2]
    .X <- stats::model.matrix(formula, data = mf)
    ## if (any(colnames(.X) == "(Intercept)"))
    ##     .X <- .X[,!(colnames(.X) == "(Intercept)"), drop = FALSE]
    if (length(grep("`|\\(", colnames(.X))) > 0)
        .X <- .X[,-grep("`|\\(", colnames(.X)), drop = FALSE]
    .id <- model.extract(mf, id)
    names(.id) <- names(.Y) <- names(.D) <- rownames(.X) <- NULL
    if (is.null(.id) | length(.id) == length(unique(.id))) {
        .id <- 1:length(.Y)
        .n0 <- length(unique(.id))
        rk <- rank(.Y, ties.method = "first")
        .X <- .X[rep(1:.n0, rk),, drop = FALSE]
        .id <- .id[rep(1:.n0, rk)]
        .Dtmp <- .D
        .D <- rep(0, sum(1:.n0))
        .D[cumsum((1:.n0)[rk])] <- .Dtmp
        .ind <- unlist(sapply(1:.n0, function(x) 1:x)[rk])
        .Y <- as.numeric(sort(.Y)[.ind])
    } 
    tmp <- aggregate(.Y ~ .id, FUN = max)
    ord <- unlist(sapply(tmp$.id[order(tmp$.Y)], function(x) which(.id == x)), use.names = F)
    .id2 <- rep(1:length(unique(.id)), table(.id)[unique(.id[ord])])
    .p <- ncol(.X)
    if (is.null(groups)) gp <- 1:.p
    else gp <- rep(1:length(unique(groups)), unlist(lapply(split(groups, groups), length)))
    .n <- length(unique(.id))
    .Y <- .Y[ord]
    .D <- .D[ord]
    .X0 <- .X <- .X[ord,, drop = FALSE]
    .Y0 <- .Y[cumsum(table(.id2))]
    .D0 <- .D[cumsum(table(.id2))]
    if (is.function(control$mtry)) control$mtry <- control$mtry(.p)
    if (is.function(control$tau)) control$tau <- control$tau(.Y0)
    if (is.function(control$h)) control$h <- control$h(control$tau)
    if (length(control$disc) == 1) {
        disc <- rep(control$disc, .p)
        tmp <- as.numeric(which(apply(.X0, 2, function(x) length(unique(x))) <= 3))
        if (length(tmp) > 0) disc[tmp] <- TRUE
    } else disc <- control$disc[1:.p]
    cutoff <- (1:control$nc) / (control$nc + 1)
    ## .tk <- quantile(unique(.Y0[.D0 > 0]), 1:control$K / (control$K + 1), names = FALSE)
    .tk <- .Y0[.D0 > 0]
    .eps <- unlist(sapply(split(.id2, .id2), function(.x) 1:length(.x)))
    discClass <- rep(list(NULL), length(disc))
    for (i in which(disc)) {
        tmp <- as.factor(unique(.X[,i]))
        discClass[[i]] <- data.frame(value = sort(unique(c(tmp))),
                                     level = levels(tmp))
        .X[,i] <- with(discClass[[i]], value[match(.X[,i], level)])
    }    
    if (any(!disc)) {
        if (trans) .X[order(.Y), !disc] <- apply(.X[, !disc, drop = FALSE], 2, function(.x)
            unlist(lapply(split(.x, .eps), fecdf)))
        else
            .X[, !disc] <- apply(.X[.eps == 1, !disc, drop = FALSE], 2, fecdf)[.id2,]
        .X[, !disc] <-
            apply(.X[, !disc, drop = FALSE], 2, function(x) findInterval(x, cutoff)) + 1
    }
    ## if (any(!disc))
    ##     .X[order(.Y), !disc] <- apply(.X[, !disc, drop = FALSE], 2, function(.x)
    ##         unlist(lapply(split(.x, .eps), fecdf)))
    ## if (any(disc)) 
    ##     .X[, disc] <- apply(.X[, disc, drop = FALSE], 2, function(.x) .x / max(.x))
    ## .X <- apply(.X, 2, function(x) findInterval(x, cutoff)) + 1
    ## .hk <- rep(control$h, length(.tk))
    ## .hk[.tk < control$h] <- .tk[.tk < control$h]
    ## if (!trans) .X <- .X[rep(1, nrow(.X)),]
    ## .mat1f <- t(.D0 * mapply(function(x,h) K2(x, .Y0, h) / h, .tk, .hk))
    .mat1f <- t(.D0 * outer(.Y0, .tk, "=="))
    ## if (trans) {
    ## .mat1Z <- .X[.eps == 1,, drop = FALSE]
    .mat1Z <- .X[cumsum(table(.id2)),, drop = FALSE]
    .mat2k <- make_mat2(.tk, .Y, .id2, .X)
    .zt <- make_mat2_t(.Y0[.D0 == 1], .Y, .id2, .X)
    .range0  <- apply(.X, 2, range)
    ## } else {
    ##     .mat1f <- matrix(colSums(.mat1f), 1)
    ##     .ind2 <- cumsum(table(.id2))
    ##     .mat1Z <- .X[.eps == 1,, drop = FALSE]
    ##     .mat2k <- make_mat2(.tk[1], .Y[.ind2], .id2[.ind2], .X[.eps == 1,])
    ##     .zt <- make_mat2_t(.Y0[.D0 == 1][1], .Y[.ind2], .id2[.ind2], .X[.eps == 1,])
    ##     .range0 <- apply(.X[.eps == 1,, drop = FALSE], 2, range)
    ## }
    .zy <- t(.mat1Z[.D0 == 1,])
    if (ensemble) {
        out <- dynforest_C(.mat1f, .mat1Z, .mat2k, .range0, .zt, .zy, .D0, 1, 
                         control$numTree,
                         control$minSplitTerm,
                         control$minSplitNode,
                         control$maxNode,
                         control$mtry, gp)
        out$Frame <- lapply(out$trees, cleanTreeMat, cutoff = cutoff, .X0 = .X0, disc = disc)
    } else {
        out <- dyntree_C(.mat1f, .mat1Z, .mat2k, .range0, .zt, .zy, .D0, 1, 
                       control$numFold,
                       control$minSplitTerm,
                       control$minSplitNode,
                       control$maxNode)
        out$Frame <- cleanTreeMat(out$treeMat, cutoff = cutoff, .X0 = .X0, disc = disc)
    }
    out$call <- Call
    ## out$data <- list(.Y = .Y, .D = .D, .X = .X0, .Y0 = .Y0, .D0 = .D0,
    ##                  .X0 = .X0[cumsum(table(.id2)),], .id = .id, .id2 = .id2)
    tmp <- table(.id2)
    out$data <- list(.Y0 = .Y0, .D0 = .D0, .X0 = .X0[cumsum(table(.id2)),], .id = unique(.id),
                     .id2 = as.numeric(names(tmp)), .id2.count = as.numeric(tmp))
    out$rName <- all.vars(formula)[1]
    out$vNames <- attr(mt, "term.labels")
    out$ensemble <- ensemble
    out$disc <- disc
    out$trans <- trans
    out$discClass <- discClass
    out$control <- control
    class(out) <- "dynTree"
    return(out)
}

#' dynTree controls
#' 
#' @keywords internal
#' @noRd
dynTree.control <- function(l) {
    ## default values
    dl <- list(numTree = 500, numFold = 10, minSplitNode = 30, minSplitTerm = 15,
               maxNode = 500, K = 20, nc = 200, disc = FALSE,
               tau = function(x) quantile(x, .9),
               h = function(x) x / 20, 
               mtry = function(x) ceiling(sqrt(x)))
    l.name <- names(l)    
    if (!all(l.name %in% names(dl)))
        warning("unknown names in control are ignored: ", l.name[!(l.name %in% names(dl))])
    dl[match(l.name, names(dl))] <- l
    ## if (is.null(dl$hN)) dl$hN <- dl$tau / 20
    return(dl)
}

#' Check if it is a `dynTree` object
#' @keywords internal
#' @noRd
is.dynTree <- function(x) inherits(x, "dynTree")



#' Clean the `treeMat` from tree and forests; make it easier to read and compatible with print function
#' @keywords internal
#' @noRd
cleanTreeMat <- function(treeMat, cutoff, .X0, disc) {
    ## prepraing treeMat
    ## Remove 0 rows and redefine child nodes
    ## 0 rows were produced from prunning
    treeMat <- data.frame(treeMat)
    names(treeMat) <- c("p", "cutOrd", "left", "right", "is.terminal")
    treeMat$p <- ifelse(treeMat$is.terminal == 1, NA, treeMat$p + 1)
    ## treeMat$p + (1 - treeMat$is.terminal)
    treeMat$left <- ifelse(treeMat$left == 0, NA, treeMat$left + 1)
    treeMat$right <- ifelse(treeMat$right == 0, NA, treeMat$right + 1)
    mv <- rowSums(treeMat[,2:5], na.rm = TRUE) == 0
    if (sum(mv) > 0) {
        treeMat <- treeMat[-which(mv),]
        treeMat$left <- match(treeMat$left, rownames(treeMat))
        treeMat$right <- match(treeMat$right, rownames(treeMat))
        rownames(treeMat) <- NULL
    }
    if (nrow(treeMat) > 1) {
        treeMat$cutVal <- cutoff[ifelse(treeMat$cutOrd > 0, treeMat$cutOrd, NA)]
        if (sum(treeMat$p %in% which(disc))) {
            for (i in which(disc)) {
                ind <- which(treeMat$p == i)
                treeMat$cutVal[ind] <- as.numeric(levels(as.factor(.X0[,i]))[treeMat$cutOrd[ind]])
            }
        }        
        if (nrow(treeMat) <= 3) {
            treeMat$nd <- 1:3
        } else {
            nd <- 1:3
            for (i in 2:(nrow(treeMat) - 2)) {
                if (treeMat$is.terminal[i] == 0) nd <- c(nd, 2 * nd[i], 2 * nd[i] + 1)
            }
            treeMat$nd <- nd
        }
    } else {
        treeMat$cutVal <- NA
        treeMat$nd <- 1
    }
    treeMat$cutOrd <- ifelse(treeMat$is.terminal == 1, NA, treeMat$cutOrd)
    return(treeMat)
}
