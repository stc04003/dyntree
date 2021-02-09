#' Predicting based on a \code{dynTree} model.
#'
#' The function gives predicted values with a \code{dynTree} fit.
#'
#' @param object is an \code{dynTree} object.
#' @param newdata is an optional data frame in which to look for variables with which to predict.
#' If omitted, the fitted predictors are used.
#' If the covariate observation time is not supplied, covariates will be treated as at baseline.
#' @param type is an optional character string specifying whether to predict the survival probability or the cumulative hazard rate. (One observation)
#' @param control a list of control parameters. See 'details' for important special
#' features of control parameters. See \code{\link{dynTree}} for more details.
#' @param ... for future developments.
#'
#' @return Returns a \code{data.frame} of the predicted survival probabilities or cumulative hazard. 
#'
#' @importFrom stats model.frame
#' @export
#' @example inst/examples/ex_predict_dynTree.R
predict.dynTree <- function(object, newdata, control = list(), ...) {
    if (!is.dynTree(object)) stop("Response must be a 'dynTree' object")
    if (missing(newdata)) stop("Argument 'newdata' is missing")
    if (!(object$rName %in% colnames(newdata)))
        stop(paste("Object '", object$rName, "' not found.", sep = ""))
    if (!all(object$vName %in% colnames(newdata))) {
        missingName <- which(!(object$vName %in% colnames(newdata)))
        if (length(missingName) == 1)
            stop(paste("Object '", object$vName[missingName], "' not found.", sep = ""))
        if (length(missingName) > 1)
            stop(paste("Objects '", paste(object$vName[missingName], collapse = ", "),
                       "' not found.", sep = ""))
    }    
    control0 <- object$control
    control0[names(control0) %in% names(control)] <- control[names(control) %in% names(control0)]
    control <- control0
    raw <- newdata[findInt(object$data$.Y0, unlist(newdata[object$rName])), object$vNames]
    rownames(raw) <- NULL
    cutoff <- (1:control$nc) / (control$nc + 1)
    .X <- object$data$.X0[rep(object$data$.id2, object$data$.id2),]
    for (i in which(object$disc)) {
        raw[,i] <- with(object$discClass[[i]], value[match(raw[,i], level)])
        ## .X[,i] <- with(object$discClass[[i]], value[match(.X[,i], level)])
    }    
    ## if (!object$trans) object$nodeSize <- object$nodeSize[, rep(1, sum(object$data$.D))]
    if (object$ensemble) {
        ## if (!object$trans) object$nodeLabel <- object$nodeLabel[, rep(1, sum(object$data$.D))]
        pred <- predict_dynforest_C(t(raw), object$data$.Y0, object$data$.D0, object, .X)
    } else
        pred <- predict_dyntree_C(t(raw), object$data$.Y0, object$data$.D0, object, .X)
    object$survFun <- stepfun(object$data$.Y0, c(1, pred))
    object$pred <- data.frame(Time = unlist(newdata[,object$rName]),
                              Survival = object$survFun(unlist(newdata[,object$rName])))
    rownames(object$pred) <- NULL
    class(object) <- "predict.dynTree"
    return(object)
}

is.predict.dynTree <- function(x) inherits(x, "predict.dynTree")

#' findInterval with 0 replaced with 1
#' @keywords internal
#' @noRd
findInt <- function(x, y) {
    pmax(1, findInterval(x, sort(y)))
}

#' findInterval with 0 replaced with 1, works with NA's in y
#' @keywords internal
#' @noRd
findInt.X <- function(x, y) {
    order(c(0, y))[pmax(1, findInterval(x, sort(c(0, y))))]
    ## order(y)[pmax(1, findInterval(x, sort(y)))]
}
