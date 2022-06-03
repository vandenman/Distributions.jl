
Kumaraswamy <- R6Class("Kumaraswamy",
    inherit = ContinuousDistribution,
    public = list(
        names = c("a", "b"),
        a = NA,
        b = NA,
        initialize = function(a=1, b=a) {
            self$a <- a
            self$b <- b
        },
        supp = function() { c(0.0, 1.0) },
        properties = function() {
            return(list())
        },
        pdf = function(x, log=FALSE) { extraDistr::dkumar(x, self$a, self$b, log=log) },
        cdf = function(x) { extraDistr::pkumar(x, self$a, self$b) },
        quan = function(v) { extraDistr::qkumar(v, self$a, self$b) }
    )
)
