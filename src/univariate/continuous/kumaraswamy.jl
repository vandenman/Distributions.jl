"""
    Kumaraswamy(a, b)

The *Kumaraswamy distribution* has probability density function

```math
f(x; a, b) = abx^{a - 1}(1 - x^a)^{b - 1}, \\quad x \\in [0, 1]
```

The Kumaraswamy distribution is related to the [`Uniform`](@ref) distribution via the
property that if ``X \\sim \\operatorname{Uniform}()`` then 
``(1 - (1 - X)^\\frac{1}{b})^\\frac{1}{a} \\sim Kumaraswamy(a, b)``.
Furthermore, the  Kumaraswamy distribution is closely related to the [`Beta`](@ref) distribution.
In particular, we have if ``X \\sim \\operatorname{Kumaraswamy}(a, 1)`` then  
``X \\sim operatorname{Beta}(a, 1)`` and if ``X \\sim \\operatorname{Kumaraswamy}(1, b)`` then  
``X \\sim operatorname{Beta}(1, b)``.


```julia
Kumaraswamy()        # equivalent to Kumaraswamy(1, 1)
Kumaraswamy(a, b)    # Kumaraswamy distribution with shape parameters a and b

params(d)            # Get the parameters, i.e. (α, β)
```

External links

* [Kumaraswamy distribution on Wikipedia](https://en.wikipedia.org/wiki/Kumaraswamy_distribution)

"""
struct Kumaraswamy{T<:Real} <: ContinuousUnivariateDistribution 
    a::T
    b::T
    Kumaraswamy{T}(a::T, b::T) where {T} = new{T}(a, b)
end

function Kumaraswamy(a::T, b::T; check_args::Bool=true) where {T<:Real}
    @check_args Kumaraswamy (a, a > zero(a)) (b, b > zero(b))
    return Kumaraswamy{T}(a, b)
end

Kumaraswamy(a::Real, b::Real; check_args::Bool=true) = Kumaraswamy(promote(a, b)...; check_args=check_args)
Kumaraswamy(a::Integer, b::Integer; check_args::Bool=true) = Kumaraswamy(float(a), float(b); check_args=check_args)


Kumaraswamy() = Kumaraswamy{Float64}(1.0, 1.0)

Distributions.@distr_support Kumaraswamy 0.0 1.0

#### Parameters

params(d::Kumaraswamy) = (d.a, d.b)
@inline partype(::Kumaraswamy{T}) where {T<:Real} = T


#### Statistics

function _rawmoments(d::Kumaraswamy, n::Integer)
    a, b = params(d)
    # TODO: computing this on a log scale might make variance, skewness, and kurtosis more precise
    b * beta(1 + n/a , b)
end

function mean(d::Kumaraswamy) 
    _rawmoments(d, 1)
    # a, b = params(d)
    # return SpecialFunctions.gamma(1 + 1 / a) * SpecialFunctions.gamma(b) / SpecialFunctions.gamma(1 + 1 / a + b)
end

function variance(d::Kumaraswamy) 
    _rawmoments(d, 2) - _rawmoments(d, 1)^2
end

function skewness(d::Kumaraswamy) 
    μ = mean(d)
    m2 = _rawmoments(d, 1)
    σ = m2 - μ^2
    m3 = _rawmoments(d, 3)
    return (m3 - 3 * μ * σ^2 - μ^3) / σ^3
end

function kurtosis(d::Kumaraswamy)
    m1, m2, m3, m4 = _rawmoments.(d, 1:4)
    return -3 * m1^3 + 6 * m1^2 * m2 - 4 * m1 * m3 + m4 
end

function median(d::Kumaraswamy)
    a, b = params(d)
    (1 - 2^(-1 / b))^(1 / a)
end

function mode(d::Kumaraswamy; check_args::Bool=true)
    a, b = params(d)
    @check_args(
        Kumaraswamy,
        (a, a >= one(a) && b >= one(b) && !(a == one(a) && b == one(b)), "mode is defined only when a >= 1 and b >= 1 and (a, b) != (1, 1).")
    )
    return (a - 1) / (a * b - 1)^(1 / a)
end

modes(d::Kumaraswamy) = [mode(d)]

#### Evaluation

function pdf(d::Kumaraswamy{T}, x::Real) where T
    insupport(x, d) || return zero(T)
    a, b = params(d)
    return a * b * x^(a - 1) * (1 - x^a)^(b - 1)
end
function cdf(d::Kumaraswamy{T}, x::Real) where T
    x < zero(x) && return zero(T)
    x > one(x)  && return one(T)
    a, b = params(d)
    return 1 - (1 - x^a)^b
end
function ccdf(d::Kumaraswamy{T}, x::Real) where T
    x < zero(x) && return one(T)
    x > one(x)  && return zero(T)
    a, b = params(d)
    return (1 - x^a)^b
end

function logpdf(d::Kumaraswamy{T}, x::Real) where T
    insupport(x, d) || return T(-Inf)
    a, b = params(d)
    return log(a) + log(b) + (a - 1) * log(x) + (b - 1) * log1p(-x^a)
end
function logcdf(d::Kumaraswamy{T},  x::Real) where T
    x < zero(x) && return T(-Inf)
    x > one(x)  && return zero(T)
    a, b = params(d)
    return log1p((1 - x^a)^b)
end
function logccdf(d::Kumaraswamy{T},  x::Real) where T
    x < zero(x) && return zero(T)
    x > one(x)  && return T(-Inf)
    a, b = params(d)
    return b * log1p(-x^a)
end

function quantile(d::Kumaraswamy,  p::Real)
    # TODO: add check for 0 <= p <= 1?
    a, b = params(d)
    (1 - (1 - p)^(1 / b))^(1 / a)
end

# TODO: maybe not implement these if they do the same thing as the fallback?
# function cquantile(d::Kumaraswamy,  p::Real) return quantile(d, 1 - p) end
# function invlogcdf(d::Kumaraswamy,  p::Real) end
# function invlogccdf(d::Kumaraswamy,  p::Real) end

function gradlogpdf(d::Kumaraswamy{T}, x::R) where {T, R <: Real}
    TP = promote_type(T, R)
    (a, b) = params(d)

    ## special cases 
    # uniform distribution
    isone(a) && isone(b) && return zero(TP)

    # boundary values
    if iszero(x)
        a < one(a) && return TP(-Inf)
        a > one(a) && return TP(Inf)
        return TP(1 - b) # a == isone(a)
        return 
    elseif isone(x)
        b < one(b) && return TP(-Inf)
        b > one(b) && return TP(Inf)
        return TP(a - 1) # b == isone(b)
    end

    # gradient of the logpdf(d, x) wrt x
    return TP(
        (1 - x^a + a * (b * x^a - 1)) / (x * (x^a - 1))
    )
end



#### Sampling
function rand(rng::AbstractRNG, d::Kumaraswamy)
    return quantile(d, rand(rng))
end


#### Fit model

# Read https://www.tandfonline.com/doi/abs/10.1080/00949655.2010.511621
"""
    fit_mle(::Type{<:Kumaraswamy}, x::AbstractArray{T})
Maximum Likelihood Estimate of `Kumaraswamy` Distribution via TODO
"""
function fit_mle(::Type{<:Kumaraswamy}, x::AbstractArray{T};
    maxiter::Int=1000, tol::Float64=1e-14) where T<:Real
end

"""
    fit(::Type{<:Kumaraswamy}, x::AbstractArray{T})
fit a `Kumaraswamy` distribution
"""
function fit(::Type{<:Kumaraswamy}, x::AbstractArray{T}) where T<:Real
end