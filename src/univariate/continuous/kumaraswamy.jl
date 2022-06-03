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

@distr_support Kumaraswamy 0.0 1.0

#### Conversions
function convert(::Type{Kumaraswamy{T}}, a::Real, b::Real) where T<:Real
    Kumaraswamy(T(a), T(b))
end
Base.convert(::Type{Kumaraswamy{T}}, d::Kumaraswamy) where {T<:Real} = Kumaraswamy{T}(T(d.a), T(d.b))
Base.convert(::Type{Kumaraswamy{T}}, d::Kumaraswamy{T}) where {T<:Real} = d

#### Parameters

params(d::Kumaraswamy) = (d.a, d.b)
@inline partype(::Kumaraswamy{T}) where {T<:Real} = T


#### Statistics

function _rawmoment(d::Kumaraswamy, n::Integer)
    a, b = params(d)
    b * beta(1 + n/a , b)
end

function mean(d::Kumaraswamy) 
    _rawmoment(d, 1)
end

function variance(d::Kumaraswamy) 
    _rawmoment(d, 2) - _rawmoment(d, 1)^2
end

function skewness(d::Kumaraswamy) 
    m1, m2, m3 = _rawmoment.(d, 1:3)
    σ = m2 - μ^2
    return (m3 - 3 * m1 * σ^2 - m1^3) / σ^3
end

function kurtosis(d::Kumaraswamy)
    m1, m2, m3, m4 = _rawmoment.(d, 1:4)
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

function entropy(d::Kumaraswamy)
    a, b = params(d)
    (1 - 1 / a) + (1 - 1 / b) * harmonic(b) - log(a * b)
end

"""
Compute the harmonic number for real value x using H(x) = digamma(x+1) + γ where γ is the Euler–Mascheroni constant.
See https://en.wikipedia.org/wiki/Harmonic_number#Harmonic_numbers_for_real_and_complex_values
"""
function harmonic(x::Real)
    SpecialFunctions.digamma(x + one(x)) + Base.MathConstants.γ
end

#### Evaluation

function pdf(d::Kumaraswamy{T}, x::R) where {T, R<:Real}
    insupport(d, x) || return zero(promote_type(T, R))
    a, b = params(d)
    return a * b * x^(a - 1) * (1 - x^a)^(b - 1)
end

function cdf(d::Kumaraswamy{T}, x::R) where {T, R<:Real}
    x < zero(x) && return zero(promote_type(T, R))
    x > one(x)  && return one(promote_type(T, R))
    a, b = params(d)
    return 1 - (1 - x^a)^b
end

function logpdf(d::Kumaraswamy{T}, x::R) where {T, R<:Real}
    insupport(d, x) || return promote_type(T, R)(-Inf)
    a, b = params(d)
    return log(a) + log(b) + (a - 1) * log(x) + (b - 1) * log1p(-x^a)
end

function logcdf(d::Kumaraswamy{T}, x::R) where {T, R<:Real}
    x < zero(x) && return promote_type(T, R)(-Inf)
    x > one(x)  && return zero(promote_type(T, R))
    a, b = params(d)
    return log1p((1 - x^a)^b)
end

function logccdf(d::Kumaraswamy{T}, x::R) where {T, R<:Real}
    x < zero(x) && return zero(promote_type(T, R))
    x > one(x)  && return promote_type(T, R)(-Inf)
    a, b = params(d)
    return b * log1p(-x^a)
end

function quantile(d::Kumaraswamy,  p::Real)
    # TODO: add check for 0 <= p <= 1?
    # TODO: could also call a, b = params(d); cdf(Kumaraswamy(1/a, 1/b), p)
    a, b = params(d)
    (1 - (1 - p)^(1 / b))^(1 / a)
end


function gradlogpdf(d::Kumaraswamy{T}, x::R) where {T, R <: Real}
    TP = promote_type(T, R)
    (a, b) = params(d)

    ## special cases 
    # uniform distribution
    isone(a) && isone(b) && return zero(TP)

    # boundary values -- limits are from p. 73 of Jones, M. C. (2009). Kumaraswamy’s distribution: A beta-type distribution with some tractability advantages. Statistical methodology, 6(1), 70-81.
    if iszero(x)
        # return TP((a - 1) * log(x))
        a < one(a) && return TP(-Inf)
        a > one(a) && return TP(Inf)
        return TP(1 - b) # a == isone(a)
    elseif isone(x)
        # return TP((b - 1 ) * log1p(-x))
        b < one(b) && return TP(Inf)
        b > one(b) && return TP(-Inf)
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

"""
    fit_mle(::Type{<:Kumaraswamy}, x::AbstractArray{T})
Maximum Likelihood Estimate of `Kumaraswamy` Distribution via Newton's Method
"""
function fit_mle(::Type{<:Kumaraswamy}, x::AbstractArray{T};
    maxiter::Int=1000, tol::Float64=1e-14) where T<:Real

    n = length(x)

    a = one(T) # starting value

    converged = false
    t = 0
    while !converged && t < maxiter
        #= 
        Newton–Raphson algorithm based on 
        
        Lemonte, Artur J. (2011). "Improved point estimation for the Kumaraswamy distribution". 
        Journal of Statistical Computation and Simulation. 81 (12): 1971–1982. 
        doi:https://doi.org/10.1080%2F00949655.2010.511621
        
        T1, T2, T3 correspond are defined on p. 1973, but without dividing by n
        =#

        # cleaner but not type stable
        # T1 = sum(x->      log(x) / (1 - x^a), x)
        # T2 = sum(x->x^a * log(x) / (1 - x^a), x)
        # T3 = sum(x-> log1p(-x^a),             x)

        # ∂T1∂a =  sum(x->x^a * log(x)^2 / (1 - x^a)^2, x)
        # ∂T2∂a =  sum(x->x^a * log(x)^2 / (1 - x^a)^2, x)
        # ∂T3∂a = -sum(x->x^a * log(x)   / (1 - x^a),   x)

        T1 = T2 = T3 = ∂T1∂a = ∂T2∂a = ∂T3∂a = zero(T)
        for y in x

            yᵃ = y^a

            temp0 = log(y) / (1 - yᵃ)
            T1 +=       temp0
            T2 += y^a * temp0
            T3 += log1p(-yᵃ)

            temp1 = log(y) / (1 - yᵃ)
            ∂T1∂a += yᵃ * temp1^2
            ∂T2∂a += yᵃ * temp1^2
            ∂T3∂a += yᵃ * temp1

        end


        ∂lp∂a   = n * (1 / a + T2 / T3) + T1
        ∂²lp∂²a = n * (-1 / a^2 + (T3 * ∂T2∂a - T2 * ∂T3∂a) / T3^2) + ∂T1∂a

        Δa = ∂lp∂a / ∂²lp∂²a
        a -= Δa
        converged = abs(Δa) <= tol
        t += 1

    end
    
    # cleaner but not type stable
    # b = -n / sum(x->log1p(-x^a), x)
    temp = zero(T)
    for y in x
        temp += log1p(-y^a)
    end
    b = -n / temp

    return Kumaraswamy(a, b)
end
