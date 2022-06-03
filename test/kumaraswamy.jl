using Test
using Distributions
using ForwardDiff

@testset "Kumaraswamy" begin
    @test_throws DomainError Kumaraswamy(-1, 1)
    d = Kumaraswamy(3, 5)
    d2 = Kumaraswamy(3.5f0, 5)
    @test partype(d) == Float64
    @test partype(d2) == Float32
    @test d == deepcopy(d)

    # out of support
    @test logpdf(d,  2.0) == -Inf
    @test logpdf(d, -2.0) == -Inf

    # check for signed infinity on support limits
    d = Kumaraswamy(.5, .5)
    @test logpdf(d, 0.0) == Inf
    @test logpdf(d, 1.0) == Inf
    d = Kumaraswamy(4, 2)
    @test logpdf(d, 0.0) == -Inf
    @test logpdf(d, 1.0) == -Inf

    # derivative
    for d in (Kumaraswamy(2.0, 1.0), Kumaraswamy(0.125, 0.1))
        for v in 0.00:0.2:1.00
            fgrad = ForwardDiff.derivative(x -> logpdf(d, x), v)
            glog = gradlogpdf(d, v)
            # ForwardDiff fails (returns NaN) at the boundaries (a = 2, b = 1, x = 1 and a = 1/8, b = 1/10, x = 0)
            @test isnan(fgrad) || fgrad ≈ glog
        end
    end
end
