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

    @test logpdf(d, 0.42) ≈ log(pdf(d, 0.42))
    # out of support
    @test isinf(logpdf(d,  2.0))
    @test isinf(logpdf(d, -2.0))

    # on support limits
    d = Kumaraswamy(.5, .5)
    @test isinf(logpdf(d, 0.0))
    @test isinf(logpdf(d, 1.0))
    d = Kumaraswamy(4, 2)
    @test isinf(logpdf(d, 0.0))
    @test isinf(logpdf(d, 1.0))

    # derivative
    for d in (Kumaraswamy(2.0, 1.0), Kumaraswamy(0.125, 0.1))
        for v in 0.00:0.2:1.00
            fgrad = ForwardDiff.derivative(x -> logpdf(d, x), v)
            glog = gradlogpdf(d, v)
            # ForwardDiff fails for a = 2, b = 1, x = 1 and a = 1/8, b = 1/10, x = 0
            @test isnan(fgrad) || fgrad ≈ glog
        end
    end
end
