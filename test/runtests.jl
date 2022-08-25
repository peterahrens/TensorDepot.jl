using TensorDepot
using MatrixDepot
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@testset "TensorDepot.jl" begin
    @test size(humansketches(1:10)) == (10,1111,1111)
    @test size(matrixdepot("humansketches", 1:10)) == (10,1111,1111)

    if "local" in ARGS
        @test size(humansketches()) == (20000,1111,1111)
        @test size(matrixdepot("humansketches")) == (20000,1111,1111)
    end

    @test_throws BoundsError humansketches(100000:1000001)
    @test_throws BoundsError matrixdepot("humansketches", 100000:1000001)

    if "local" in ARGS
        @test size(spgemm()) == (241600,18)
        @test size(matrixdepot("spgemm")) == (241600,18)
    end
end