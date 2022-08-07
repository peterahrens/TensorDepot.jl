using TensorDepot
using MatrixDepot
using Test

@testset "TensorDepot.jl" begin
    @test size(mnist(1:10)) == (10,28,28)
    @test size(matrixdepot("mnist", 1:10)) == (10,28,28)

    @test size(mnist()) == (60000,28,28)
    @test size(matrixdepot("mnist")) == (60000,28,28)

    @test_throws BoundsError mnist(100000:1000001)
    @test_throws BoundsError matrixdepot("mnist", 100000:1000001)

    @test size(fashionmnist(1:10)) == (10,28,28)
    @test size(matrixdepot("fashionmnist", 1:10)) == (10,28,28)

    @test size(fashionmnist()) == (60000,28,28)
    @test size(matrixdepot("fashionmnist")) == (60000,28,28)

    @test_throws BoundsError fashionmnist(100000:1000001)
    @test_throws BoundsError matrixdepot("fashionmnist", 100000:1000001)

    @test size(humansketches(1:10)) == (10,1111,1111)
    @test size(matrixdepot("humansketches", 1:10)) == (10,1111,1111)

    @test size(humansketches()) == (20000,1111,1111)
    @test size(matrixdepot("humansketches")) == (20000,1111,1111)

    @test_throws BoundsError humansketches(100000:1000001)
    @test_throws BoundsError matrixdepot("humansketches", 100000:1000001)

    @test size(spgemm()) == (241600,18)
    @test size(matrixdepot("spgemm")) == (241600,18)
end
