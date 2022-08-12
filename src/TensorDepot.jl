module TensorDepot

using Random
using StatsBase
using LinearAlgebra

using MatrixDepot
using MatrixDepot: include_generator, FunctionName, Group, publish_user_generators

using Scratch
download_cache = ""

include("OtherDownload.jl")

export humansketches, mnist, fashionmnist, census, covtype, kddcup, poker, power, spgemm

"""
random sparse Stochastic Kronecker tensor
========================
stockronrand([rng], [T], D, p, [rand]) = (I, V, dims)

generate a random tensor according to the probability distribution

kron(D...) ./ sum(kron(D...))

*Input options:*
+ [rng]: a random number generator
+ [T]: an element type
+ D: an iterator over probability Arrays
+ p: the number of nonzero values to sample, as an integer or fraction of the total size.
+ [rand]: a random function to generate values

*Outputs:*
+ I: unsorted output coordinate vectors, with duplicates
+ V: The output values, with duplicates
+ dims: The size of the output tensor

*Examples*

The output of this function may be passed to sparse, as:

```
sparse(mdopen("stockronrand", Iterators.repeated([0.9 0.1; 0.9 0.1], 4), 200).A...)
```
"""
stockronrand(D, m) = stockronrand(Float64, D, m, rand)
stockronrand(D, m, rand) = stockronrand(Float64, D, m, rand)
stockronrand(T::Type, D, m) = stockronrand(Random.default_rng(), T, D, m, rand)
stockronrand(T::Type, D, m, rand) = stockronrand(Random.default_rng(), T, D, m, rand)
stockronrand(rng::AbstractRNG, D, m) = stockronrand(rng, Float64, D, m, rand)
stockronrand(rng::AbstractRNG, D, m, rand) = stockronrand(rng, Float64, D, m, rand)
stockronrand(rng::AbstractRNG, T::Type, D, m) = stockronrand(rng, T, D, m, rand)
stockronrand(rng::AbstractRNG, T::Type, D, m::AbstractFloat, rand) = 
    stockronrand(rng, T, D, ceil(Int, mapreduce(length, *, D) * m), rand)
function stockronrand(rng::AbstractRNG, T::Type, D, m::Integer, rand::Rand) where {Rand}
    dims = mapreduce(size, .*, D)
    N = length(dims)
    D = map(d -> (d ./ sum(d)), D)
    I = ntuple(_->Int[], N)
    V = rand(rng, T, dims...)
    for _ = 1:m
        i = ntuple(n->1, N)
        for d in D
            i = (i .- 1) .* size(D) .+ Tuple(sample(CartesianIndices(size(d)), Weights(reshape(d, :))))
        end
        push!.(I, i)
    end
    return (A = (I, V, dims), )
end

function __init__()
    include_generator(FunctionName, "stockronrand", stockronrand)
    include_generator(Group, :random, stockronrand)
    include_generator(FunctionName, "humansketches", humansketches)
    include_generator(FunctionName, "mnist", mnist)
    include_generator(FunctionName, "fashionmnist", fashionmnist)
    include_generator(FunctionName, "census", census)
    include_generator(FunctionName, "covtype", covtype)
    include_generator(FunctionName, "kddcup", kddcup)
    include_generator(FunctionName, "poker", poker)
    include_generator(FunctionName, "power", power)
    include_generator(FunctionName, "spgemm", spgemm)
    publish_user_generators()

    if haskey(ENV, "TENSORDEPOT_DATA")
        global download_cache = ENV["TENSORDEPOT_DATA"]
    else 
        global download_cache = get_scratch!(@__MODULE__, "tensors")
    end
end

end
