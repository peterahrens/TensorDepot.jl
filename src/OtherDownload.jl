# hwd+, ilsvrc, sketches

# sketches: https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
# hwd+ (drive folder): https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or 
# hwd+ (500x500 npy): https://drive.google.com/file/d/1CInd1YOC0lsEq4_q089SVb7PkhxtrwyE/view?usp=sharing
# ilsvrc (kaggle login needed, and its huge!): https://www.kaggle.com/competitions/imagenet-object-localization-challenge/

# using HTTP
using ZipFile
using Images

function unzip(file,exdir="",flatten=false)
    fileFullPath = isabspath(file) ?  file : joinpath(pwd(),file)
    basePath = dirname(fileFullPath)
    outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
    isdir(outPath) ? "" : mkdir(outPath)
    zarchive = ZipFile.Reader(fileFullPath)
    for f in zarchive.files
        fullFilePath = joinpath(outPath,f.name)
        if flatten 
            if !(endswith(f.name,"/") || endswith(f.name,"\\"))
                write(joinpath(outPath,basename(fullFilePath)), read(f))
            end
        else
            if (endswith(f.name,"/") || endswith(f.name,"\\"))
                mkdir(fullFilePath)
            else
                write(fullFilePath, read(f))
            end
        end
    end
    close(zarchive)
end

function download_dataset(url, name)
    path = joinpath(download_cache, name)
    fname = joinpath(path, basename(url))
    if !isfile(fname)
        mkpath(path)
        download(url, fname)
        return fname, true
    else
        return fname, false
    end
end

function download_humansketches()
    sketches_link = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    loc, unpack = download_dataset(sketches_link, "sketches")
    unzip_path = joinpath(dirname(loc), "pngs")
    if unpack
        unzip(loc, unzip_path, true)
    end
    return unzip_path
end

"""
humansketches dataset tensor
========================
humansketches([idxs])

Return a 3-tensor A[sketch number, vertical pixel position, horizontal pixel
position], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. `idxs` is an optional list specifying which sketch images to
load. The sketches number from 1:20_000.
"""
function humansketches(idxs = 1:20_000)
    @boundscheck begin
        extrema(idxs) ⊆ 1:20_000 || throw(BoundsError(sketches, idxs))
    end

    path = download_humansketches()

    out = Array{Gray{N0f8}, 3}(undef, length(idxs), 1111,1111)

    for (n, i) in enumerate(idxs)
        out[n, :, :] = load(joinpath(path, "$i.png"))
    end
    return out
end