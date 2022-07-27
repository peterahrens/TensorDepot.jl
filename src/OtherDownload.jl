# hwd+, ilsvrc, sketches

# sketches: https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
# hwd+ (drive folder): https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or 
# hwd+ (500x500 npy): https://drive.google.com/file/d/1CInd1YOC0lsEq4_q089SVb7PkhxtrwyE/view?usp=sharing
# ilsvrc (kaggle login needed, and its huge!): https://www.kaggle.com/competitions/imagenet-object-localization-challenge/

# using HTTP
using ZipFile
using Images
using ImageView

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

function download_sketches()
    sketches_link = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    loc, unpack = download_dataset(sketches_link, "sketches")
    unzip_path = joinpath(dirname(loc), "pngs")
    if unpack
        unzip(loc, unzip_path, true)
    end
    return unzip_path
end

function sketches(start=1, stop=20000)
    if start < 1 || start > 20000 || stop < 1 || stop > 20000 || stop < start
        error("Sketch argument error: start=$start, stop=$stop")
    end

    path = download_sketches()

    out = Array{Gray{N0f8}, 3}(undef, stop-start+1, 1111,1111)

    for i in start:stop
        arr_idx = i - start + 1
        out[arr_idx, :, :] = load(joinpath(path, "$i.png"))
    end
    return out
end