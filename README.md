# VectorAIzer

Semantic vectorization of raster images using object segmentation

## CLI usage

The `vectoraizer.py` command line interface can be used to run inference on a single image file.

```shell
$ python vectoraizer.py <weights file> <input image> [output image] 
```

If an output image file name is not provided, it will be composed starting from the input file name.

IE

```shell
$ python vectoraizer.py ./weights.h5 ./image.jpg
```

Will output an `image.svg` file in the CWD.
