The script have to be used in the following order:

1. randomForest_CLEANING_TRAININGDATA:

  (neccessary input: Sentinel-2 data and training polygons)
  rasterizes the training polygons and creates clean training data pixels and saves it as a tif. 
  Only pixels that are covered by more than 80% of a polygon of one class get used, everything else will be set to NA.

2. randomForest_newIndices:

  (neccessary input: results from 1. randomForest_CLEANING_TRAININGDATA) 
   caculates a number of spectral indices and texture on basis of the Sentinel-2 images.
   Saves a multiband tiff with all the original Sentinel-2 bands, the indices and textures and the rasterized training data.

3. randomForest_Training

   (neccessary input: results from 2. randomForest_newIndices)
   trains a random Forest to classify the forest into "clear", "deadwood" and "undisturbed".
   Its a very flexibel script, which can do oversampling and undersampling,
   use different regions as an training and validation basis, use different bands and indices. 
