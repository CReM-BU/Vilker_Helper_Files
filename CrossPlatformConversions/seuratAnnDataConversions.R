library(Seurat)
library(reticulate)
library(sceasy)
library(SingleCellExperiment)
library(SeuratDisk)

### Note: The functions in this file require AnnData through a Python environment. Conda recommended

# Function to convert a modern Seurat object to AnnData.
convert_Rds_Seurat_to_h5ad <- function(conversionInput, mainLayer=NULL, transferLayers=NULL, assay="RNA"){
  if (is.null(transferLayers)){ transferLayers <- c("data", "counts", "scale.data") }
  if (is.null(mainLayer)){ mainLayer <- "counts" }
  
  sceasy::convertFormat(conversionInput$obj, from="seurat", to="anndata", outFile=paste0(conversionInput$outFile, ".h5ad"), 
                        main_layer=mainLayer, transfer_layers=transferLayers, assay=assay)
  return(1)
}

# Function to convert an Robj format Seurat object to AnnData. May need adjusting if there are complications with data vs scaled data, as SeuratDisk's convert function doesn't account for layers
convert_Robj_Seurat_to_h5ad <- function(conversionInput, outFile, assay="RNA"){  
  intermediateFile <- paste0(conversionInput$outFile, ".h5seurat")
  SaveH5Seurat(conversionInput$obj, filename=intermediateFile, overwrite = TRUE)
  return(convert_h5seurat_Seurat_to_h5ad(intermediateFile, assay=assay))
}

# Function to convert an h5seurat file to AnnData. Note that it will automatically use the same name and place, just a different suffix
convert_h5seurat_Seurat_to_h5ad <- function(conversionInput, assay="RNA"){
  Convert(conversionInput$inFile, dest="h5ad", overwrite = TRUE, assay=assay)
  return(1)
}

# Check if all necessary parameters are set and return the Seurat input object or file location
getSeuratInputIfValid <- function(type, inputFile=NULL, outputFile=NULL, seuratObj=NULL){
  if (type == "h5seurat"){
    if (is.null(inputFile)){
      cat("No input file provided")
      return(NULL)
    }
    return(list(inFile=inputFile))
  }
        
  else if (type == "Rds" || type == "Robj"){
    if (!is.null(inputFile)){ 
      seuratObj <- readRDS(inputFile) # Load Seurat object if file specified
      if (is.null(outputFile)) { outputFile <- inputFile }
    }
    else if (is.null(seuratObj) || is.null(outputFile)){ 
      cat("Must specify input file or both Seurat object and output file!") 
      return(NULL)
    }
    if (type == "Robj"){ seuratObj <- UpdateSeuratObject(seuratObj) }   # Get h5 version of Seurat object

    return(list(obj=seuratObj, outFile=getFilenameWithoutExtension(outputFile)))
  }
}

# Get a file's name sans part after the last period
getFilenameWithoutExtension <- function(filename){
  splitName <- strsplit(filename, split="[.]")
  unSplit <- unlist(splitName)
  if (length(unSplit) > 1){ return(paste(unSplit[-length(unSplit)], collapse=".")) }
  return(NULL)
}

# Main function to call the appropriate conversion function given the type of input format. 
## condaEnvironment: Set if you don't have a Python environment set up with AnnData or are struggling, or if you just have a conda environment with all the functions you want
## type: the file format, of which the options are Rds (case insensitive), h5seurat, and Robj
## inputFile: location of the Seurat object. Not needed if an object is provided, except for h5seurat, where the path must be used
## outputFile: where the new file will be generated. Not used if type == "h5seurat" as the file must be generated in place (and Robj conversion will do this as an intermediate step but deposit the result where you specify
## seuratObj: use instead of inputFile for Rds or Robj
## mainLayer: Which layer will be stored in adata.X. By default, counts
## transferLayers: Which layers should be transferred along with the main layers. By default, includes data and scale.data
## assay: Which assay type from which to extract the layers. By default RNA, but SCT and other types like spliced exist
# which typically will refer to how the data were transformed
convert_Seurat_to_AnnData <- function(condaEnvironment=NULL, type="Rds", inputFile=NULL, outputFile=NULL, seuratObj=NULL, mainLayer=NULL, transferLayers=NULL, assay="RNA"){

  # Initialize parameters
  if (type == "rds"){ type <- "Rds" }
  if (!is.null(condaEnvironment)){ reticulate::use_condaenv(condaenv=condaEnvironment, required = TRUE) }
  conversionInput <- getSeuratInputIfValid(type, inputFile=inputFile, outputFile=outputFile, seuratObj=seuratObj)
  if (is.null(conversionInput)){ return(0) }
    
  if (type == "Rds"){ success <- convert_Rds_Seurat_to_h5ad(conversionInput, mainLayer=mainLayer, transferLayers=transferLayers, assay=assay) }
  else if (type == "h5seurat"){ success <- convert_h5seurat_Seurat_to_h5ad(conversionInput, assay=assay) }
  else if (type == "Robj"){ success <- convert_Robj_Seurat_to_h5ad(conversionInput, assay=assay) }
  return(success)
}

# Function to convert an AnnData object (h5ad) to Seurat (rds)
convert_h5ad_to_Rds <- function(filename, outFile, condaEnvironment=NULL){
  if (!is.null(condaEnvironment)){ reticulate::use_condaenv(condaenv=condaEnvironment, required = TRUE) }
  sceasy::convertFormat(filename, from="anndata", to="seurat", outFile=outFile)
}


