library(Seurat)
library(reticulate)
library(sceasy)
library(SingleCellExperiment)
library(SeuratDisk)

### Note: The functions in this file require a conda environment with versions of Seurat you won't get installing through base R

# Function to convert a modern Seurat object to AnnData. You may need to adjust which data are sent over with transfer_layers and main_layer
convert_Rds_Seurat_to_h5ad <- function(out_file, conda_environment_path, main_layer, transfer_layers, input_file=NULL, obj=NULL, assay="RNA"){
  # Set Seurat input
  if (!is.null(input_file)){ obj <- readRDS(filename) }
  else if (is.null(obj)){ 
    cat("No Seurat input provided!") 
    return(0)
  }
  
  # Perform conversion
  reticulate::use_condaenv(condaenv=conda_environment_path, required = TRUE)
  sceasy::convertFormat(obj, from = "seurat", to = "anndata", outFile = out_file, main_layer=main_layer, transfer_layers=transfer_layers, assay=assay)
  return(1)
}

# Function to convert an Robj format Seurat object to AnnData. May need adjusting if there are complications with layers, but ideally you don't need to use this.
convert_Robj_Seurat_to_h5ad <- function(filename, out_file, conda_environment_path, main_layer, transfer_layers){
  # This section is just so that the intermediate h5seurat file can then be opened easily with LoadH5Seurat and converted with the right name
  splitOutfile = strsplit(out_file, split="[.]")
  unSplit <- unlist(split)
  if(length(unSplit) > 1){
    out_file <- paste0(unSplit[-length(unSplit)], collapse="")
  }
  
  # Get h5 version of Seurat object
  reticulate::use_condaenv(condaenv=conda_environment_path, required = TRUE)
  obj <- readRDS(filename)
  obj <- UpdateSeuratObject(obj)
  
  # Save file and convert to AnnData
  intermediateFile <- paste0(out_file, ".h5seurat", overwrite = TRUE)
  SaveH5Seurat(obj, filename = intermediateFile)
  return(convert_h5seurat_Seurat_to_h5ad(intermediateFile, main_layer, transfer_layers))
}

# Function to convert an h5seurat file to AnnData. Note that it will automatically use the same name and place, just a different suffix
convert_h5seurat_Seurat_to_h5ad <- function(filename, main_layer, transfer_layers, conda_environment_path=NULL){
  if (!is.null(conda_environment_path)){ reticulate::use_condaenv(condaenv=conda_environment_path, required = TRUE) }
  Convert(filename, dest = "h5ad", overwrite = TRUE, main_layer=main_layer, transfer_layers=transfer_layers)
  return(1)
}

# Main function to call the appropriate conversion function given the type of input format
convert_Seurat_to_AnnData <- function(conda_environment_path, type="Rds", out_file=NULL, input_file=NULL, obj=NULL, main_layer=NULL, transfer_layers=NULL, assay="RNA"){
  # Set expression data types
  if (is.null(transfer_layers)){ transfer_layers <- c("data", "counts", "scale.data") }
  if (is.null(main_layer)){ main_layer <- "counts" }
  
  if (type == "Rds"){
    if (is.null(out_file)){ 
      cat("Provide output file")
      return(0)
    }
    success <- convert_Rds_Seurat_to_h5ad(out_file, conda_environment_path, main_layer, transfer_layers, obj=obj, input_file=input_file, assay=assay)
  }
  else if (type == "h5seurat"){
    if (is.null(input_file)){ 
      cat("Provide input file")
      return(0)
    }
    success <- convert_h5seurat_Seurat_to_h5ad(input_file, main_layer, transfer_layers, conda_environment_path=conda_environment_path)
  }
  else if (type == "Robj"){
    if (is.null(out_file)){ 
      cat("Provide output file")
      return(0)
    }
    success <- convert_Robj_Seurat_to_h5ad(input_file, out_file, conda_environment_path, main_layer, transfer_layers)
  }
  return(success)
}

# Function to convert an AnnData object to Seurat
convert_h5ad_to_Rds <- function(filename, out_file, conda_environment_path){
  reticulate::use_condaenv(condaenv=conda_environment_path, required = TRUE)
  sceasy::convertFormat(filename, from = "anndata", to = "seurat", outFile = out_file)
}

### Examples

## Choose conda environment (use ls -a in bash to see hidden directories and files, such as .conda)
condaEnvironment <- "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/.conda/envs/KottonLab"

## Seurat to AnnData
# convert_Seurat_to_AnnData("", "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/BibekPneumonectomy/objects/Bibek.h5ad", condaEnvironment, obj=seuratObject)
kathiriyaInput <- "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Kathiriya/Kathiriya.h5seurat"
kathiriyaOutput <- "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Kathiriya/Kathiriya.h5ad"
# success <- convert_Seurat_to_AnnData(condaEnvironment, type="h5seurat", input_file=kathiriyaInput)
success <- convert_Seurat_to_AnnData(condaEnvironment, type="Rds", out_file=kathiriyaOutput, obj=kathiriya, assay="spliced")

## AnnData to Seurat
inputFile <- "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/OutsidePaperObjects/KathiriyaAnnData.h5ad"
outputFile <- "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/OutsidePaperObjects/KathiriyaSeurat.rds"
convert_h5ad_to_Rds(inputFile, outputFile, condaEnvironment)


