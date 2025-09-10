# scTOP Tutorial

## Setup

### Open Jupyter Notebook
1. If you don't have access to a workspace in the SCC, ask someone in the Bioinformatics Core
2. Go to scc-ondemand2.bu.edu/pun/sys/dashboard/batch_connect/sessions
3. Click Jupyter Notebook
4. Enter these parameters: python3 for List of modules to load, lab for Interface, and the folder and project you are assigned to work in for Working Directory and Project
5. Click Launch, then Connect to Jupyter after waiting a little for it to set up

### Prepare Coding Packages (One Time Only)
I have set up a conda environment with all the libraries you should need to run these analyses.
1. Click File -> New -> Terminal
2. In the terminal, run `module unload python3`
3. Run `module load miniconda`
4. Run `mamba env create -f /restricted/projectnb/crem-trainees/Kotton_Lab/Vilker_Helper_Files/scTOP/scTOP.yml`
5. Click File -> New -> Notebook
6. Click Select Kernel in the top right and select scTOP
7. Go to the first vignette in the scTOP folder
8. Copy and paste the contents of the first cell (with # Load libraries in Line 1) into the top cell of your file
9. Press Control + Enter to run the cell

## Core Functionality
To simplify usage of scTOP, I have created the TopObject class to store common functions and variables
1. To initialize, run `[objectName] = TopObject.TopObject([DatasetName])`, where DatasetName is a name in a file in DatasetInformation.csv. You can pass in `True` for skipProcess if using a dataset as a basis only, as well as lists of cell types for `keep` or `exclude` to filter the dataset. `True` can also be passed in for either `keep` or `exclude` to use a preset list of cell types.
2. To set a basis, run `[objectName].setBasis()`. You can then get the basis any time with [objectName].basis
3. To project onto a basis, run `[objectName].projectOntoBasis(basis, ["basisDescription"])`. The projection can then be found at `[objectName].projection["basisDescription"]`
4. To combine bases, run `[objectName].combineBases(otherBasis, name=["basisDescription"])`. Inlude firstKeep or secondKeep as parameters if you want to choose specific cell types in either basis to include. The combined basis can then be found at `[objectName].combinedBases["basisDescription"]`
5. To test a basis, run `[objectName].testBasis()`

## Plotting Functions
The following functions are useful for generating figures flexibly. There are many optional parameters you may wish to use; check out PlottingFunctions.md or the actual functions at SimilarityHelper.py and TopObject.py to see all the options available to you.
1. To view a dataset's projection against two cell types in the basis, run `SimilarityHelper.plotTwo([objectName].projections["[basisDescription]"], [objectName].annotations, [basisCelltype1], [basisCelltype2])`. Options include to view gene expressions or supervised or unsupervised contours.
2. To view how a projection changes over time, run `SimilarityHelper.plotTwoMultiple([objectName], [objectName].projections["[basisDescription]"], [basisCelltype1], [basisCelltype2])`. The same options as for plotTwo apply.
3. To view a boxplot of all the cell types in the source against all the cell types in the basis, run TODO:`SimilarityHelper.similai
