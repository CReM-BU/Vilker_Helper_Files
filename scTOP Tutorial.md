# scTOP Tutorial

## Setup

### Open Jupyter Notebook
1. If you don't have access to a workspace in the SCC, ask someone in the Bioinformatics Core
2. Go to scc-ondemand2.bu.edu/pun/sys/dashboard/batch_connect/sessions
3. Click Jupyter Notebook
4. Enter these parameters: python3 for List of modules to load, lab for Interface, and the folder and project you are assigned to work in for Working Directory and Project
5. Click Launch, then Connect to Jupyter after waiting a little for it to set up

### Prepare Coding Packages
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
