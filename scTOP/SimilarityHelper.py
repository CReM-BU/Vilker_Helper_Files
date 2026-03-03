### File containing functions for working with scTOP, made by Maria Yampolskaya, Huan Souza, and Pankaj Mehta
# Author: Eitan Vilker

import sys
sys.path.append('/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Vilker_Helper_Files/scTOP')
import TopObject
import sctop as top
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import math
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import textwrap
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import os
os.environ['SCIPY_ARRAY_API'] = '1'
from imblearn.under_sampling import RandomUnderSampler
# import hdf5plugin


# Function to load a basis given a file location or basis name and summary csv
def loadBasis(file=None, basisCollection=None, basisName=None, geneIndex="gene", basisKeep=None):
    if file is None:
        if basisCollection is None or basisName is None:
            print("Must enter either a filename or a file containing multiple bases and the name of the basis you want")
            return None
        file = pd.read_csv(basisCollection, index_col="Name").loc[basisName, "File"]

    if file.endswith("h5"):
        with h5py.File(file, "r") as f:
            cellTypes = f["df"]["axis0"][:]
            var = f["df"]["axis1"][:]
            X = f["df"]["block0_values"][:]

        basis = pd.DataFrame(X)
        basis.columns = [col.decode() for col in cellTypes]
        basis.index = [row.decode() for row in var]

    elif file.endswith("csv"):
        basis = pd.read_csv(file, index_col=geneIndex)
    else:
        print("Unsupported file type")
        return None

    # Reduce wordiness in basis names
    newCols = {}
    for col in basis.columns:
        idx = col.find("cell")
        if idx != -1:
            newCols[col] = col[:idx - 1]

    basis = basis.rename(columns=newCols)

    if basisKeep is not None:
        basis = basis[[colName for colName in basis.columns if colName in basisKeep]]
    print("Loaded " + basisName + " basis!")
    return basis


# Converts files in raw format (straight from GEO usually) to AnnData. Set geneHeader to None if no header
def rawToAnnData(countsPath, genesPath, metadataPath,
                 matrix=False, transposeCounts=False, geneSeparator="\t",
                 metadataSeparator="\t", metadataIndexColumn=None, geneHeader="infer", geneColumnIdx=0, skipMeadataRow=None, skipCountsRow=None):
    # Set counts
    print("Setting counts...")
    counts = sc.read_mtx(countsPath) if matrix else ad.AnnData(pd.read_csv(countsPath))
    try:
        annObject = counts.T if transposeCounts else counts
    except:
        print("You may need to set transposeCounts to True")
        return None

    # Set metadata
    if metadataPath is not None:
        try:
            print("Setting metadata...")
            metadata = pd.read_csv(metadataPath, sep=metadataSeparator) if skipRow is None else pd.read_csv(metadataPath, sep=metadataSeparator, skiprows=[skipRow])
            metadata.replace(np.nan, '', inplace=True)
            metadata.set_index(metadataIndexColumn or metadata.columns[0], inplace=True)
            annObject.obs = metadata
            annObject.obs.index.names = ["index"]
        except:
            print("Something went wrong setting metadata!")
            return annObject

    # Set genes
    try:
        print("Setting genes...")
        genes = pd.read_csv(genesPath, sep=geneSeparator, header=geneHeader)
        # genes.columns[geneColumnIdx] = ["name"]
        genes.rename(columns={genes.columns[geneColumnIdx]: "name"}, inplace=True)
        genes.set_index("name", inplace=True)
        annObject.var = genes
        annObject.var.index.names = ["index"]
    except:
        print("Something went wrong setting genes!")
        return annObject

    print("Done!")
    return annObject


# Write an AnnData object to an h5ad file
def writeAnnData(annDataObj, outFile, indexReplace=None):

    # Copy and compress data as sparse matrix
    obj = annDataObj.copy()
    print("Setting X as csr_matrix...")
    obj.X = csr_matrix(obj.X)

    # If object has raw data in raw layer, can adjust names here so old formats don't break
    if obj.raw is not None:
        print("Setting raw...")
        if indexReplace is not None:
            obj._raw._var.rename(columns={indexReplace: 'index'}, inplace=True)
            obj.raw.var.index.name(columns={indexReplace: 'index'}, inplace=True)

    # Write the object
    print("Writing h5ad...")
    obj.write_h5ad(outFile)
    del obj
    print("Finished!")


# Get average projections given time series data
def getTimeAveragedProjections(basis, df, cellLabels, times, timeSortFunc, substituteMap=None):
    projections = {}
    processed = {}
    timesSorted = sorted([str(time) for time in set(times)], key=timeSortFunc)

    for time in tqdm(timesSorted):
        types = pd.DataFrame(cellLabels.value_counts())
        for current_type in list(types.index):
            current_processed = top.process(df.loc[:, np.logical_and(times==time, cellLabels == current_type)], average=True)
            processed[current_type] = current_processed
            current_scores = top.score(basis, current_processed)
            if substituteMap is not None:
                projectionKey = substituteMap[current_type] + "_" + time
            else:
                projectionKey = current_type + "_" + time
            projections[projectionKey] = current_scores

    return projections


# Get a dict of cell types in basis to the similarity scores of each type in the source. 
# Format: Key (basis label) -> Key (source label) -> Value (projection score list for cells with source label onto the basis label)
def getMatchingProjections(topObject, projectionName, basisKeep=None, testKeep=None, includeCriteria=None, prefix=None, alternateAnnotations=None):

    # If no specific columns provided, use all columns
    projections = topObject.projections[projectionName]
    if includeCriteria is not None:
        projections = projections.loc[:, includeCriteria]
    if basisKeep is None:
        basisKeep = sorted(projections.index)
    if testKeep is None:
        testKeep = sorted(set(topObject.annotations if alternateAnnotations is None else alternateAnnotations))

    # Initialize map of basis labels
    similarityMap = {}
    for label in basisKeep:
        similarityMap[label] = {}

    # Broaden map to go from basis labels to source labels (with option to include prefix usually to indicate name of source)
    for trueLabel in testKeep:
        for label in similarityMap:
            adjustedTrueLabel = prefix + trueLabel if prefix else trueLabel
            similarityMap[label][adjustedTrueLabel] = []

    # For every sample, add its projection scores for each label in the basis, but only for the label assigned in the source
    for sampleId, sampleProjections in projections.items():
        trueLabel = topObject.metadata.loc[sampleId, topObject.cellTypeColumn]
        if trueLabel in testKeep:
            projectionTypes = sampleProjections.index
            for label in basisKeep:
                labelIndex = projectionTypes.get_loc(label)
                similarityScore = sampleProjections.iloc[labelIndex]
                adjustedTrueLabel = prefix + trueLabel if prefix else trueLabel
                similarityMap[label][adjustedTrueLabel].append(similarityScore)

    print("Similarity map built!")
    return similarityMap


# Undersample cell types based on a count maximum
def underSample(df, annotations, maxCount, seed=1):
    valueCounts = annotations.value_counts()
    labelCountsMap = {label: int(valueCounts[label]) if int(valueCounts[label]) < maxCount else maxCount for label in set(annotations)}
    rus = RandomUnderSampler(sampling_strategy=labelCountsMap, random_state=seed)
    dfUntransposed, annotations = rus.fit_resample(df.T, annotations)
    return dfUntransposed.T, annotations


# Takes as input a geneMap which maps genes against directions of expression in the expected cluster, adds cutoffs required for significance
def setDEGCluster(topObject, geneMap, minSD=1):
    for gene in geneMap:
        geneMap[gene] = {}
        geneMap[gene]["Mean"] = topObject.processed.loc[gene].T.mean()
        geneMap[gene]["SD"] = topObject.processed.loc[gene].T.std()
        geneMap[gene]["Threshold"] = geneMap[gene]["Mean"] + geneMap[gene]["SD"] * minSD * geneMap[gene]["Direction"]
    return geneMap


# WIP to get how high variance a cluster is
def getDEGClusterValues(topObject, geneMap, includeCriteria=None):
    geneExpressions = topObject.processed.loc[list(geneMap.keys()), :]
    geneExpressions = geneExpressions.loc[:, includeCriteria] if includeCriteria else geneExpressions
    sampleValues = []
    for sample in geneExpressions.columns:
        totalHits = 0
        for gene in geneMap.keys():
            hit = geneExpressions[gene, sample] > geneMap[gene]["Threshold"] if geneMap[gene]["Direction"] == 1 else geneExpressions[gene, sample] < geneMap[gene]["Threshold"]
            if hit:
                totalHits += 1
        sampleValues.append(totalHits)
    return sampleValues


# Set criteria for displaying data when overlaying different values from the same category of a dataset
def setupIncludeCriteria(topObject, columnName, acceptedValues, additionalCriteriaAll=None, additionalCriteriaOthers=None):
    includeCriteriaList = []
    for i in range(len(acceptedValues)):
        criterion = topObject.metadata[columnName] == acceptedValues[i]
        criterion = np.logical_and(criterion, additionalCriteriaAll) if additionalCriteriaAll is not None else criterion
        criterion = np.logical_and(criterion, additionalCriteriaOthers) if i != 0 and additionalCriteriaOthers is not None else criterion
        includeCriteriaList.append(criterion)
    return includeCriteriaList


# Get separate projections, annotations, etc. for purpose of overlaying different conditions
def setupOverlay(topObjects, basisName, includeCriteriaList, basis=None, forceProject=False, filterShared=False, getExpressions=False, getAlternates=False):

    # Initialize structures
    multipleTopObjects = len(topObjects) > 1
    otherProjections = []
    otherAnnotations = []
    otherExpressions = [] if getExpressions else None
    otherNames = [] if multipleTopObjects else None
    toReturn = []

    # If processing just off genes common to each datasets
    if filterShared:
        unsharedGenes = {gene for topObject in topObjects[1:] for gene in topObjects[0].df.index if gene not in topObject.df.index}
        sharedGenes = [gene for gene in topObjects[0].df.index if gene not in unsharedGenes]

    # For each projection to overlay
    for i in range(len(includeCriteriaList)):
        topObject = topObjects[i] if multipleTopObjects else topObjects[0]
        
        # Filter to common genes and perform projection if not done already or forced
        if filterShared:
            topObject.setAnndata(topObject.anndata[:, topObject.df.index.isin(sharedGenes)])
        projection = topObject.project(basis, basisName) if forceProject or basisName not in topObject.projections.keys() else topObject.projections[basisName]

        # Get data for everything but the first set, which is handled by the plotTwo function normally
        if i != 0:
            otherProjections.append(projection.loc[:, includeCriteriaList[i]] if includeCriteriaList[i] is not None else projection)
            otherAnnotations.append(topObject.annotations[includeCriteriaList[i]] if includeCriteriaList[i] is not None else topObject.annotations)
            if getExpressions:
                otherExpressions.append(topObject.processed[includeCriteriaList[i]] if includeCriteriaList[i] is not None else topObject.processed)
        if multipleTopObjects:
            otherNames.append(topObject.name)

    # Arrange return tuple
    toReturn = [otherProjections, otherAnnotations]
    if multipleTopObjects:
        toReturn.append(otherNames)
    if getExpressions:
        toReturn.append(otherExpressions)
    return tuple(toReturn)


## ========================= ##
## Define plotting functions ##
## ========================= ##

# Helper function for creating a color bar
def createColorbar(data, colormap='rocket_r'):
    cmap = plt.get_cmap(colormap)
    scalarmap = cm.ScalarMappable(norm=plt.Normalize(min(data), max(data)),
                               cmap=cmap)
    scalarmap.set_array([])
    return cmap, scalarmap


# Helper function to set correspondonce between labels and colors
def setPalette(labels, source="seaborn"):
    palette = {}
    if source == "seaborn":
        colors = list(sns.color_palette()) + list(sns.color_palette("bright"))
    elif source == "ggplotDiscrete":
        colors = ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3"]
        # colors = ["#F8766D", "#7CAE00", "#00BE67", "#00BFC4", "#C77CFF", "#FF61C3", "#E68613", "#B79F00", "#00BA38", "#00A9FF", "#619CFF", "#F564E3"]
    elif source == "ggplot2":
        colors = ["#F8766D", "#B79F00", "#00BA38", "#00BFC4", "#619CFF", "#F564E3", "#00C08B", "#00B0F6", "#9590FF", "#E76BF3", "#FF62BC", "#FF6A98"]
    elif source == "matplotlib":
        colors = sns.color_palette("tab10")
    elif source == "lauren":
        colors = ["#6E8B3D", "#EEAD0E", "#CD3333", "#008B8B", "#009ACD"]
    elif source == "PRC2":
        colors = ["#1f77b4", "#ff800f", "#279257", "#da3d3e", "#b696d2", "#964B00"]
    else:
        print("Enter a valid palette source!")
        return None
    for i in range(len(labels)):
        palette[labels[i]] = colors[i]
    return palette


# Helper function to set correspondonce between labels and markers
def setMarkers(labels):
    markers = {}
    # markerList = list(Line2D.markers.keys())
    markerList = ["o", "^", "s", "d", "p", "*", "X", "<", ">", "v", "H", "h", "D", ".", ",", "1", "2", "3", "4"]
    for i in range(len(labels)):
        markers[labels[i]] = markerList[i]
    return markers


# Create scatter plot showing projections of each cell in a UMAP plot, for a given cell type
def plot_UMAP(projections, embedding, cell_type, ax=None, **kwargs):
    ax = ax or plt.gca()
    type_projections = np.array(projections.loc[cell_type]).T
    palette, scalarmap = createColorbar(type_projections)
    plt.colorbar(scalarmap, label='Projection onto {}'.format(cell_type), ax=ax)
    plot = sns.scatterplot(x = embedding[:,0],
                           y = embedding[:,1],
                           hue = type_projections,
                           palette = palette,
                           alpha = 0.5,
                           ax = ax,
                           **kwargs
                          )
    plot.legend_.remove()


# Create scatter plot showing top projection types for each cell
def plot_top(projections, tSNE_data, minimum_cells=50, ax=None, **kwargs):
    ax = ax or plt.gca()
    top_types = projections.idxmax().values
    unique_types = np.unique(top_types, return_counts=True)
    other_types = []

    for i, count in enumerate(unique_types[1]):
        if count < minimum_cells:
            other_types += [unique_types[0][i]]

    for i, cell_type in enumerate(top_types):
        if cell_type in other_types:
            top_types[i] = "Other"
    print(len(top_types))
    sns.scatterplot(x = tSNE_data[:,0],
                           y = tSNE_data[:,1],
                           hue = top_types,
                           alpha = 0.5,
                           ax = ax,
                           **kwargs
                    )


# Create scatter plot showing projection scores for two cell types, with the option to color according to marker gene
def plotTwo(topObject, projectionName, celltype1, celltype2, 
       additionalProjections=[], additionalAnnotations=[], additionalExpressions=[], additionalNames=[], additionalAlternates=[],
       gene=None, geneExpressions=None, plotMultiple=False, singleColorbar=False, unsupervisedContour=False, supervisedContour=False, maxLabelCount=None, 
       alternateColumn=None, alternateAnnotations=None, alternateProjections=None, alternateExpressions=None,
       ax=None, figX=8, figY=8, DPI=100, includeCriteria=None, name=None, hue=None, labels=None, palette=None, alpha=0.5, source="seaborn", 
       markers=None, markerSize=40, legendMarkerScale=2, legendWidth=116.625, lineWidth=1.5, legendSpacing=0.05, xBounds=None, yBounds=None, plotInRow=False, plotThreshold=True, seed=0,
       axisFontSize=15, legendFontSize=13, legendTitle="Test Set Cell Labels", title="", outFile=None, show=True):

    # Set data elements
    annotations = topObject.annotations if alternateAnnotations is None else alternateAnnotations
    projections = topObject.projections[projectionName] if alternateProjections is None else alternateProjections
    geneExpressions = None if not gene else topObject.processed if alternateExpressions is None else alternateExpressions
    annotations = topObject.annotations if alternateAnnotations is None else alternateAnnotations
    alternateColumnValues = None if not alternateColumn else topObject.metadata[alternateColumn]
    name = topObject.name if name is None else name

    # Filter dataset
    if includeCriteria is not None:
        annotations = annotations[includeCriteria]
        projections = projections.loc[:, includeCriteria]    
        geneExpressions = None if not gene else geneExpressions.loc[:, includeCriteria]
        alternateColumnValues = None if not alternateColumn else alternateColumnValues[includeCriteria]
    
    # If overlaying multiple sets of projections, combine along shared genes (ideally filter shared genes before projecting to reduce bias)
    if len(additionalProjections) > 0:
        projections = pd.concat(additionalProjections + [projections], axis=1, join='inner')
        annotations = additionalAnnotations + [annotations]
        alternateColumnValues = additionalAlternates + [alternateColumnValues] if alternateColumn is not None else None
        combinedNames = additionalNames + [name]
        combinedAnnotations = []
        combinedAlternates = [] if alternateColumn is not None else None
        for i in range(len(combinedNames)):
            combinedAnnotations.append(annotations[i].apply(lambda annotation: combinedNames[i] + ' ' + annotation))
            if combinedAlternates is not None:
                combinedAlternates.append(alternateColumnValues[i].apply(lambda alternate: combinedNames[i] + ' ' + alternate))
        annotations = pd.concat(combinedAnnotations)
        alternateColumn = pd.concat(combinedAlternates) if alternateColumn is not None else None
        geneExpressions = None if not gene else pd.concat(additionalExpressions + [geneExpressions], axis=1, join='inner')

    # Undersample cell types based on a count maximum
    projections, annotations = (projections, annotations) if maxLabelCount is None else underSample(projections, annotations, maxLabelCount, seed=seed)

    # Set axes and key parameters for current plot
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['axes.linewidth'] = lineWidth

    legendSpace = 0 if gene or alternateColumn else legendWidth / plt.rcParams['figure.dpi'] + legendSpacing
    fig, ax = plt.subplots(1, 1, figsize=(figX + legendSpace, figY)) if ax is None else (None, ax)
    if gene:
        order = geneExpressions.loc[gene].T.sort_values(ascending=True).index
        projections = projections.loc[:, order]
        annotations = annotations.loc[order]
        geneExpressions = geneExpressions.loc[:, order]
    x, y = (projections.loc[celltype1], projections.loc[celltype2])
    labels = sorted(annotations.unique()) if labels is None else labels
    palette = setPalette(labels, source=source) if palette is None and gene is None else palette
    markers = setMarkers(labels) if markers is None else markers
    legendTitle = "" if legendTitle is None else legendTitle

    # Create core plots
    if gene:  # If labeling by gene expression instead of source labels
        ax = geneExpressionPlot(ax, x, y, gene, geneExpressions, annotations, palette,
                labels=labels, markers=markers, markerSize=markerSize, alpha=alpha, axisFontSize=axisFontSize, singleColorbar=singleColorbar)
    elif alternateColumn:  # If labeling by a specifc other column, such as disease type
        ax = alternateColumnPlot(ax, x, y, alternateColumn, alternateColumnValues, annotations, palette, 
                labels=labels, markers=markers, markerSize=markerSize, alpha=alpha, axisFontSize=axisFontSize, singleColorbar=singleColorbar)
    else:  # If labeling is by cell type
        ax = testLabelPlot(ax, x, y, annotations, palette, 
                labels=labels, title=legendTitle, markers=markers, markerSize=markerSize, alpha=alpha, legendMarkerScale=legendMarkerScale, axisFontSize=axisFontSize, legendFontSize=legendFontSize, plotMultiple=plotMultiple, legendSpace=legendSpace)
        # ax.set_aspect('equal')

    # Add contours if desired
    ax = unsupervisedContourPlot(ax, x, y) if unsupervisedContour else ax
    ax = supervisedContourPlot(ax, x, y, annotations, labels, palette=palette) if supervisedContour else ax

    # Set plot's visual tools
    ax.tick_params(axis='both', which='major', labelsize=axisFontSize // 1.4)
    if plotThreshold:
        ax.axvline(x=0.1, color='black', linestyle='--', linewidth=lineWidth / 3, dashes=(5, 10))
        ax.axhline(y=0.1, color='black', linestyle='--', linewidth=lineWidth / 3, dashes=(5, 10))

    # Set plot dimensions
    if xBounds is not None:
        ax.set_xlim(xBounds[0], xBounds[1])
    if yBounds is not None:
        ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_xlabel(projectionName + " " + celltype1 + " Cell Score", fontsize=axisFontSize)
    ax.set_ylabel(projectionName + " " + celltype2 + " Cell Score", fontsize=axisFontSize)
    # ax.set_xlabel(celltype1 + " Cell Score", fontsize=axisFontSize)
    # ax.set_ylabel(celltype2 + " Cell Score", fontsize=axisFontSize)

    # Set text for plot
    if title == "" and show and len(additionalProjections) == 0:
        title = topObject.name + " Projected Onto " + projectionName + " Reference"
    if title is not None:
        plt.title(title, fontsize=axisFontSize // 0.9)

    if plotMultiple:
        # ax.set_aspect('equal')
        pass
    else:
        ax.set_box_aspect(1)
        plt.tight_layout()
        if outFile is not None:
            plt.savefig(outFile, bbox_inches='tight', dpi=DPI)
        if show:
            plt.show()
    return ax


# Creates a Seaborn 2D scatter plot using projections onto basis columns as axes and gene expressions to identify points. Helper for plotTwo
def geneExpressionPlot(ax, x, y, gene, geneExpressions, annotations, palette,
                       labels=None, markers=True, markerSize=40, axisFontSize=16, alpha=0.5, plotMultiple=False, singleColorbar=False):
    palette, scalarmap = createColorbar(geneExpressions.loc[gene]) if palette is None else (palette, None)
    plot = sns.scatterplot(x=x, y=y, ax=ax, hue=geneExpressions.loc[gene], style=annotations, style_order=labels, markers=markers, s=markerSize, palette=palette, alpha=alpha, linewidth=0.15)
    if not singleColorbar:
        cbar = plt.colorbar(scalarmap, ax=ax, fraction=0.044) # scalarmap won't be defined unless palette wasn't, which is only in single plot
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        # cbar.set_label('{} expression'.format(gene), size=axisFontSize, labelpad=20)
        cbar.set_label('{} expression'.format(gene), size=axisFontSize)

    plot.legend_.remove()
    return ax


# Creates a Seaborn 2D scatter plot using projections onto basis columns as axes and a specified column's values to identify points. Helper for plotTwo
def alternateColumnPlot(ax, x, y, alternateColumn, alternateColumnValues, annotations, palette,
                       labels=None, markers=True, markerSize=40, axisFontSize=16, alpha=0.5, plotMultiple=False, singleColorbar=False):
    palette, scalarmap = createColorbar(alternateColumnValues) if palette is None else (palette, None)
    plot = sns.scatterplot(x=x, y=y, ax=ax, hue=alternateColumnValues, style=annotations, style_order=labels, markers=markers, s=markerSize, palette=palette, alpha=alpha, linewidth=0.15)

    if not singleColorbar:
        cbar = plt.colorbar(scalarmap, ax=ax) # scalarmap won't be defined unless palette wasn't, which is only in single plot
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        cbar.set_label('{}'.format(alternateColumn), size=axisFontSize, labelpad=20)

    plot.legend_.remove()
    return ax


# Creates a Seaborn 2D scatterplot using projections onto basis columns as axes and source labels to identify points
def testLabelPlot(ax, x, y, annotations, palette, title="", labels=None, markers=True, markerSize=40, legendMarkerScale=2, axisFontSize=16, legendFontSize=16, alpha=0.5, plotMultiple=False, legendSpace=1):
    plot = sns.scatterplot(x=x, y=y, ax=ax, hue=annotations, style=annotations, hue_order=labels, style_order=labels, markers=markers, s=markerSize, palette=palette, alpha=alpha, linewidth=0.15)
    if plotMultiple:
        plot.legend_.remove()
    else:
        leg = ax.legend(title=title, title_fontsize=legendFontSize // 0.9, fontsize=legendFontSize, markerscale=legendMarkerScale, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.subplots_adjust(right=legendSpace)
    return ax


# Creates contour ellipses in unsupervised manner
def unsupervisedContourPlot(ax, x, y, contourColor=sns.color_palette()[0]):
    sns.kdeplot(ax=ax, x=x, y=y, fill=True, color=contourColor, alpha=0.2)
    sns.kdeplot(ax=ax, x=x, y=y, color=contourColor)
    return ax


# Creates contour ellipses in supervised manner
def supervisedContourPlot(ax, x, y, annotations, labels, palette=None):
    palette = palette or setPalette(labels)

    for label in labels:
        xLabel = x[annotations == label]
        yLabel = y[annotations == label]
        sns.kdeplot(ax=ax, x=xLabel, y=yLabel, fill=True, color=palette[label], alpha=0.3, label=label, thresh=0.1)
        sns.kdeplot(ax=ax, x=xLabel, y=yLabel, color=palette[label], label=label, thresh=0.1)

    return ax


# Plot multiple 2D similarity plots at once based on some field, such as time (Note: figure out difference if any between labels and annotations[toInclude])
def plotTwoMultiple(topObject, projectionName, celltype1, celltype2,
                    projections=None, subsetCategory=None, subsetNames=None, gene=None, includeCriteria=None, singleColorbar=True,
                    unsupervisedContour=False, supervisedContour=False, maxLabelCount=None, alternateColumn=None, seed=None,
                    xBounds=None, yBounds=None, plotInRow=False, axisFontSize=24, legendFontSize=24, labels=None, alpha=0.5, DPI=100, source="seaborn",
                    legendMarkerScale=0.5, markerSize=40, titleFontSize=36, title="", caption=None, outFile=None):

    # Initialize categories
    includeCriteria = includeCriteria if includeCriteria is not None else ~topObject.annotations.isin([])
    annotations = topObject.annotations if includeCriteria is None else topObject.annotations[includeCriteria]
    projections = topObject.projections[projectionName]
    projections = projections if includeCriteria is None else projections.loc[:, includeCriteria]
    x, y = (projections.loc[celltype1], projections.loc[celltype2])
    xBounds = (math.floor(x.min() * 10) / 10, math.ceil(x.max() * 10) / 10) if xBounds is None else xBounds
    yBounds = (math.floor(y.min() * 10) / 10, math.ceil(y.max() * 10) / 10) if yBounds is None else yBounds
    subsetCategory = topObject.metadata[topObject.timeColumn] if subsetCategory is None else subsetCategory
    subsetNames = topObject.timesSorted if subsetNames is None else subsetNames

    # Get subplots
    subsetCount = len(subsetNames)
    dimX = subsetCount if plotInRow else math.ceil(math.sqrt(subsetCount))
    dimY = 1 if plotInRow else math.ceil(subsetCount / dimX)
    fig = plt.figure(figsize=(dimX * 12, dimY * 12), constrained_layout=True)
    gs = GridSpec(dimY, dimX, figure=fig)
    availableSpots = dimX * dimY

    # Set up label colors and shapes
    labels = sorted(annotations.unique()) if labels is None else labels
    labelMarkerMap = setMarkers(labels)
    if gene:
        geneExpressions = topObject.processed
        palette, scalarmap = createColorbar(geneExpressions.loc[gene, includeCriteria])
    elif alternateColumn:
        palette, scalarmap = createColorbar(topObject.metadata[alternateColumn][includeCriteria])
    else:
        palette = setPalette(labels, source=source)
        legendItems = []
        for label in labels:
            legendItems.append(Line2D([0], [0], marker=labelMarkerMap[label], color="w", label=label,
               markerfacecolor=palette[label], markersize=markerSize))

    # Plot for each subset
    axs = [fig.add_subplot(gs[i, j]) for i in range(dimY) for j in range(dimX) if dimX * i + j < subsetCount]
    for i in range(subsetCount):
        subset = subsetNames[i]
        # toInclude = np.logical_and(subsetCategory == subset, includeCriteria) if includeCriteria is not None else subsetCategory == subset
        ax = plotTwo(topObject, projectionName, celltype1, celltype2, alternateProjections=projections,
            ax=axs[i], labels=labels, gene=gene, geneExpressions=geneExpressions if gene else None, alternateColumn=alternateColumn, 
            unsupervisedContour=unsupervisedContour, supervisedContour=supervisedContour, plotMultiple=True, singleColorbar=singleColorbar, 
            plotInRow=plotInRow, maxLabelCount=maxLabelCount, xBounds=xBounds, yBounds=yBounds, seed=seed, includeCriteria=subsetCategory == subset,
            palette=palette, alpha=alpha, markers=labelMarkerMap, markerSize=markerSize, lineWidth=2.5, legendMarkerScale=legendMarkerScale, axisFontSize=axisFontSize, legendFontSize=legendFontSize
        )
        if not plotInRow:
            ax.set_title(subset, fontsize=axisFontSize)

    # Add colorbar to end if appropriate
    if gene:
        cbar = fig.colorbar(scalarmap, label='{} expression'.format(gene), ax=ax)
        cbar.ax.tick_params(labelsize=legendFontSize // 1.3)
        cbar.set_label('{} expression'.format(gene), size=legendFontSize, labelpad=20)
    elif alternateColumn:
        cbar = fig.colorbar(scalarmap, label='{}'.format(alternateColumn), ax=ax)
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        cbar.set_label('{}'.format(alternateColumn), size=axisFontSize, labelpad=20)
    else:
        # Place legend where space available
        if subsetCount < availableSpots:
            ax = fig.add_subplot(gs[dimY - 1, dimX - (availableSpots - subsetCount)])
            ax.axis("off")
            ax.legend(handles=legendItems, title=topObject.name + " Labels", title_fontsize=axisFontSize, fontsize=legendFontSize, markerscale=legendMarkerScale, loc='upper left', frameon=False)
        else:
            fig.legend(legendItems, labels, loc="upper left", bbox_to_anchor=(1.025, 1), title=topObject.name + " Labels",  title_fontsize=axisFontSize,  fontsize=legendFontSize,  markerscale=legendMarkerScale, borderaxespad=0., frameon=True)

    # Add text and display/save
    if caption:
        caption = (
            caption
        )
        wrappedCaption = "\n".join(textwrap.wrap(caption, width=170))
        fig.text(0, -0.05, wrappedCaption, ha='left', va='bottom', fontsize=axisFontSize // 1.1)
    fig.suptitle(title, fontsize=titleFontSize)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=DPI)
    plt.show()


# Make 2D similarity plot of each gene in a selected list
def plotMultipleGenes(topObject, projectionName, celltype1, celltype2, geneList,
                     includeCriteria=None, xBounds=None, yBounds=None, maxLabelCount=None, 
                     legendMarkerScale=0.5, markerSize=40, axisFontSize=24, legendFontSize=24, titleFontSize=36, title="", outFile=None):
    # Filter data
    annotations = topObject.annotations if includeCriteria is None else topObject.annotations[includeCriteria]
    projections = topObject.projections[projectionName] if includeCriteria is None else topObject.projections[projectionName].loc[:, includeCriteria]
    geneExpressions = topObject.processed if includeCriteria is None else topObject.processed.loc[:, includeCriteria]

    # Undersample cell types based on a count maximum
    projections, annotations = underSample(projections, annotations, maxLabelCount, seed=seed) if maxLabelCount is not None else (projections, annotations)
    
    # Ensure genes are in dataset
    validGenes = [gene for gene in geneList if gene in topObject.df.index]
    invalidGenes = [gene for gene in geneList if gene not in validGenes]
    if len(invalidGenes) > 0:
        print("These genes are not in the dataset: " + str(invalidGenes))

    # Create subplot for each gene using GridSpec
    geneCount = len(validGenes)
    dimX = math.ceil(math.sqrt(geneCount))
    dimY = math.ceil(geneCount / dimX)
    fig = plt.figure(figsize=(dimX * 12, dimY * 12), constrained_layout=True)
    gs = GridSpec(dimY, dimX, figure=fig)
    availableSpots = dimX * dimY

    # Set markers for legend
    labels = sorted(annotations.unique())
    labelMarkerMap = setMarkers(labels)
    palette = setPalette(labels)
    legendItems = []
    for label in labels:
        legendItems.append(Line2D([0], [0], marker=labelMarkerMap[label], color="w", label=label,
           markerfacecolor=palette[label], markersize=markerSize))

    # Plot for each gene
    axs = [fig.add_subplot(gs[i, j]) for i in range(dimY) for j in range(dimX) if dimX * i + j < geneCount]
    for i in range(geneCount):        
        # Create the 2D plot
        _ = plotTwo(topObject, projectionName, celltype1, celltype2,
                gene=validGenes[i], ax=axs[i], show=False, xBounds=xBounds, yBounds=yBounds, labels=labels, markers=labelMarkerMap,
                alternateAnnotations=annotations, alternateProjections=projections, alternateExpressions=geneExpressions, singleColorbar=False, plotMultiple=True)
    # Place legend where space available
    if geneCount < availableSpots:
        ax = fig.add_subplot(gs[dimY - 1, dimX - (availableSpots - geneCount)])
        ax.axis("off")
        ax.legend(handles=legendItems, title=topObject.name + " Labels", title_fontsize=axisFontSize, fontsize=legendFontSize, markerscale=legendMarkerScale, loc='upper left', frameon=False)
    else:
        fig.legend(legendItems, labels, loc="upper left", bbox_to_anchor=(1.025, 1), title=topObject.name + " Labels",  title_fontsize=axisFontSize,  fontsize=legendFontSize,  markerscale=legendMarkerScale, borderaxespad=0., frameon=True)

    # Add text and display/save
    fig.suptitle(title, fontsize=titleFontSize)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight')
    plt.show()

    
# 3D Similarity plot
def plotThree(projections, axis1, axis2, axis3, names, figureTitle="Similarity Plot", legendTitle="Source Annotations"):
    # fig.add_trace(go.Scatter3d(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9], mode='markers', name='Group A'))
    nameSet = set(names)
    colorMapping = {}
    i = 0
    for name in nameSet:
        colorMapping[name] = i
        i += 1
        
    fig = go.Figure()
    for name in nameSet:
        filteredProjections = projections.loc[:, names == name]
        x = filteredProjections.loc[axis1, :]
        y = filteredProjections.loc[axis2, :]
        z = filteredProjections.loc[axis3, :]
        
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color=colorMapping[name]), name=name, hovertemplate= axis1 + ": %{x:.4f}<br>"+ axis2 +": %{y:.4f}<br>"+ axis3 +": %{z:.4f}<extra></extra>"
        ))
    
    fig.update_layout(
        scene=dict(
            # xaxis=dict(range=[0, 0.4]),
            # yaxis=dict(range=[0, 0.4]),
            # zaxis=dict(range=[0, 0.4]),
            aspectmode="cube",  # Ensures all axes look the same width
            xaxis_title=axis1,
            yaxis_title=axis2,
            zaxis_title=axis3
        ),
        title=figureTitle,
        width=800,
        height=800,
        # legend_title_font=dict(size=16, color="blue"),
        legend=dict(title=legendTitle, x=1, y=1, orientation='v', xanchor='right', yanchor='top')
    )

    fig.add_trace(go.Scatter3d(x=[0.1, 0.1], y=[-0.2, 0.5], z=[0, 0],
        mode='lines', line=dict(dash='dash', width=5, color='red'), name=axis1 + "=0.1"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0.1, 0.1], z=[-0.2, 0.5],
        mode='lines', line=dict(dash='dash', width=5, color='red'), name=axis2 + "=0.1"))
    fig.add_trace(go.Scatter3d(x=[-0.2, 0.5], y=[0, 0], z=[0.1, 0.1],
        mode='lines', line=dict(dash='dash', width=5, color='red'), name=axis3 + "=0.1")) 
    fig.show()


# Make proportions plot over time
def plot_proportions(categories, times, timeSortFunc, rawCounts=False):

    # Set up the axes
    timesSorted = sorted([str(time) for time in set(times)], key=timeSortFunc)
    categoriesSorted = sorted([str(category) for category in set(categories)])
    valueCountsFrame = pd.DataFrame({"Category": categoriesSorted})
    categories = np.array(categories)
    maxY = 0

    # Collect the proportions of each category for each time
    for time in timesSorted:
        currentCategories = categories[times==time]
        values, counts = np.unique(currentCategories, return_counts=True)
        values = [str(value) for value in values]
        countsSum = sum(counts)

        if not rawCounts:
            countProportions = [float(count/countsSum) for count in counts]
            maxY = 1
        else:
            if countsSum > maxY:
                maxY = countsSum
            countProportions = [int(count) for count in counts]

        valueCountsMap = dict(zip(values, countProportions))
        proportions = []
        for label in categoriesSorted:
            if label not in valueCountsMap.keys():
                proportions.append(0)
            else:
                proportions.append(valueCountsMap[label])
        valueCountsFrame[time] = proportions
    valueCountsFrame = valueCountsFrame.set_index('Category')

    # Make the plot
    fig, ax = plt.subplots(1, 1, figsize = (2 * len(timesSorted), 8))
    ax.stackplot(timesSorted, valueCountsFrame.to_numpy(), labels=valueCountsFrame.index)
    ax.legend(bbox_to_anchor=(1.05, 1.0))
    ax.set_xlim(timesSorted[0], timesSorted[-1])
    ax.set_ylim(0, maxY)

    return fig, ax


# Get the ellipse most closely describing a cluster
def getEllipse(projections, annotations, targetLabel, colorMap, palette, axis1, axis2):
    centroids = {"x": [], "y": []}
    labelMap = {}
    labelMap[0] = []
    labelMap[1] = []
    points = []

    for i in range(len(annotations)):
        if annotations[i] == targetLabel:
            xVal = projections.loc[axis1].iloc[i]
            yVal = projections.loc[axis2].iloc[i]
            labelMap[0].append(xVal)
            labelMap[1].append(yVal)
            points.append((xVal, yVal))

    centroids["x"].append(np.array(labelMap[0]).mean())
    centroids["y"].append(np.array(labelMap[1]).mean())

    SCALE = 1
    points = np.array(points)
    width = np.quantile(points[:,0], 0.95) - np.quantile(points[:,0], 0.05)
    height = np.quantile(points[:,1], 0.95) - np.quantile(points[:,1], 0.05)

    # Calculate angle
    x_reg, y_reg = [[p[0]] for p in points], [[p[1]] for p in points]
    grad = LinearRegression().fit(x_reg, y_reg).coef_[0][0]
    angle = np.degrees(np.arctan(grad))

    # Account for multiple solutions of arctan
    if angle < -45: angle += 90
    elif angle > 45: angle -= 90
    return Ellipse((centroids["x"][0], centroids["y"][0]), width * SCALE, height * SCALE, angle=angle, fill=False, 
                   color=palette[colorMap[targetLabel]], linewidth=2)


# Create dict between cell annotations and integers representing fixed colors
def getLabelColorMap(annotations, includeOther=True):
    labelColorOrderMap = {}
    i = 0
    for label in annotations:
        if label not in labelColorOrderMap:
            if includeOther or label != "Other":
                labelColorOrderMap[label] = i
                i += 1
    return labelColorOrderMap


# Create a set of boxplots of the projections from a test set onto a basis, using similarity map outputted by getMatchingProjections
def similarityBoxplot(similarityMap, testKeep=None, basisKeep=None, groupLengths=None, title="", titleFontSize=24, 
                      labels=None, source="seaborn", labelFontSize=18, xLabelRotation=90, showOutliers=True, figY=8, outFile=None):
    
    # If no specific columns provided, use all columns of similarity map
    basisKeep = basisKeep or sorted(similarityMap.keys()) if labels is None else labels
    testKeep = testKeep or list(set([key for keyList in [list(similarityMap[val].keys()) for val in similarityMap.keys()] for key in keyList]))
    numGroups = len(testKeep)  # Number of groups
    boxesPerGroup = len(basisKeep)  # Number of boxplots per group

    # Define positions for each group
    widths = []
    groupWidth = 0.8  # Controls how wide the groups are
    if groupLengths is None:
        boxWidth = groupWidth / boxesPerGroup  # Width of each boxplot
        for i in range(numGroups):
            widths.append(boxWidth)
    else:
        for groupLength in groupLengths:
            widths.append(groupWidth / groupLength)
    fig, ax = plt.subplots(figsize=(10 + numGroups * boxesPerGroup / 20, figY))

    # Colors for each boxplot within a group
    palette = setPalette(basisKeep, source=source)
    medianlineprops = dict(linewidth=1.5, color='black')  # If you don't want a median line comment this out

    # For each label in the test set
    for i in range(numGroups):
        testLabel = testKeep[i]
        
        # For each label in the basis
        for j in range(boxesPerGroup):
            label = basisKeep[j]

            # Make boxplot for projection of cells with test labels onto the basis label
            if testLabel in similarityMap[label]:
                currentBoxWidth = widths[i]
                pos = i + j * currentBoxWidth - (groupWidth / 2) + currentBoxWidth / 2  # Offset positions
                flierProps, showFliers = (dict(marker='o', markersize=2, markerfacecolor='black', fillstyle='full'), None) if showOutliers else (None, False)
                bp = ax.boxplot(similarityMap[label][testLabel], positions=[pos], widths=currentBoxWidth, patch_artist=True, medianprops=medianlineprops, 
                        boxprops={'edgecolor': 'black'}, flierprops=flierProps, showfliers=showFliers) #, showmeans=True, meanline=True)
                for box in bp['boxes']:  # Set box color
                    box.set(facecolor=palette[label])
            else:
                print("True label not found")

    # Add vertical lines to separate groups
    y_min, y_max = ax.get_ylim()
    for i in range(1, numGroups):  # Skip first category
        group_border = i - 0.5  # Position of separator between groups
        ax.vlines(x=group_border, ymin=y_min, ymax=y_max,
                  color='black', linestyle='solid', linewidth=1)

    # Set overall plot visual elements
    ax.set_xticks(range(numGroups))
    ax.set_xticklabels(testKeep, fontsize=labelFontSize, rotation=xLabelRotation)
    ax.tick_params(labelsize=labelFontSize * 0.8)
    ax.set_xlabel("Test Set Labels", weight="bold", fontsize=labelFontSize)
    ax.set_ylabel("Cell Projection Scores", weight="bold", fontsize=labelFontSize)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=0.5, dashes=(5, 10))
    ax.legend([plt.Rectangle((0,0),1,1,facecolor=c) for c in list(palette.values())[:boxesPerGroup]], 
               basisKeep, loc="upper left", title="Reference Labels", title_fontsize=labelFontSize*0.9, fontsize=labelFontSize*0.8, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    fig.suptitle(title, fontsize=titleFontSize)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile, dpi=300)
    plt.show()


# Display correlation matrix of basis against itself
def plotBasisCorrelationMatrix(topObject, figX=8, figY=8, textSize=8, title="Basis Column Correlations", metric=None, outFile=None):

    corrCopy = topObject.corr.copy() if hasattr(topObject, "corr") else topObject.getBasisCorrelations(metric=metric)
    if type(corrCopy) is dict:
        corrCopy = pd.DataFrame.from_dict(corrCopy, orient="index")
    labels = sorted(topObject.basis.columns)

    # Plot the result
    plt.subplots(1, 1, figsize=(figX, figY))
    sns.heatmap(corrCopy, annot=True, fmt=".2f", cmap='plasma', xticklabels=labels, yticklabels=labels,
            annot_kws={"size": textSize}, cbar=True)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()


# Display confusion matrix of basis back-prediction results
def plotBasisTestConfusionMatrix(topObject, figX=8, figY=8, textSize=5, axisFontSize=18, title="Basis Test Confusion Matrix", outFile=None):

    # Build confusion matrix and set colors according to the normalized rows
    cm = confusion_matrix(topObject.testResults[1]["True"], topObject.testResults[1]["Top1"])
    colors = preprocessing.normalize(cm, axis=1)
    labels = sorted(list(set(topObject.testResults[1]["True"] + topObject.testResults[1]["Top1"])))

    # Plot the result
    plt.subplots(1, 1, figsize=(figX, figY))
    sns.heatmap(colors, annot=cm, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels, annot_kws={"size": axisFontSize // 1.1})
    plt.xticks(fontsize=axisFontSize // 1.2, rotation=90)
    plt.yticks(fontsize=axisFontSize // 1.2, rotation=0)
    plt.xlabel('Highest Projection Score', fontsize=axisFontSize)
    plt.ylabel('Predicted Label', fontsize=axisFontSize)
    plt.title(title, fontsize=axisFontSize)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()


# Volcano plot of most significant genes to scTOP predictions for a cell type
def plotPredictivity(topObject, label, basis=None, showHigh=10, labelOnly=True, figX=8, figY=8, title="", outFile=None):

    # Get predictivity and associated genes
    predictivity = topObject.predictivity if basis is None else topObject.getBasisPredictivity(basis=basis) 
    genes = predictivity.loc[label]

    # Plot expression of cell type of interest
    genesWithZeroes = topObject.processed.loc[:, topObject.annotations == label].reindex(genes.index, fill_value=0) # Fill missing genes
    averageExpressions = genesWithZeroes.loc[genes.index].T.mean()
    fig, ax = plt.subplots(1, 1, figsize=(figX, figY))
    ax.scatter(averageExpressions, genes, color='gray', alpha=0.6, label='All Genes', s=3)

    # Sort genes to get the top values and plot contributions
    if type(showHigh) is int and showHigh > 0: 
        includeCriteria = topObject.annotations == label if labelOnly else None
        scoreContributions = topObject.scoreContributions if not labelOnly and basis is None else topObject.getScoreContributions(basis=basis, includeCriteria=includeCriteria)
        highContributions = scoreContributions[label].mean(axis=1).sort_values(ascending=False).head(showHigh).index
        xHigh = averageExpressions[highContributions]
        yHigh = genes.get(highContributions).values
        ax.scatter(xHigh, yHigh, color='blue', label='Top ' + str(showHigh) + ' Genes', s=3)

        # Annotate top genes
        for gene, (x, y) in zip(highContributions, zip(xHigh, yHigh)):
            ax.text(x, y, gene, fontsize=12)

    # Set labels and title
    ax.set_xlabel('Gene Expression', fontsize=20)
    ax.set_ylabel(f'Predictivity {label}', fontsize=20)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True)
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()


# Plot expression of a given gene for each cell type in dataset
def geneViolinPlot(topObject, gene, outFile=None, figX=10, figY=4, title=""):
    data = {}
    labels = topObject.sortedCellTypes
    for celltype in labels:
        data[celltype] = topObject.processed.loc[gene, topObject.annotations == celltype]
    
    fig, ax = plt.subplots(1, 1, figsize=(figX, figY))
    violinResults = sns.violinplot(pd.DataFrame(data), inner="quartile")
    if title == "":
        title = topObject.name + " " + gene + " Normalized Expression"
    plt.title(title)
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()