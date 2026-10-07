### File containing functions for working with scTOP, made by Maria Yampolskaya, Huan Souza, and Pankaj Mehta
# Author: Eitan Vilker

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
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import polars as pl
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from scipy.stats import ttest_1samp
import os
os.environ['SCIPY_ARRAY_API'] = '1'
from imblearn.under_sampling import RandomUnderSampler
import random


## ========================= ##
## Data formatting and helper functions ##
## ========================= ##
    
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
def getMatchingProjections(topObject, projectionName, basisKeep=None, testKeep=None, includeCriteria=None, prefix=None, annotations=None):

    projection, annotations = topObject.filter(df=topObject.projections[projectionName], annotations=annotations, condition=includeCriteria)
    # projection = topObject.projection[projectionName]
    # projection = projection if includeCriteria is None else projection.loc[:, includeCriteria]
    # annotations = topObject.annotations if alternateAnnotations is None else alternateAnnotations
    # annotations = annotations if includeCriteria is None else annotations[includeCriteria]

    basisKeep = basisKeep or sorted(projection.index)
    testKeep = testKeep or sorted(set(annotations))

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
    for sampleId, sampleProjection in projection.items():
        trueLabel = topObject.metadata.loc[sampleId, topObject.cellTypeColumn]
        if trueLabel in testKeep:
            projectionTypes = sampleProjection.index
            for label in basisKeep:
                labelIndex = projectionTypes.get_loc(label)
                similarityScore = sampleProjection.iloc[labelIndex]
                adjustedTrueLabel = prefix + trueLabel if prefix else trueLabel
                similarityMap[label][adjustedTrueLabel].append(similarityScore)

    print("Similarity map built!")
    return similarityMap


# Get the dimensions a plot should be
def getBounds(projection, celltype1, celltype2, forceX=None, forceY=None, celltype3=None, forceZ=None, boundaryMaxDistance=0.06):
    x, y = (forceX or projection.loc[celltype1], forceY or projection.loc[celltype2])
    scale = 1 / boundaryMaxDistance
    xBounds = forceX or (math.floor(x.min() * scale) / scale, math.ceil(x.max() * scale) / scale)
    yBounds = forceY or (math.floor(y.min() * scale) / scale, math.ceil(y.max() * scale) / scale)
    if celltype3 is not None:
        z = forceZ or projection.loc[celltype3]
        return xBounds, yBounds, (math.floor(z.min() * scale) / scale, math.ceil(z.max() * scale) / scale)
    return xBounds, yBounds


# Set criteria for displaying data when overlaying different values from the same category of a dataset
def setupIncludeCriteria(topObject, columnName, acceptedValues, additionalCriteriaAll=None, additionalCriteriaOthers=None):
    includeCriteriaList = []
    for i, acceptedValue in enumerate(acceptedValues):
        criterion = topObject.metadata[columnName] == acceptedValue
        criterion = np.logical_and(criterion, additionalCriteriaAll) if additionalCriteriaAll is not None else criterion
        criterion = np.logical_and(criterion, additionalCriteriaOthers) if additionalCriteriaOthers is not None else criterion
        includeCriteriaList.append(criterion)
    return includeCriteriaList


# Get separate projections, annotations, etc. for purpose of overlaying different conditions
def setupOverlay(topObjects, basisName, includeCriteriaList, basis=None, forceProject=False, filterShared=False, getExpressions=False, alternateColumns=None):

    # If processing you can filter to only genes common to all datasets (not necessarily advised, but helpful for speed)
    if filterShared:
        unsharedGenes = [gene for topObject in topObjects[1:] for gene in topObjects[0].df.index if gene not in topObject.df.index]
        sharedGenes = [gene for gene in topObjects[0].df.index if gene not in unsharedGenes]

    # For each projection to overlay
    multipleTopObjects = len(topObjects) > 1
    projections, annotations, geneExpressions, identifiers, alternateColumnValues = ([], [], [] if getExpressions else None, [] if multipleTopObjects else None, [] if alternateColumns else None)
    for i, includeCriteria in enumerate(includeCriteriaList):
        topObject = topObjects[i] if multipleTopObjects else topObjects[0]
        
        # Filter to common genes if desired and perform projection if not done already or forced
        if filterShared:
            topObject.setAnndata(topObject.anndata[:, topObject.df.index.isin(sharedGenes)])
        projection = topObject.project(basis, basisName) if forceProject or basisName not in topObject.projections else topObject.projections[basisName]

        # Get data for everything but the first set, which is handled by the plotTwo function normally
        projection, annotation = topObject.filter(df=projection, condition=includeCriteria)
        projections, annotations = (projections + [projection], annotations + [annotation])
        geneExpressions = geneExpressions + [topObject.processed[annotation.index]] if getExpressions else None
        identifiers = identifiers + [topObject.identifier] if multipleTopObjects else None
        alternateColumnValues = alternateColumnValues + [topObject.metadata[annotation.index][alternateColumns[i]]] if alternateColumns else None

    # Arrange return list
    toReturn = [projections, annotations]
    toReturn = toReturn + [identifiers] if multipleTopObjects else toReturn
    toReturn = toReturn + [geneExpressions] if getExpressions else toReturn
    toReturn = toReturn + [alternateColumnValues] if alternateColumns else toReturn
    return tuple(toReturn)
    
## ========================= ##
## Decriptive statistics functions ##
## ========================= ##

# Create df of descriptive values for a cluster's projection as related to clusters in the basis
def getProjectionStats(topObject, projectionName, celltype, includeCriteria=None, target="", onlyQuantiles=False, outFile=None):
    projection = topObject.projections[projectionName]
    criteria = topObject.annotations == celltype
    criteria = criteria if includeCriteria is None else np.logical_and(criteria, includeCriteria)
    projection = projection.loc[:, criteria]
    output = {label: {} for label in projection.index}

    for label in projection.index:
        projFilt = projection.loc[label, :]
        quantiles = np.quantile(projFilt, [0.05, 0.25, 0.5, 0.75, 0.95]) #Examine
        output[label]["5% Quantile"], output[label]["25% Quantile"], output[label]["50% Quantile"], output[label]["75% Quantile"], output[label]["95% Quantile"] = quantiles
        if onlyQuantiles:
            continue
        output[label][celltype + " Count"] = len(projFilt)
        projFilt = projFilt[projFilt > 0.1]
        output[label]["Over Threshold Count"] = len(projFilt)
        output[label]["Over Threshold Proportion"] = output[label]["Over Threshold Count"] / output[label][celltype + " Count"]
        if label == target:
            newMeans = projection[projFilt.index].mean(axis=1)
            for newLabel in newMeans.index:
                output[newLabel]["Over Target Threshold Mean"] = newMeans[newLabel]
    output = pd.DataFrame.from_dict(output).T
    if outFile is not None:
        output.to_csv(outFile)
    return output


# Create df of descriptive values for each label in a datset's projections as related to a target basis cell type
def getProjectionStatsFocused(topObject, projectionName, target, includeCriteria=None, outFile=None):
    output = {}
    projection = topObject.projections[projectionName]
    
    for label in topObject.sortedCellTypes:
        criteria = topObject.annotations == label
        criteria = criteria if includeCriteria is None else np.logical_and(criteria, includeCriteria)
        projFilt = projection.loc[:, criteria]
        output[label] = {}
        output[label]["Count"] = len(projFilt.columns)
        projFilt = projFilt.loc[target, :]
        output[label]["Mean Projection"] = projFilt.mean(axis=0)
        projFilt = projFilt[projFilt > 0.1]
        output[label]["Over Threshold Count"] = len(projFilt)
        output[label]["Over Threshold Proportion"] = output[label]["Over Threshold Count"] / output[label]["Count"]
    output = pd.DataFrame.from_dict(output).T
    if outFile is not None:
        output.to_csv(outFile)
    return output


## ========================= ##
## Helper functions for plotting ##
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
        colors = ["#6E8B3D", "#EEAD0E", "#CD3333", "#008B8B", "#009ACD", "#9966CC"]
    elif source == "PRC2":
        colors = ["#1f77b4", "#ff800f", "#279257", "#da3d3e", "#b696d2", "#964B00"]
    elif source == "Jonathan":
        colors = ["#008000", "#FFA500"]
    elif source == "Jonathan2":
        colors = ["#ADD8E6", "#A65E34"]
    elif source == "Hirofumi": # Blue, green, orange, red, lilac
        colors = ["#1f77b4", "#279257", "#ff800f", "#da3d3e", "#b696d2", "#964B00"]
    else:
        if type(source) is list:
            colors = source
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


# Set the plt visual parameters for fonts, lines, etc.
def setPlotParameters(params={}, DPI=100, edgeColor="black", lineWidth=1, fontFamily="serif", fontSerif="STIXGeneral", mathFont="stix"):
    plt.rcParams.update({
        "axes.edgecolor": edgeColor,
        "figure.dpi": DPI,
        "axes.linewidth": lineWidth,
        "font.family": fontFamily,
        "font.serif": fontSerif,
        "mathtext.fontset": mathFont
    })
    plt.rcParams.update(params)

## ========================= ##
## Plotting functions ##
## ========================= ##

# Create scatter plot showing projection scores for two cell types, with the option to color according to marker gene
def plotTwo(topObject, projectionName, celltype1, celltype2, 
       includeCriteria=None, projectionsList=[], annotationsList=[], expressionsList=[], namesList=[], alternatesList=[],
       alternateColumn=None, annotations=None, projections=None, gene=None, geneExpressions=None, randomOrder=False,
       plotMultiple=False, singleColorbar=False, unsupervisedContour=False, supervisedContour=False, maxLabelCount=None, keepFull=[], seed=1, axisRenames=(None, None),
       ax=None, figX=10, figY=10, DPI=100, name=None, hue=None, labels=None, labelDimensions=False, overlayLabels=None, palette=None, alpha=1, source="seaborn", 
       markers=None, markerSize=80, legendMarkerScale=2.5, legendWidth=116.625, lineWidth=1.5, legendSpacing=0.05, xBounds=None, yBounds=None, maxBounds=False, boundaryMaxDistance=0.053, plotThreshold=True, 
       plotParameters={}, axisFontSize=36, legendFontSize=20, titleFontSize=None, legendTitle=None, legendInner=True, legendReplacements={},
       title=None, getSamples=False, outFile=None, show=True):

    # Prepare data elements if single dataset
    overlayCount = len(namesList)
    if overlayCount == 0:
        annotations = topObject.annotations if annotations is None else annotations
        projections = topObject.projections[projectionName] if projections is None else projections
        geneExpressions = None if not gene else (topObject.processed if geneExpressions is None else geneExpressions).loc[gene, :]
        alternateColumnValues = None if not alternateColumn else topObject.metadata[alternateColumn]
        name = topObject.identifier if name is None else name

    # If overlaying multiple sets of projections, combine along shared genes
    else:
        topObject = topObject or TopObject.TopObject("", datasetCollection=None) # Create dummy TopObject for multiple overlays
        projections = pd.concat(projectionsList, axis=1, join='inner')
        annotations = []

        # For each projection, add its name to its features so they can be distinguished from each other
        for i in range(overlayCount):
            annotations.append(annotationsList[i].apply(lambda annotation: namesList[i] + ' ' + annotation))
        annotations = pd.concat(annotations)
        geneExpressions = None if not gene else pd.concat(expressionsList, axis=1, join='inner').loc[gene, :]
        alternateColumnValues = pd.concat(alternatesList) if alternateColumn else None

        if overlayLabels is not None:
            xBounds, yBounds = getBounds(projections.loc[:, annotations.index], celltype1, celltype2, forceX=xBounds, forceY=yBounds, boundaryMaxDistance=boundaryMaxDistance)
            includeCriteria = annotations.isin(overlayLabels)

    # Filter dataset
    projections, annotations = topObject.filter(df=projections, annotations=annotations, condition=includeCriteria, maxSamples=maxLabelCount, keepFull=keepFull, seed=seed)
    samples = annotations.index
    geneExpressions = geneExpressions[samples] if gene else None
    alternateColumnValues = alternateColumnValues[samples] if alternateColumn else None
    xBounds, yBounds = getBounds(projections, celltype1, celltype2, forceX=xBounds, forceY=yBounds, boundaryMaxDistance=boundaryMaxDistance)

    # Reorder so high expression genes are stacked on top if plotting gene heatmap, otherwise random
    order = geneExpressions.sort_values(ascending=True).index if gene and not randomOrder else random.sample(list(samples), len(samples))
    geneExpressions = geneExpressions[order] if gene else None
    projections, annotations, alternateColumnValues = (projections[order], annotations[order], alternateColumnValues[order] if alternateColumn else None)
    
    # Set axes and key parameters for plot
    setPlotParameters(params=plotParameters, DPI=DPI, lineWidth=lineWidth)
    fig, ax = plt.subplots(1, 1, figsize=(figX + legendSpace, figY)) if ax is None else (None, ax)
    x, y = (projections.loc[celltype1], projections.loc[celltype2])
    labels = sorted(annotations.unique()) if labels is None else labels
    palette = setPalette(labels, source=source) if palette is None and gene is None and alternateColumn is None else palette
    markers = setMarkers(labels) if markers is None else markers
    legendFontSize = legendFontSize or axisFontSize * 4/3
    if show and len(projectionsList) == 0:
        title = topObject.identifier + " Projected Onto " + projectionName + " Reference" if title is None else title
        legendTitle = topObject.identifier + " Cell Labels" if legendTitle is None else legendTitle

    # Create core plots
    if gene:  # If labeling by gene expression instead of source labels
        ax = geneExpressionPlot(ax, x, y, gene, geneExpressions, annotations, palette,
                labels=labels, markers=markers, markerSize=markerSize, alpha=alpha, axisFontSize=axisFontSize, singleColorbar=singleColorbar)
    elif alternateColumn:  # If labeling by a specifc other column, such as disease type
        ax = alternateColumnPlot(ax, x, y, alternateColumn, alternateColumnValues, annotations, palette, 
                labels=labels, markers=markers, markerSize=markerSize, alpha=alpha, axisFontSize=axisFontSize, singleColorbar=singleColorbar)
    else:  # If labeling is by cell type
        ax, legend = testLabelPlot(ax, x, y, annotations, palette, 
                labels=labels, title=legendTitle, markers=markers, markerSize=markerSize, alpha=alpha, legendMarkerScale=legendMarkerScale, axisFontSize=axisFontSize, legendFontSize=legendFontSize, legendInner=legendInner, plotMultiple=plotMultiple, legendSpace=legendSpace, legendReplacements=legendReplacements)

    # Add contours if desired
    ax = unsupervisedContourPlot(ax, x, y) if unsupervisedContour else ax
    ax = supervisedContourPlot(ax, x, y, annotations, labels, palette=palette, source=source) if supervisedContour else ax

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
    ax.set_xlabel(axisRenames[0] or projectionName + " " + celltype1 + " Cell Score", fontsize=axisFontSize)
    ax.set_ylabel(axisRenames[1] or projectionName + " " + celltype2 + " Cell Score", fontsize=axisFontSize)

    if title is not None and title != "":
        plt.title(title, fontsize=titleFontSize or axisFontSize // 0.95, pad=10)

    # Adjust plot to be larger according to desired fraction of the legend
    if not (gene or alternateColumn or plotMultiple):
        fig.canvas.draw()        
        renderer = fig.canvas.get_renderer()
        bbox_fig = legend.get_window_extent().transformed(fig.transFigure.inverted())
        plt.subplots_adjust(right=1.0 - bbox_fig.width - 0.05)
    
    if plotMultiple:
        pass
    else:
        ax.set_box_aspect(1)
        plt.tight_layout()
        if outFile is not None:
            plt.savefig(outFile, bbox_inches='tight', dpi=DPI)
        if show:
            plt.show()
    return (ax, annotations.index) if getSamples else ax


# Creates a Seaborn 2D scatter plot using projections onto basis columns as axes and gene expressions to identify points. Helper for plotTwo
def geneExpressionPlot(ax, x, y, gene, geneExpressions, annotations, palette,
                       labels=None, markers=True, markerSize=40, axisFontSize=16, alpha=1, plotMultiple=False, singleColorbar=False):
    palette, scalarmap = createColorbar(geneExpressions) if palette is None else (palette, None)
    plot = sns.scatterplot(x=x, y=y, ax=ax, hue=geneExpressions, style=annotations, style_order=labels, markers=markers, s=markerSize, palette=palette, alpha=alpha, linewidth=0.15)
    if not singleColorbar:
        cbar = plt.colorbar(scalarmap, ax=ax, fraction=0.044) # scalarmap won't be defined unless palette wasn't, which is only in single plot
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        cbar.set_label('{} expression'.format(gene), size=axisFontSize)

    plot.legend_.remove()
    return ax


# Creates a Seaborn 2D scatter plot using projections onto basis columns as axes and a specified column's values to identify points. Helper for plotTwo
def alternateColumnPlot(ax, x, y, alternateColumn, alternateColumnValues, annotations, palette,
                       labels=None, markers=True, markerSize=40, axisFontSize=16, alpha=1, plotMultiple=False, singleColorbar=False):
    palette, scalarmap = createColorbar(alternateColumnValues) if palette is None else (palette, None)
    plot = sns.scatterplot(x=x, y=y, ax=ax, hue=alternateColumnValues, style=annotations, style_order=labels, markers=markers, s=markerSize, palette=palette, alpha=alpha, linewidth=0.15)

    if not singleColorbar:
        cbar = plt.colorbar(scalarmap, ax=ax, fraction=0.044) # scalarmap won't be defined unless palette wasn't, which is only in single plot
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        cbar.set_label('{}'.format(alternateColumn), size=axisFontSize, labelpad=20)

    plot.legend_.remove()
    return ax


# Creates a Seaborn 2D scatterplot using projections onto basis columns as axes and source labels to identify points
def testLabelPlot(ax, x, y, annotations, palette, title="", labels=None, markers=True, markerSize=40, legendMarkerScale=2, axisFontSize=16, legendFontSize=16, legendInner=False, alpha=1, plotMultiple=False, legendSpace=1, legendReplacements={}):
    plot = sns.scatterplot(x=x, y=y, ax=ax, hue=annotations, style=annotations, hue_order=labels, style_order=labels, markers=markers, s=markerSize, palette=palette, alpha=alpha, linewidth=0.15)
    if plotMultiple:
        plot.legend_.remove()
    else:
        newHandles, newLabels = ax.get_legend_handles_labels()
        if labels is not None:
            keepIndices = [i for i, label in enumerate(labels) if label in sorted(annotations.unique())]
            newHandles, newLabels = ([newHandles[i] for i in keepIndices], [newLabels[i] for i in keepIndices])
        newLabels = [legendReplacements[label] if label in legendReplacements else label for label in newLabels]
        bbox, loc, padding = (None, 'upper right', 0.5) if legendInner else ((1.05, 1), 'upper left', 0.) 
        leg = ax.legend(newHandles, newLabels, title=title, title_fontsize=legendFontSize // 0.9, fontsize=legendFontSize, markerscale=legendMarkerScale, bbox_to_anchor=bbox, loc=loc, borderaxespad=padding)
    
    return (ax, None if plotMultiple else leg)


# Creates contour ellipses in unsupervised manner
def unsupervisedContourPlot(ax, x, y, contourColor=sns.color_palette()[0]):
    sns.kdeplot(ax=ax, x=x, y=y, fill=True, color=contourColor, alpha=0.2)
    sns.kdeplot(ax=ax, x=x, y=y, color=contourColor)
    return ax


# Creates contour ellipses in supervised manner
def supervisedContourPlot(ax, x, y, annotations, labels, palette=None, source="seaborn"):
    palette = palette or setPalette(labels, source=source)
    for label in labels:
        xLabel = x[annotations == label]
        yLabel = y[annotations == label]
        sns.kdeplot(ax=ax, x=xLabel, y=yLabel, fill=True, color=palette[label], alpha=0.3, label=label, thresh=0.1)
        sns.kdeplot(ax=ax, x=xLabel, y=yLabel, color=palette[label], label=label, thresh=0.1)
    return ax


# Plot multiple 2D similarity plots at once based on some field, such as time (Note: figure out difference if any between labels and annotations[toInclude])
def plotTwoMultiple(topObject, projectionName, celltype1, celltype2,
                    annotations=None, subsetCategory=None, subsetNames=None, gene=None, randomOrder=False, includeCriteria=None, singleColorbar=True, figX=8, figY=8,
                    unsupervisedContour=False, supervisedContour=False, maxLabelCount=None, keepFull=[], alternateColumn=None, axisRenames=(None, None), seed=None,
                    plotParameters={}, xBounds=None, yBounds=None, boundaryMaxDistance=0.06, plotInRow=False, plotThreshold=True, axisFontSize=32, legendFontSize=32, legendReplacements={}, 
                    labels=None, alpha=1, DPI=100, source="seaborn",
                    legendMarkerScale=0.7, markerSize=40, titleFontSize=36, title=None, legendTitle=None, caption=None, outFile=None):

    # Initialize categories
    setPlotParameters(params=plotParameters, DPI=DPI)
    projections, annotations = topObject.filter(df=topObject.projections[projectionName], annotations=annotations, condition=includeCriteria, seed=seed)
    x, y = (projections.loc[celltype1], projections.loc[celltype2])
    xBounds = (math.floor(x.min() * 20) / 20, math.ceil(x.max() * 20) / 20) if xBounds is None else xBounds
    yBounds = (math.floor(y.min() * 20) / 20, math.ceil(y.max() * 20) / 20) if yBounds is None else yBounds
    if subsetNames is None and subsetCategory is None:
        subsetCategory = (topObject.metadata[topObject.timeColumn])[annotations.index]
        subsetNames = [time for time in topObject.timesSorted if time in set(subsetCategory)]
    else:
        subsetCategory = (topObject.metadata[topObject.timeColumn] if subsetCategory is None else subsetCategory)[annotations.index]
        subsetNames = sorted(list(set(subsetCategory)))

    # Get subplots
    subsetCount = len(subsetNames)
    dimX = subsetCount if plotInRow else math.ceil(math.sqrt(subsetCount))
    dimY = 1 if plotInRow else math.ceil(subsetCount / dimX)
    fig = plt.figure(figsize=(dimX * figX, dimY * figY), constrained_layout=True)
    gs = GridSpec(dimY, dimX, figure=fig)
    availableSpots = dimX * dimY

    # Set up label colors and shapes
    labels = sorted(annotations.unique()) if labels is None else labels
    labelMarkerMap = setMarkers(labels)
    if gene:
        geneExpressions = topObject.processed
        palette, scalarmap = createColorbar(geneExpressions.loc[gene, annotations.index])
    elif alternateColumn:
        palette, scalarmap = createColorbar(topObject.metadata[alternateColumn][annotations.index])
    else:
        palette = setPalette(labels, source=source)
        legendItems = []
        for label in labels:
            legendItems.append(Line2D([0], [0], marker=labelMarkerMap[label], color="w", label=label,
               markerfacecolor=palette[label], markersize=markerSize))
    legendFontSize = legendFontSize or axisFontSize * 4/3
    title = topObject.identifier + " Projected Onto " + projectionName + " Reference" if title is None else title
    legendTitle = topObject.identifier + " Cell Labels" if legendTitle is None else legendTitle

    # Plot for each subset
    axs = [fig.add_subplot(gs[i, j]) for i in range(dimY) for j in range(dimX) if dimX * i + j < subsetCount]
    for i, subset in enumerate(subsetNames):
        ax = plotTwo(topObject, projectionName, celltype1, celltype2, projections=projections, annotations=annotations, alternateColumn=alternateColumn,
            ax=axs[i], includeCriteria=subsetCategory == subset, labels=labels, gene=gene, geneExpressions=geneExpressions if gene else None, randomOrder=randomOrder, axisRenames=axisRenames,
            unsupervisedContour=unsupervisedContour, supervisedContour=supervisedContour, plotMultiple=True, singleColorbar=singleColorbar, 
            plotParameters=plotParameters, plotThreshold=plotThreshold, maxLabelCount=maxLabelCount, xBounds=xBounds, yBounds=yBounds, boundaryMaxDistance=boundaryMaxDistance, seed=seed,
            palette=palette, alpha=alpha, markers=labelMarkerMap, markerSize=markerSize, lineWidth=2.5, legendMarkerScale=legendMarkerScale, axisFontSize=axisFontSize, legendFontSize=legendFontSize, title="", legendTitle=""
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
        labels = [legendReplacements[label] if label in legendReplacements else label for label in labels]
        if subsetCount < availableSpots:
            ax = fig.add_subplot(gs[dimY - 1, dimX - (availableSpots - subsetCount)])
            ax.axis("off")
            ax.legend(legendItems, labels, title=legendTitle, title_fontsize=legendFontSize, fontsize=legendFontSize, markerscale=legendMarkerScale, loc='upper left', frameon=False)
        else:
            fig.legend(legendItems, labels, loc="upper left", bbox_to_anchor=(1.0125 if plotInRow else 1.025, 1), title=legendTitle, title_fontsize=legendFontSize, fontsize=legendFontSize, markerscale=legendMarkerScale, borderaxespad=0., frameon=True)

    # Add text and display/save
    if caption:
        caption = (caption)
        wrappedCaption = "\n".join(textwrap.wrap(caption, width=170))
        fig.text(0, -0.05, wrappedCaption, ha='left', va='bottom', fontsize=axisFontSize // 1.1)
    fig.suptitle(title, fontsize=titleFontSize)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=DPI)
    plt.show()


# Make 2D similarity plot of each gene in a selected list
def plotMultipleGenes(topObject, projectionName, celltype1, celltype2, geneList,
                     includeCriteria=None, xBounds=None, yBounds=None, boundaryMaxDistance=0.06, randomOrder=False, maxLabelCount=None, keepFull=[], axisRenames=(None, None), seed=0,
                     plotParameters={}, DPI=100, lineWidth=1.5, legendMarkerScale=0.7, markerSize=80, axisFontSize=40, legendFontSize=60, legendTitle=None, titleFontSize=48, title="", 
                     outFile=None):
        
    # Filter data
    annotations = topObject.annotations if includeCriteria is None else topObject.annotations[includeCriteria]
    projections = topObject.projections[projectionName] if includeCriteria is None else topObject.projections[projectionName].loc[:, includeCriteria]
    geneExpressions = topObject.processed if includeCriteria is None else topObject.processed.loc[:, includeCriteria]

    # Undersample cell types based on a count maximum
    if maxLabelCount is not None:
        projections, annotations = TopObject.downsample(annotations, maxLabelCount, df=projections, seed=seed, keepFull=keepFull)
        geneExpressions = geneExpressions[annotations.index]
    
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

    # Set legend and other visual elements
    setPlotParameters(params=plotParameters, DPI=DPI, lineWidth=lineWidth)
    legendTitle = legendTitle or topObject.identifier + " Labels"
    labels = sorted(annotations.unique())
    labelMarkerMap = setMarkers(labels)
    palette = setPalette(labels)
    legendItems = []
    for label in labels:
        legendItems.append(Line2D([0], [0], marker=labelMarkerMap[label], color="w", label=label,
           markerfacecolor=palette[label], markersize=markerSize))

    # Plot for each gene
    axs = [fig.add_subplot(gs[i, j]) for i in range(dimY) for j in range(dimX) if dimX * i + j < geneCount]
    for i, gene in enumerate(validGenes):        
        # Create the 2D plot
        _ = plotTwo(topObject, projectionName, celltype1, celltype2,
                gene=gene, ax=axs[i], show=False, xBounds=xBounds, yBounds=yBounds, boundaryMaxDistance=boundaryMaxDistance, labels=labels, markers=labelMarkerMap, axisRenames=axisRenames,
                annotations=annotations, projections=projections, geneExpressions=geneExpressions, randomOrder=randomOrder, singleColorbar=False, plotMultiple=True,
                plotParameters=plotParameters, legendMarkerScale=legendMarkerScale, markerSize=markerSize, axisFontSize=axisFontSize, legendFontSize=legendFontSize)
    # Place legend where space available
    if geneCount < availableSpots:
        ax = fig.add_subplot(gs[dimY - 1, dimX - (availableSpots - geneCount)])
        ax.axis("off")
        ax.legend(handles=legendItems, title=legendTitle, title_fontsize=axisFontSize, fontsize=legendFontSize, markerscale=legendMarkerScale, loc='upper left', frameon=False)
    else:
        fig.legend(legendItems, labels, loc="upper left", bbox_to_anchor=(1.025, 1), title=legendTitle,  title_fontsize=axisFontSize,  fontsize=legendFontSize,  markerscale=legendMarkerScale, borderaxespad=0., frameon=True)

    # Add text and display/save
    fig.suptitle(title, fontsize=titleFontSize)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight')
    plt.show()

    
# 3D Similarity plot (less polished than other plots)
def plotThree(topObject, projectionName, axis1, axis2, axis3, labels=None, maxLabelCount=None, figureTitle="Similarity Plot", legendTitle="Source Annotations"):
    colorMapping = {}
    i = 0
    labels = labels or topObject.sortedCellTypes
    projections = topObject.projections[projectionName].loc[:, topObject.annotations.isin(labels)]
    annotations = topObject.annotations[projections.columns]
    if maxLabelCount is not None:
        projections, annotations = TopObject.downsample(annotations, maxLabelCount, df=projections, keepFull=[], seed=1)

    for name in labels:
        colorMapping[name] = i
        i += 1
        
    fig = go.Figure()
    for label in labels:
        filteredProjections = projections.loc[:, annotations == label]
        x = filteredProjections.loc[axis1, :]
        y = filteredProjections.loc[axis2, :]
        z = filteredProjections.loc[axis3, :]
        
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color=colorMapping[name]), name=label, hovertemplate= axis1 + ": %{x:.4f}<br>"+ axis2 +": %{y:.4f}<br>"+ axis3 +": %{z:.4f}<extra></extra>"
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
        legend=dict(title=legendTitle, x=1, y=1, orientation='v', xanchor='right', yanchor='top')
    )
    xBounds, yBounds, zBounds = getBounds(projections, axis1, axis2, celltype3=axis3)
    meshY, meshZ = np.meshgrid(np.linspace(yBounds[0], yBounds[1], 2), np.linspace(zBounds[0], zBounds[1], 2))
    fig.add_trace(go.Surface(x=np.full_like(meshY, 0.1), y=meshY, z=meshZ, opacity=0.3, colorscale=[[0, "red"], [1, "red"]], showscale=False, name=axis1 + "=0.1"))
    meshX, meshZ = np.meshgrid(np.linspace(xBounds[0], xBounds[1], 2), np.linspace(zBounds[0], zBounds[1], 2))
    fig.add_trace(go.Surface(y=np.full_like(meshZ, 0.1), x=meshX, z=meshZ, opacity=0.3, colorscale=[[0, "blue"], [1, "blue"]], showscale=False, name=axis2 + "=0.1"))
    meshX, meshY = np.meshgrid(np.linspace(xBounds[0], xBounds[1], 2), np.linspace(yBounds[0], yBounds[1], 2))
    fig.add_trace(go.Surface(z=np.full_like(meshX, 0.1), y=meshY, x=meshX, opacity=0.3, colorscale=[[0, "green"], [1, "green"]], showscale=False, name=axis3 + "=0.1"))
    fig.show()


# Create a set of boxplots of the projections from a test set onto a basis, using similarity map outputted by getMatchingProjections
def similarityBoxplot(similarityMap, testKeep=None, basisKeep=None, groupLengths=None, groupWidth=0.8, title="", titleFontSize=24, DPI=100,
                      labels=None, source="seaborn", labelFontSize=18, xLabelRotation=90, showOutliers=True, figY=8, outFile=None):
    
    # If no specific columns provided, use all columns of similarity map
    basisKeep = basisKeep or sorted(similarityMap.keys()) if labels is None else labels
    testKeep = testKeep or list(set([key for keyList in [list(similarityMap[val].keys()) for val in similarityMap.keys()] for key in keyList]))
    numGroups = len(testKeep)  # Number of groups
    boxesPerGroup = len(basisKeep)  # Number of boxplots per group
    widths = [groupWidth / boxesPerGroup for i in range(numGroups)] if groupLengths is None else [groupWidth / groupLength for groupLength in groupLengths]
    fig, ax = plt.subplots(figsize=(10 + numGroups * boxesPerGroup / 20, figY))

    # Colors for each boxplot within a group
    palette = setPalette(basisKeep, source=source)
    medianlineprops = dict(linewidth=1.5, color='black')  # If you don't want a median line comment this out

    # For each label in the test set
    for i, testLabel in enumerate(testKeep):
        
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
    yMin, yMax = ax.get_ylim()
    for i in range(1, numGroups):  # Skip first category
        groupBorder = i - 0.5  # Position of separator between groups
        ax.vlines(x=groupBorder, ymin=yMin, ymax=yMax,
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
        plt.savefig(outFile, dpi=DPI)
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
def plotBasisTestConfusionMatrix(topObject, figX=12, figY=12, axisFontSize=30, xRotation=90, showPercent=False, square=False, cbar=False, decimalMode="", labelMode="Decimal", fmt='', title="Basis Test Confusion Matrix", outFile=None):

    # Build confusion matrix and set colors according to the normalized rows
    trueLabels, highScoreLabels = (topObject.testResults[1]["True"], topObject.testResults[1]["Top1"])
    cm = confusion_matrix(trueLabels, highScoreLabels)
    cmDecimal = confusion_matrix(trueLabels, highScoreLabels, normalize="true")
    xLabels = sorted(list(set(trueLabels + highScoreLabels)))
    yLabels = xLabels.copy()
    if "Unspecified" in yLabels:
        yLabels.remove("Unspecified")
        cm, cmDecimal = (cm[:-1, :], cmDecimal[:-1, :])
    if "Unspecified" in xLabels and topObject.testResults[0]["Unspecified"] / topObject.testResults[0]["Total test count"] < 0.01:
        xLabels.remove("Unspecified")
        cm, cmDecimal = (cm[:, :-1], cmDecimal[:, :-1])        

    # colors = preprocessing.normalize(cm, axis=1) if colorNormal else cm #preprocessing.normalize(cm, axis=1)

    if decimalMode:
        labels = cmDecimal
        if decimalMode == "NoLead":
            labels = np.char.replace(np.around(labels, 2).astype(str), '0.', '.')
        elif decimalMode == "Clean":
            labels = np.around(labels, 2).astype(str)
            colCount, rowCount = labels.shape
            for i in range(colCount):
                for j in range(rowCount):
                    label = labels[i, j]
                    labels[i, j] = '0' if label == '0.0' else label
    elif showPercent:
        labels = cmDecimal
        # colors = cmDecimal
        fmt = '.0%'  # Normal percent
        # cm = np.round(cm * 100, decimals=0).astype(int) if showPercent else cm  # Symbol-less percent
    else:
        labels = cm

    # Set colors
    colors = cmDecimal if labelMode == "Decimal" else preprocessing.normalize(cm, axis=1) if labelMode == "Normal" else cm

    # Plot the result
    plt.subplots(1, 1, figsize=(figX, figY))
    sns.heatmap(colors, annot=labels, fmt=fmt, cmap='Blues', cbar=cbar, square=square, cbar_kws={"shrink": 0.6} if square else None, xticklabels=xLabels, yticklabels=yLabels, annot_kws={"size": axisFontSize // 1.1})
    plt.xticks(fontsize=axisFontSize // 1.2, rotation=xRotation)
    plt.yticks(fontsize=axisFontSize // 1.2, rotation=0)
    plt.xlabel('Reclassified Label', fontsize=axisFontSize)
    plt.ylabel('Original Label', fontsize=axisFontSize)
    plt.title(title, fontsize=axisFontSize)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=300)
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
        plt.savefig(outFile, bbox_inches='tight', dpi=300)
    plt.show()


# Plot expression of a given gene for each cell type in dataset
def geneViolinPlot(topObject, gene, outFile=None, figX=10, figY=4, axisFontSize=12, title=""):
    data = {}
    labels = topObject.sortedCellTypes
    df = topObject.processed if topObject.processed is not None else topObject.df
    for celltype in labels:
        data[celltype] = df.loc[gene, topObject.annotations == celltype]
    
    fig, ax = plt.subplots(1, 1, figsize=(figX, figY))
    plt.rcParams.update({
        "font.size": axisFontSize,
    })
    violinResults = sns.violinplot(pd.DataFrame(data), inner="quartile")
    if title == "":
        title = topObject.name + " " + gene + " Normalized Expression"
    plt.title(title)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=300)
    plt.show()


# Plot expression of a given gene for each cell type in dataset
def projectionViolinPlot(topObject, projectionName, target, labels=None, outFile=None, figY=2, title="", source="seaborn", axisFontSize=6, includeCriteria=None, DPI=100):
    data = {}
    projections, annotations = topObject.filter(df=topObject.projections[projectionName], condition=includeCriteria)
    labels = sorted(list(set(annotations))) if labels is None else labels
    for celltype in labels:
        data[celltype] = projections.loc[target, annotations == celltype]
    
    fig, ax = plt.subplots(1, 1, figsize=(len(labels), figY))
    plt.rcParams.update({
        "font.size": axisFontSize,
    })
    violinResults = sns.violinplot(pd.DataFrame(data), inner="quartile", palette=setPalette(labels, source=source))
    if title == "":
        title = topObject.identifier + " " + target + " Projection Scores"
    plt.title(title)
    plt.ylabel(target + " Projection Scores")
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=DPI)
    plt.show()


# Plot histogram of a cell state's projections onto a specific basis cell state
def cellCellProjectionHistogram(topObject, projectionName, testCellType, basisCellType, bins=30, density=True, stacked=True, cumulative=True, outFile=None):
    plt.hist(topObject.projections[projectionName].loc[basisCellType, :][topObject.annotations == testCellType], bins=bins, density=density, stacked=stacked, cumulative=cumulative)
    if outFile is not None:
        plt.savefig(outFile)

    
# Display confusion matrix of basis back-prediction results
def plotProjectionResultsMatrix(topObject, projectionName, figX=8, figY=8, textSize=5, axisFontSize=18, labels=None, square=False, cbar=False, title="Mean Projection Scores", outFile=None):

    projection = topObject.projections[projectionName]
    projectionMap = {}
    labels = labels or topObject.sortedCellTypes
    for label in labels:
        currentProjection = projection.loc[:, topObject.annotations == label]
        projectionMap[label] = list(currentProjection.mean(axis=1))

    projectionLabels = list(projection.index)
    projectionsFrame = pd.DataFrame(projectionMap, index=projectionLabels).T
    colors = projectionsFrame #preprocessing.normalize(projectionsFrame, axis=1)
    
    # Plot the result
    plt.subplots(1, 1, figsize=(figX, figY))
    sns.heatmap(colors, annot=projectionsFrame, fmt='.2f', cmap='plasma', cbar=cbar, square=square, cbar_kws={"shrink": 0.6} if square else None, xticklabels=projectionLabels, yticklabels=labels, annot_kws={"size": axisFontSize // 1.1})
    plt.xticks(fontsize=axisFontSize // 1.2, rotation=90)
    plt.yticks(fontsize=axisFontSize // 1.2, rotation=0)
    plt.xlabel('Projected Annotation', fontsize=axisFontSize)
    plt.ylabel('Original Annotation', fontsize=axisFontSize)
    plt.title(title, fontsize=axisFontSize)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=300)
    plt.show()


# Plot heatmap of the significance metrics for cell states' projections onto basis cell states based on t-tests over test cells
def getOverThresholdSignificanceMatrix(topObject, projectionName, threshold=0.1, test="pvalue", textSize=5, axisFontSize=18, labels=None, square=False, cbar=False, title="Over Threshold Statistics", includeCriteria=None, outFile=None):
    
    projection = topObject.projections[projectionName]
    projection = projection if includeCriteria is None else projection.loc[:, includeCriteria]
    labels = labels or topObject.sortedCellTypes
    projectionMap = {label: {} for label in labels}
    
    for label in labels:
        labelProjection = projection.loc[:, topObject.annotations == label]

        for target in labelProjection.index:
            labelTargetProjection = labelProjection.loc[target, :]
            testResult = ttest_1samp(labelTargetProjection, threshold, alternative="greater")
            if test == "pvalue":
                projectionMap[label][target] = testResult.pvalue
            elif test == "statistic":
                projectionMap[label][target] = testResult.statistic

    projectionLabels = list(projection.index)
    projectionsFrame = pd.DataFrame(projectionMap, index=projectionLabels).T
    colors = preprocessing.normalize(projectionsFrame, axis=1) if test == "statistic" else projectionsFrame
    
    # Plot the result
    plt.subplots(1, 1, figsize=(1.5 * len(projection.index) + 2, 1.5 * len(labels) + 2))
    sns.heatmap(colors, annot=projectionsFrame, fmt='.2f', cmap='plasma', cbar=cbar, square=square, cbar_kws={"shrink": 0.6} if square else None, xticklabels=projectionLabels, yticklabels=labels, annot_kws={"size": axisFontSize // 1.1})
    plt.xticks(fontsize=axisFontSize // 1.2, rotation=90)
    plt.yticks(fontsize=axisFontSize // 1.2, rotation=0)
    plt.xlabel('Projected Annotation', fontsize=axisFontSize)
    plt.ylabel('Original Annotation', fontsize=axisFontSize)
    plt.title(title, fontsize=axisFontSize)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()

