### File containing functions for working with scTOP, made by Maria Yampolskaya and Pankaj Mehta
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
from matplotlib.cm import ScalarMappable
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


# Gets the MC-KO basis made by Michael Herriges in the Kotton Lab with only mouse lung epithelial cells
def loadMCKOBasis(lung=True):
    basis = top.load_basis("MC-KO", 50)
    basis = basis[0]
    if lung:
        basis = basis[[colName for colName in basis.columns if "Lung" in colName]]
    basis.drop("Lung Endothelial Cells WK6-10 MC20", axis=1, inplace=True)
    newCols = {}
    for col in basis.columns:
        idx = col.find("Cell")
        if idx == -1:
            idx = col.find("WK6")
        if idx != -1:
            newCols[col] = col[5:idx - 1]

    basis = basis.rename(columns=newCols)
    return basis


# Function to load a basis given a file location or basis name and summary csv
def loadBasis(file=None, basisCollection=None, basisName=None, filtering=False, includeRas=False, basisKeep=["Alveolar type 2", "Alveolar type 1", "Basal", "Ciliated", "Goblet", "Secretory"]):
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
        basis = pd.read_csv(file, index_col="gene")
    else:
        print("Unsupported file type")
        return None

    # Reduce wordiness in basis names for LungMAP
    newCols = {}
    for col in basis.columns:
        idx = col.find("cell")
        if idx != -1:
            newCols[col] = col[:idx - 1]

    basis = basis.rename(columns=newCols)
    # if includeRas:
    #     basisKeep.append("Respiratory airway secretory")

    if filtering:
        basis = basis[[colName for colName in basis.columns if colName in basisKeep]]
    print("Loaded basis!")
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
    obj = annDataObj.copy()
    print("Setting X as csr_matrix...")
    obj.X = csr_matrix(obj.X)
    # varFrame = pd.DataFrame(index=obj.var.index)
    # obj.var.index = varFrame
    # obj.var.drop(columns=obj.var.columns, inplace=True)
    if obj.raw is not None:
        print("Setting raw...")
        if indexReplace is not None:
            obj._raw._var.rename(columns={indexReplace: 'index'}, inplace=True)
            obj.raw.var.index.name(columns={indexReplace: 'index'}, inplace=True)
    print("Writing h5ad...")
    obj.write_h5ad(outFile)
    print("Finished!")


# Get average projections given time series data
def getTimeAveragedProjections(basis, df, cellLabels, times, timeSortFunc, substituteMap=None):

    # basis = obj.basis
    # df = obj.df
    # cellLabels = obj.annotation
    # times = obj.metadata[obj.timeColumn]
    # timeSortFunc = obj.timeSortFunction
    projections = {}
    processedData = {}
    timesSorted = sorted([str(time) for time in set(times)], key=timeSortFunc)

    for time in tqdm(timesSorted):
        types = pd.DataFrame(cellLabels.value_counts())
        for current_type in list(types.index):
            current_processed = top.process(df.loc[:, np.logical_and(times==time, cellLabels == current_type)], average=True)
            processedData[current_type] = current_processed
            current_scores = top.score(basis, current_processed)
            if substituteMap is not None:
                projectionKey = substituteMap[current_type] + "_" + time
            else:
                projectionKey = current_type + "_" + time
            projections[projectionKey] = current_scores

    return projections


# Get a dict of cell types in basis to the similarity scores of each type in the source. 
# Format: Key (basis label) -> Key (source label) -> Value (projection score list for cells with source label onto the basis label)
def getMatchingProjections(topObject, projectionName, basisKeep=None, sourceKeep=None, includeCriteria=None, prefix=None):

    # If no specific columns provided, use all columns
    projections = topObject.projections[projectionName]
    if includeCriteria is not None:
        projections = projections.loc[:, includeCriteria]
    if basisKeep is None:
        basisKeep = sorted(projections.index)
    if sourceKeep is None:
        sourceKeep = sorted(set(topObject.annotations))

    # Initialize map of basis labels
    similarityMap = {}
    for label in basisKeep:
        similarityMap[label] = {}

    # Broaden map to go from basis labels to source labels (with option to include prefix usually to indicate name of source)
    for trueLabel in sourceKeep:
        for label in similarityMap:
            adjustedTrueLabel = prefix + trueLabel if prefix else trueLabel
            similarityMap[label][adjustedTrueLabel] = []

    # For every sample, add its projection scores for each label in the basis, but only for the label assigned in the source
    for sampleId, sampleProjections in projections.items():
        trueLabel = topObject.metadata.loc[sampleId, topObject.cellTypeColumn]
        if trueLabel in sourceKeep:
            projectionTypes = sampleProjections.index
            for label in basisKeep:
                labelIndex = projectionTypes.get_loc(label)
                similarityScore = sampleProjections.iloc[labelIndex]
                adjustedTrueLabel = prefix + trueLabel if prefix else trueLabel
                similarityMap[label][adjustedTrueLabel].append(similarityScore)

    return similarityMap


# Takes as input a geneMap which maps genes against directions of expression in the expected cluster, adds cutoffs required for significance
def setDEGCluster(topObject, geneMap, minSD=1):
    geneMap = {}
    for gene in geneMap:
        geneMap[gene] = {}
        geneMap[gene]["Mean"] = topObject.processedData.loc[gene].T.mean()
        geneMap[gene]["SD"] = topObject.processedData.loc[gene].T.std()
        geneMap[gene]["Threshold"] = geneMap[gene]["Mean"] + geneMap[gene]["SD"] * minSD * geneMap[gene]["Direction"]

    return geneMap


def getDEGClusterValues(topObject, geneMap, includeCriteria=None):
    geneExpressions = topObject.processedData.loc[list(geneMap.keys()), :]
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


# =========================
# Define plotting functions
# =========================

# Create bar plot of the highest projection scores for a particular sample
def plot_highest(projections, n=10, ax=None, **kwargs):
    ax = ax or plt.gca()
    projections_sorted = projections.sort_values(by=projections.columns[0])
    projections_top10 = projections_sorted.iloc[-n:]
    return projections_top10.plot.barh(ax=ax, **kwargs)


# Helper function for creating a color bar
def createColorbar(data, colormap='rocket_r'):
    cmap = plt.get_cmap(colormap)
    scalarmap = ScalarMappable(norm=plt.Normalize(min(data), max(data)),
                               cmap=cmap)
    scalarmap.set_array([])
    return cmap, scalarmap


# Helper function to set correspondonce between labels and colors
def setPalette(labels):
    palette = {}
    colors = list(sns.color_palette("bright")) + list(sns.color_palette())
    for i in range(len(labels)):
        palette[labels[i]] = colors[i]
    return palette


# Helper function to set correspondonce between labels and markers
def setMarkers(labels):
    markers = {}
    # markerList = list(Line2D.markers.keys())
    markerList = ["X", "o", "^", "s", "d", "p", "*",  "<", ">", "v", "H", "h", "D", "x", ".", ",", "1", "2", "3", "4", "+", "|", "_"]
    for i in range(len(labels)):
        markers[labels[i]] = markerList[i]
    return markers


# Gets a dict for passing in numbers of each cell type to downsampling function
def getLabelCountsMap(annotations, maxCount=None):
    labelCountsMap = {}
    valueCounts = annotations.value_counts()
    for label in set(annotations):
        count = int(valueCounts[label])
        labelCountsMap[label] = count if maxCount is None or count < maxCount else maxCount
    return labelCountsMap


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
               gene=None, plotMultiple=False, unsupervisedContour=False, supervisedContour=False, maxLabelCount=None, alternateColumn=None,
               geneExpressions=None, ax=None, figX=8, figY=8, includeCriteria=None,
               hue=None, labels=None, palette=None, markers=True, markerSize=40, legendMarkerScale=2,
               xBounds=None, yBounds=None, plotInRow=False, seed=0,
               axisFontSize=15, legendFontSize=13, title="", outFile=None, show=True):

    # Filter labeling columns
    annotations = topObject.annotations if includeCriteria is None else topObject.annotations[includeCriteria]
    projections = topObject.projections[projectionName] if includeCriteria is None else topObject.projections[projectionName].loc[:, includeCriteria]
    geneExpressions = None if not gene else topObject.processedData if includeCriteria is None else topObject.processedData.loc[:, includeCriteria]
    alternateColumnValues = None if not alternateColumn else topObject.metadata[alternateColumn] if includeCriteria is None else topObject.metadata[alternateColumn][includeCriteria]
    
    # Undersample some cell types based on a count maximum
    if maxLabelCount is not None:
        rus = RandomUnderSampler(sampling_strategy=getLabelCountsMap(annotations, maxCount=maxLabelCount), random_state=seed)
        projectionsUntransposed, annotations = rus.fit_resample(projections.T, annotations)
        projections = projectionsUntransposed.T

    # Set axes and key parameters for current plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(figX, figY))
    x, y = (projections.loc[celltype1], projections.loc[celltype2])
    labels = labels if labels is not None else [str(label) for label in set(annotations)]

    # Create core plots
    if gene:  # If labeling by gene expression instead of source labels
        ax = geneExpressionPlot(ax, x, y, gene, geneExpressions, annotations, palette, 
                markerSize=markerSize, alpha=0.5, axisFontSize=axisFontSize, plotMultiple=plotMultiple)
    elif alternateColumn:
        ax = alternateColumnPlot(ax, x, y, alternateColumn, alternateColumnValues, annotations, palette, 
                markerSize=markerSize, alpha=0.5, axisFontSize=axisFontSize, plotMultiple=plotMultiple)
    else:  # If labeling is by cell type
        ax = sourceLabelPlot(ax, x, y, annotations, labels, 
                palette=palette, markers=markers, markerSize=markerSize, legendMarkerScale=legendMarkerScale, axisFontSize=axisFontSize, legendFontSize=legendFontSize, plotMultiple=plotMultiple)

    # Add contours if desired
    ax = unsupervisedContourPlot(ax, x, y) if unsupervisedContour else ax
    ax = supervisedContourPlot(ax, x, y, annotations, labels, palette=palette) if supervisedContour else ax

    # Set plot visual tools
    ax.tick_params(axis='both', which='major', labelsize=axisFontSize // 1.4)
    ax.axvline(x=0.1, color='black', linestyle='--', linewidth=0.7, dashes=(5, 10))
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=0.7, dashes=(5, 10))

    # Set plot dimensions
    if xBounds is not None:
        ax.set_xlim(xBounds[0], xBounds[1])
    if yBounds is not None:
        ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_xlabel(projectionName + " " + celltype1, fontsize=axisFontSize)
    ax.set_ylabel(projectionName + " " + celltype2, fontsize=axisFontSize)

    # Set text for plot
    if title != "":
        plt.title(title, fontsize=axisFontSize // 0.9)

    if plotMultiple:
        ax.set_aspect('equal')
    else:
        plt.tight_layout()
        if outFile is not None:
            plt.savefig(outFile, bbox_inches='tight')
        if show:
            plt.show()
    return ax


# Craetes a Seaborn 2D scatter plot using projections onto basis columns as axes and gene expressions to identify points. Helper for plotTwo
def geneExpressionPlot(ax, x, y, gene, geneExpressions, annotations, palette, markerSize=40, alpha=0.5, plotMultiple=False, axisFontSize=15):
    palette, scalarmap = createColorbar(geneExpressions.loc[gene]) if palette is None else (palette, None)
    plot = sns.scatterplot(x=x, y=y, hue=geneExpressions.loc[gene], palette=palette, alpha=alpha, 
                           ax=ax, s=markerSize, style=annotations)
    if not plotMultiple:
        cbar = plt.colorbar(scalarmap, ax=ax) # scalarmap won't be defined unless palette wasn't, which is only in single plot
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        cbar.set_label('{} expression'.format(gene), size=axisFontSize, labelpad=20)

    plot.legend_.remove()
    return ax


# Craetes a Seaborn 2D scatterplot using projections onto basis columns as axes and source labels to identify points
def sourceLabelPlot(ax, x, y, annotations, labels=None, palette=None, markers=True, markerSize=40, legendMarkerScale=2, axisFontSize=16, legendFontSize=16, alpha=0.5, plotMultiple=False):
    plot = sns.scatterplot(x=x, y=y, alpha=alpha, ax=ax, hue=annotations, style=annotations,
                           hue_order=labels, style_order=labels, markers=markers, s=markerSize, palette=palette)
    if plotMultiple:
        plot.legend_.remove()
    else:
        ax.legend(title="Source Cell Labels", title_fontsize=legendFontSize // 0.9, fontsize=legendFontSize, markerscale=legendMarkerScale, loc="upper right")
    return ax


# Craetes a Seaborn 2D scatter plot using projections onto basis columns as axes and gene expressions to identify points. Helper for plotTwo
def alternateColumnPlot(ax, x, y, alternateColumn, alternateColumnValues, annotations, palette, markerSize=40, alpha=0.5, plotMultiple=False, axisFontSize=15):
    palette, scalarmap = createColorbar(alternateColumnValues) if palette is None else (palette, None)
    plot = sns.scatterplot(x=x, y=y, hue=alternateColumnValues, palette=palette, alpha=alpha, 
                           ax=ax, s=markerSize, style=annotations)
    if not plotMultiple:
        cbar = plt.colorbar(scalarmap, ax=ax) # scalarmap won't be defined unless palette wasn't, which is only in single plot
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        cbar.set_label('{}'.format(alternateColumn), size=axisFontSize, labelpad=20)

    plot.legend_.remove()
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
        sns.kdeplot(ax=ax, x=xLabel, y=yLabel, fill=True, color=palette[label], alpha=0.3, label=label)
        sns.kdeplot(ax=ax, x=xLabel, y=yLabel, color=palette[label], label=label)

    return ax


# Plot multiple 2D similarity plots at once based on some field, such as time (Note: figure out difference if any between labels and annotations[toInclude])
def plotTwoMultiple(topObject, projectionName, celltype1, celltype2,
                    annotations=None, subsetCategory=None, subsetNames=None, gene=None, includeCriteria=None,
                    unsupervisedContour=False, supervisedContour=False, maxLabelCount=None, alternateColumn=None, seed=None,
                    xBounds=None, yBounds=None, plotInRow=False, axisFontSize=24, legendFontSize=24,
                    legendMarkerScale=0.5, markerSize=40, titleFontSize=36, title="", caption=None, outFile=None):

    # Initialize categories
    annotations = topObject.annotations if includeCriteria is None else topObject.annotations[includeCriteria]
    subsetCategory = topObject.metadata[topObject.timeColumn] if subsetCategory is None else subsetCategory
    subsetNames = topObject.timesSorted if subsetNames is None else subsetNames

    # Get subplots
    subsetCount = len(subsetNames)
    subsetCountWithLegend = subsetCount + 1
    dimX = subsetCountWithLegend if plotInRow else math.ceil(math.sqrt(subsetCountWithLegend))
    dimY = 1 if plotInRow else math.ceil(subsetCountWithLegend / dimX)
    fig = plt.figure(figsize=(dimX * 12, dimY * 12), constrained_layout=True)
    gs = GridSpec(dimY, dimX, figure=fig)

    # Set up label colors and shapes
    labels = [str(label) for label in set(annotations)]
    labelMarkerMap = setMarkers(labels)

    if gene:
        geneExpressions = topObject.processedData
        palette, scalarmap = createColorbar(geneExpressions.loc[gene, includeCriteria])
    elif alternateColumn:
        palette, scalarmap = createColorbar(topObject.metadata[alternateColumn][includeCriteria])
    else:
        palette = setPalette(labels)
        legendItems = []
        for label in labels:
            legendItems.append(Line2D([0], [0], marker=labelMarkerMap[label], color="w", label=label,
               markerfacecolor=palette[label], markersize=markerSize))

    # Plot for each subset
    subsetIdx = 0
    for i in range(dimY):
        for j in range(dimX):
            if subsetIdx > subsetCount:
                break
            ax = fig.add_subplot(gs[i, j])
            if subsetIdx == subsetCount:
                ax.axis("off")  # Hide this axis
                if not (gene or alternateColumn):
                    ax.legend(handles=legendItems, title="Source Labels", title_fontsize=axisFontSize, fontsize=legendFontSize, markerscale=legendMarkerScale, loc='upper left', frameon=False)
                break

            subset = subsetNames[subsetIdx]
            toInclude = np.logical_and(subsetCategory == subset, includeCriteria) if includeCriteria is not None else subsetCategory == subset
            ax = plotTwo(
                topObject, projectionName, celltype1, celltype2,
                ax=ax, labels=labels, gene=gene, geneExpressions=geneExpressions if gene else None, alternateColumn=alternateColumn, unsupervisedContour=unsupervisedContour, supervisedContour=supervisedContour,
                plotMultiple=True, plotInRow=plotInRow, maxLabelCount=maxLabelCount, xBounds=xBounds, yBounds=yBounds, seed=seed, includeCriteria=toInclude,
                palette=palette, markers=labelMarkerMap, markerSize=markerSize, legendMarkerScale=legendMarkerScale, axisFontSize=axisFontSize, legendFontSize=legendFontSize
            )
            ax.set_xlabel(celltype1, fontsize=axisFontSize)
            ax.set_ylabel(celltype2, fontsize=axisFontSize)
            if not plotInRow:
                ax.set_title(subset, fontsize=axisFontSize)

            subsetIdx += 1

    if gene:
        cbar = fig.colorbar(scalarmap, label='{} expression'.format(gene), ax=ax)
        cbar.ax.tick_params(labelsize=legendFontSize // 1.3)
        cbar.set_label('{} expression'.format(gene), size=legendFontSize, labelpad=20)
    elif alternateColumn:
        cbar = fig.colorbar(scalarmap, label='{}'.format(alternateColumn), ax=ax)
        cbar.ax.tick_params(labelsize=axisFontSize // 1.3)
        cbar.set_label('{}'.format(alternateColumn), size=axisFontSize, labelpad=20)

    if caption:
        caption = (
            caption
        )
        wrappedCaption = "\n".join(textwrap.wrap(caption, width=170))
        fig.text(0, -0.05, wrappedCaption, ha='left', va='bottom', fontsize=axisFontSize // 1.1)
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


# Create a boxplot of the similarities between any number of sources and a basis
def similarityBoxplot(similarityMap, sourceKeep=None, basisKeep=None, groupLengths=None, labelIdxStartMap=None, title="", titleFontSize=36, labelFontSize=18, xLabelRotation=90, showOutliers=True, figY=8, outFile=None):
    # If no specific columns provided, use all columns of similarity map
    basisKeep = basisKeep or sorted(similarityMap.keys())
    sourceKeep = sourceKeep or sorted(set(list(similarityMap.values())[0]))
    numGroups = len(sourceKeep)  # Number of groups
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
    colors = list(sns.color_palette("bright")) + list(sns.color_palette())
    medianlineprops = dict(linewidth=1.5, color='black')  # If you don't want a median line comment this out

    # Plot each set of boxplots
    for i in range(numGroups):
        trueLabel = sourceKeep[i]
        for j in range(boxesPerGroup):
            label = basisKeep[j]
            if trueLabel in similarityMap[label]:
                currentBoxWidth = widths[i]
                pos = i + j * currentBoxWidth - (groupWidth / 2) + currentBoxWidth / 2  # Offset positions
                # pos = i + j * currentBoxWidth - (groupWidth) + currentBoxWidth / 2  # Offset positions
                if showOutliers:
                    bp = ax.boxplot(similarityMap[label][trueLabel], positions=[pos], widths=currentBoxWidth, patch_artist=True, medianprops=medianlineprops, boxprops={'edgecolor': 'black'}, flierprops=dict(marker='o', markersize=2, markerfacecolor='black', fillstyle='full')) #, showmeans=True, meanline=True)
                else:
                    bp = ax.boxplot(similarityMap[label][trueLabel], positions=[pos], widths=currentBoxWidth, patch_artist=True, medianprops=medianlineprops, boxprops={'edgecolor': 'black'}, showfliers=False)

                for box in bp['boxes']:
                    box.set(facecolor=colors[j])
            else:
                print("True label not found")

    # Labels
    ax.set_xticks(range(numGroups))
    if labelIdxStartMap is not None:
        individualLabels = []
        indices = list(labelIdxStartMap.keys())
        nextIndex = indices[1]
        currentSource = labelIdxStartMap[0]
        sourcesUsed = 1
        for i in range(numGroups):
            if i >= nextIndex:
                sourcesUsed += 1
                currentSource = labelIdxStartMap[nextIndex]
                nextIndex = indices[sourcesUsed] if sourcesUsed < len(indices) else 999999999
            individualLabels.append(currentSource + "\n" + sourceKeep[i])
    else:
        individualLabels = sourceKeep

    # Add vertical lines to separate groups
    y_min, y_max = ax.get_ylim()
    for i in range(1, numGroups):  # Skip first category
        group_border = i - 0.5  # Position of separator between groups
        ax.vlines(x=group_border, ymin=y_min, ymax=y_max,
                  color='black', linestyle='solid', linewidth=1)

    ax.set_xticklabels(individualLabels, fontsize=labelFontSize, rotation=xLabelRotation)
    ax.tick_params(labelsize=labelFontSize * 0.7)
    ax.set_xlabel("Source Labels", weight="bold", fontsize=labelFontSize)
    ax.set_ylabel("Similarity Scores", weight="bold", fontsize=labelFontSize)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=0.5, dashes=(5, 10))
    # fig.subplots_adjust(right=13.7)
    ax.legend([plt.Rectangle((0,0),1,1,facecolor=c) for c in colors[:boxesPerGroup]], 
               basisKeep, loc="upper left", title="Basis Labels", title_fontsize=labelFontSize*0.9, fontsize=labelFontSize*0.9, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    fig.suptitle(title, fontsize=titleFontSize)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()


# Display correlation matrix
def plotBasisCorrelationMatrix(topObject, figX=8, figY=8, textSize=8, title="Basis Column Correlations", outFile=None):
    plt.subplots(1, 1, figsize=(figX, figY))

    if topObject.basis is None:
        topObject.setBasis()
    corrCopy = topObject.corr.copy() if hasattr(topObject, "corr") else topObject.getBasisCorrelations()
    # corrCopy = topObject.basis.corr()
    if type(corrCopy) is dict:
        corrCopy = pd.DataFrame.from_dict(corrCopy, orient="index")
    labels = sorted(topObject.basis.columns)
    sns.heatmap(corrCopy, annot=True, fmt=".2f", cmap='plasma', xticklabels=labels, yticklabels=labels,
            annot_kws={"size": textSize}, cbar=True)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()

# Display correlation matrix
def plotBasisTestConfusionMatrix(topObject, figX=8, figY=8, textSize=5, axisFontSize=18, title="Basis Test Confusion Matrix", outFile=None):
    plt.subplots(1, 1, figsize=(figX, figY))

    # Build confusion matrix and set colors according to the normalized rows
    cm = confusion_matrix(topObject.testResults[1]["True"], topObject.testResults[1]["Top1"])
    colors = preprocessing.normalize(cm, axis=1)
    labels = sorted(list(set(topObject.testResults[1]["True"] + topObject.testResults[1]["Top1"])))

    # Plot the result
    sns.heatmap(colors, annot=cm, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels, annot_kws={"size": axisFontSize // 1.1})
    plt.xticks(fontsize=axisFontSize // 1.2)
    plt.yticks(fontsize=axisFontSize // 1.2)
    plt.xlabel('Predicted Label', fontsize=axisFontSize)
    plt.ylabel('True Label', fontsize=axisFontSize)
    plt.title(title, fontsize=axisFontSize)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()


# Get test accuracies
def getTestAccuracies(topObject):
    accuracyMap = {"Top1": {}, "Top3": {}}
    for label in set(topObject.testResults[1]["True"]):
        accuracyMap["Top1"][label] = []
        accuracyMap["Top3"][label] = []

    for i in range(topObject.testResults[0]["Total test count"]):
        trueLabel = topObject.testResults[1]["True"][i]
        accuracyMap["Top1"][trueLabel].append(int(trueLabel == topObject.testResults[1]["Top1"][i]))
        accuracyMap["Top3"][trueLabel].append(int(topObject.testResults[1]["Top3"][i]))

    top1Accuracies = []
    top3Accuracies = []
    for label in accuracyMap["Top1"]:
        top1Matches = accuracyMap["Top1"][label]
        top1Accuracies.append(sum(top1Matches) / len(top1Matches))
        top3Matches = accuracyMap["Top3"][label]
        top3Accuracies.append(sum(top3Matches) / len(top3Matches))

    return pd.DataFrame({"Top 1 Accuracy": top1Accuracies, "Top 3 Accuracy": top3Accuracies}, index=list(accuracyMap["Top1"].keys()))


# Volcano plot of most significant genes to scTOP predictions for a cell type
def plotPredictivity(topObject, label, basis=None, showHigh=10, sourceLabel=None, figX=8, figY=8, title="", outFile=None):
    fig, ax = plt.subplots(1, 1, figsize=(figX, figY))
    
    predictivity = topObject.getBasisPredictivity(specificBasis=basis) if basis is not None else topObject.predictivity
    genes = topObject.predictivity.loc[label]
    # genesWithZeroes = topObject.processedData.loc[:, topObject.annotations == label].reindex(genes.index, fill_value=0) # Fill missing genes
    genesWithZeroes = topObject.processedData.reindex(genes.index, fill_value=0) # Fill missing genes
    if sourceLabel is not None:
        genesWithZeroes = genesWithZeroes.loc[:, topObject.annotations == sourceLabel]
    averageExpressions = genesWithZeroes.loc[genes.index].T.mean()
    ax.scatter(averageExpressions, genes, color='gray', alpha=0.6, label='All Genes', s=3)

    # Sort expected_genes to get the top values
    if type(showHigh) is int and showHigh > 0: 
        subsetCategory, subsetName = (topObject.annotations, sourceLabel) if sourceLabel is not None else (None, None)
        scoreContributions = topObject.scoreContributions if hasattr(topObject, "scoreContributions") and sourceLabel is None and basis is None else topObject.getScoreContributions(specificBasis=basis, subsetCategory=subsetCategory, subsetName=subsetName)
        highContributions = scoreContributions[label].mean(axis=1).sort_values(ascending=False).head(showHigh).index
        xHigh = averageExpressions[highContributions]
        yHigh = genes.get(highContributions).values
        ax.scatter(xHigh, yHigh, color='blue', label='Top ' + str(showHigh) + ' Genes', s=3)

        # Annotate top 10 genes
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
