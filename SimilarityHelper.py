### File containing functions for working with scTOP, made by Maria Yampolskaya and Pankaj Mehta
# Author: Eitan Vilker

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import sctop as top
import scanpy as sc
import h5py
import umap
from tqdm import tqdm
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
import anndata as ad
from scipy.sparse import csr_matrix
import hdf5plugin


# Gets AnnObject and basis df given h5ad file
def loadBasis(fileName, cellTypeColumn=None, filtering=False, toExclude=None, toInclude=None):
    print("Reading h5ad...")
    annObject = sc.read_h5ad(fileName)

    print("Filtering and making df...")
    if filtering and cellTypeColumn is not None:
        if toExclude is not None:
            annObject = annObject[~annObject.obs[cellTypeColumn].isin(toExclude)]
        if toInclude is not None:
            annObject = annObject[annObject.obs[cellTypeColumn].isin(toInclude)]

    df = pd.DataFrame(annObject.X.toarray(), index = annObject.obs.index , columns = annObject.var.index).T
    return annObject, df, cellTypeColumn


# Gets the MC-KO basis made by Michael Herriges in the Kotton Lab with only mouse lung epithelial cells
def loadMCKOBasis():
    print("Loading...")
    basis = top.load_basis("MC-KO", 50)
    cleanedBasis = basis[0]
    cleanedBasis = cleanedBasis[[colName for colName in cleanedBasis.columns if "Lung" in colName]]
    cleanedBasis.drop("Lung Endothelial Cells WK6-10 MC20", axis=1, inplace=True)
    newCols = {}
    for col in cleanedBasis.columns:
        idx = col.find("Cell")
        if idx == -1:
            idx = col.find("WK6")
        if idx != -1:
            newCols[col] = "MC-KO " + col[5:idx - 1]

    cleanedBasis = cleanedBasis.rename(columns=newCols)
    return cleanedBasis


# Function to make a basis using LungMap human data
def loadHumanBasis(filtering=True, includeRas=False):
    with h5py.File('/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/scripts/humanBasis.h5', "r") as f:
        cellTypes = f["df"]["axis0"][:]
        var = f["df"]["axis1"][:]
        X = f["df"]["block0_values"][:]

    humanBasis = pd.DataFrame(X)
    humanBasis.columns = [col.decode() for col in cellTypes]
    humanBasis.index = [row.decode() for row in var]
    newCols = {}
    for col in humanBasis.columns:
        idx = col.find("cell")
        if idx != -1:
            newCols[col] = col[:idx - 1]

    humanBasis = humanBasis.rename(columns=newCols)
    basisKeep = ["Alveolar type 2", "Alveolar type 1", "Basal", "Ciliated", "Goblet", "Secretory"]
    if includeRas:
        basisKeep.append("Respiratory airway secretory")

    if filtering:
        humanBasis = humanBasis[[colName for colName in humanBasis.columns if colName in basisKeep]]
    print("Loaded basis!")
    return humanBasis


# Function to get the AnnObject and associated objects for a test dataset, with additional handling if there is no h5ad file
def processDataset(datasetName, simplifying=False, keepAll=False, filteringAnnObject=True, buildingUp=False, normalized=False, useAverage=False):
    cell_type_column, toKeep, filePath = getDatasetSepcificInfo(datasetName)

    print("Processing dataset...")
    # If there is no h5ad, assemble AnnData objet
    if buildingUp:
        if datasetName == "Kobayashi":
            matrix = pd.read_csv(filePath + '/rawcounts.mtx')
            annObject = ad.AnnData(matrix)
        else:
            annObject = sc.read_mtx(filePath + '/rawcounts.mtx')
        annObject.var = pd.read_csv(filePath + '/genes.txt', header=None)
        annObject.obs = pd.read_csv(filePath + '/cellinfo.csv',sep="\t")
        annObject.var.columns = ["name"]
        annObject.var.set_index("name", inplace=True)
        annObject.obs.set_index("cell_barcode", inplace=True)
        annObject.obs.index.names = ["index"]
        annObject.var.index.names = ["index"]
    else:
        annObject = sc.read_h5ad(filePath)

    if filteringAnnObject and not keepAll:
        annObject = annObject[annObject.obs[cell_type_column].isin(toKeep)]
        
    print("Created Ann Object!")
        
    df = pd.DataFrame(annObject.X.toarray(), index = annObject.obs.index , columns = annObject.var.index).T
    metadata = annObject.obs
    clusters = metadata[cell_type_column].values
    if not normalized:
        processedData = top.process(df, average=useAverage)
    else:
        processedData = df
    print("Created scTOP processed data")
    annotations = setSourceAnnotations(clusters, toKeep, keepAll=keepAll, simplifying=False)
    kwargs = setArguments(annotations[annotations!='Other'])

    return annObject, df, metadata, processedData, annotations, kwargs, toKeep


# Get the info needed to load the specific test set
def getDatasetSepcificInfo(dataset):

    if dataset == "Kostas":
        filePath = "../Kostas/Kostas.h5ad"
        cell_type_column = "cell_type_epithelial_mesenchymal_final"
        toKeep = ["AT1", "AT2", "AT2 activated", "Proliferating", "Transitional epithelial"]
    elif dataset == "Riemondy":
        filePath = "../Riemondy/Riemondy.h5ad"
        cell_type_column = "labeled_clusters"
        toKeep =  ["Basal", "Injured Type II", "Naive Type I", "Naive Type II", "Transdifferentiating Type II", "Cell Cycle Arrest Type II", "Proliferating Type II"]
    elif dataset == "Strunz":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Strunz"
        cell_type_column = "cell_type"
        toKeep = ["AT2", "AT2 activated", "Krt8+ ADI", "AT1", "Basal", "Mki67+ Proliferation"]
    elif dataset == "Choi":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Choi"
        cell_type_column = "celltype_4"
        toKeep =  ["AT1", "AT2", "Primed", "Intermediate", "Cycling AT2"]
    elif dataset == "Kobayashi":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Kobayashi"
        cell_type_column = "cell_type"
        toKeep = ["AEC1", "AEC2", "Ctgf+", "AEC2-proliferating", "Lgals3+"]
    elif dataset == "Kathiriya":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Kathiriya/Kathiriya.h5ad"
        cell_type_column = "celltypes"
        toKeep = [0, 1, 2, 3, 4, 5, 6]
    elif dataset == "Habermann":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Habermann/Habermann.h5ad"
        cell_type_column = "celltype"
        toKeep = ["AT1", "AT2", "KRT5-/KRT17+", "Proliferating Epithelial Cells", "Transitional AT2", "SCGB3A2+", "SCGB3A2+ SCGB1A1+"]
    elif dataset == "Bibek":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/BibekPneumonectomy/objects/Bibek.h5ad"
        cell_type_column = "annotation_update"
        toKeep = ["AT1", "AT2", "Krt8 high AT2", "Activated AT2", "Ciliated", "Proliferating AT2", "Secretory"]
    elif dataset == "Rawlins":
        filePath = '/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Andrea/EmmaRawlinsBasis.h5ad'
        cell_type_column = 'new_celltype'
        toKeep = ["AT1", "AT2", "Krt8 high AT2", "Activated AT2", "Ciliated", "Proliferating AT2", "Secretory"]
    elif dataset == "Tsukui":
        filePath = '/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/BibekPneumonectomy/objects/Tsukui.h5ad'
        cell_type_column = "celltype"
        toKeep = []

    return cell_type_column, toKeep, filePath

# Set the labels for the given cluster
def setSourceAnnotations(clusters, keepList, keepAll=False, simplifying=False):
    sourceAnnotations = []
    for val in clusters:
        if val in keepList or keepAll:
            if simplifying:
                if val == "Naive Type I":
                    sourceAnnotations.append("AT1")
                elif val == "Injured Type II":
                    sourceAnnotations.append("AT2 activated")
                elif val == "Proliferating Type II":
                    sourceAnnotations.append("Proliferating")
                elif val == "Naive Type II":
                    sourceAnnotations.append("AT2")
                else:
                    sourceAnnotations.append(val)
            else:
                sourceAnnotations.append(val)
        else:
            sourceAnnotations.append("Other")

    return np.array(sourceAnnotations)


# Write an AnnData object to an h5ad file
def writeAnnDataObject(annDataObj, outFile):
    obj = annDataObj
    obj.X = csr_matrix(obj.X)
    # varFrame = pd.DataFrame(index=obj.var.index)
    # obj.var.index = varFrame
    # obj.var.drop(columns=obj.var.columns, inplace=True)
    if obj.raw is not None:
        obj._raw._var.rename(columns={'_index': 'index'}, inplace=True)    
        obj.raw.var.index.name(columns={'_index': 'index'}, inplace=True)    
        obj.write_h5ad(outFile)
    print("Finished!")


# Get average projections given time series data
def getTimeAveragedProjections(basis, df, cellLabels, times, timeSortFunc, substituteMap=None):

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


# Get the UMAP info
def getEmbedding(data):
    pca = PCA(100)
    PCA_data = pca.fit_transform(data.T)
    reducer = umap.UMAP()
    reducer.fit(PCA_data)
    embedding = reducer.embedding_
    return embedding


# Set  keyword arguments to re-use for some plots
def setArguments(source_annotations):

    kwargs = {'s': 40,
              'style': source_annotations
             }
    return kwargs


# Using any dataset with well-defined clusters, set it as a basis
def setBasis(basis_df, basis_metadata, cell_type_column = 'labeled_clusters'):

    # Count the number of cells per type
    type_counts = basis_metadata[cell_type_column].value_counts()

    # Using fewer than 150-200 cells leads to nonsensical results, due to noise. More cells -> less sampling error
    threshold = 200 # only use cell types with at least this many cells (but use all cells for training)
    types_above_threshold = type_counts[type_counts > threshold].index
    basis_list = []
    training_IDs = []

    rng = np.random.default_rng()

    for cell_type in tqdm(types_above_threshold):
        cell_IDs = basis_metadata[basis_metadata[cell_type_column] == cell_type].index
        current_IDs = rng.choice(cell_IDs, size=threshold, replace=False) # This line is for using only the threshold number of cells for the reference basis. This can be useful for testing the accuracy of the basis, but it performs notably worse in accuracy metrics compared to using all possible cells.
        # current_IDs = cell_IDs

        cell_data = basis_df[current_IDs]
        training_IDs += [current_IDs] # Keep track of training_IDs so that you can exclude them if you want to test the accuracy

        # Average across the cells and process them using the scTOP processing method
        processed = top.process(cell_data, average=True)
        basis_list += [processed]

    training_IDs = np.concatenate(training_IDs)
    basis = pd.concat(basis_list, axis=1)
    basis.columns = types_above_threshold
    basis.index.name = "gene"
    print("Basis set!")
    return basis


# Add the desired columns of a smaller basis to a primary basis
def combineBases(basis1, df, metadata, colsToKeep, cell_type_column='labeled_clusters'):
    basis2 = setBasis(df, metadata, cell_type_column=cell_type_column)
    basis1.index.name = basis2.index.name
    return pd.merge(basis1, basis2[colsToKeep], on=basis1.index.name, how="inner")


# First step of testing the accuracy of a basis. Trains a basis and outputs holdouts 
def testBasis1(basis_df, basis_metadata, cell_type_column='labeled_clusters'):

    # Count the number of cells per type
    type_counts = basis_metadata[cell_type_column].value_counts()

    # Using fewer than 150-200 cells leads to nonsensical results, due to noise. More cells -> less sampling error
    threshold = 200 # only use cell types with at least this many cells (but use all cells for training)
    types_above_threshold = type_counts[type_counts > threshold].index
    types_above_threshold
    basis_list = []
    training_IDs = []

    rng = np.random.default_rng()
    print("Processing cell types...")

    for cell_type in tqdm(types_above_threshold):
        cell_IDs = basis_metadata[basis_metadata[cell_type_column] == cell_type].index
        current_IDs = rng.choice(cell_IDs, size=threshold, replace=False) # This line is for using only the threshold number of cells for the reference basis. This can be useful for testing the accuracy of the basis, but it performs notably worse in accuracy metrics compared to using all possible cells.
        # current_IDs = cell_IDs

        cell_data = basis_df[current_IDs]
        training_IDs += [current_IDs] # Keep track of training_IDs so that you can exclude them if you want to test the accuracy

        # Average across the cells and process them using the scTOP processing method
        processed = top.process(cell_data, average=True)
        basis_list += [processed]

    training_IDs = np.concatenate(training_IDs)
    trainBasis = pd.concat(basis_list, axis=1)
    trainBasis.columns = types_above_threshold

    test_IDs = np.setdiff1d(basis_df.columns, training_IDs)
    # test_IDs = training_IDs
    split_IDs = np.array_split(test_IDs, 10) # I split this test dataset because it's very large and took up a lot of memory -- you don't need to do this if you have enough memory to test the entire dataset at once
    return trainBasis, test_IDs, split_IDs


# Second step of testing a basis, using the outputs of the first step. Optionally adjust the minimum accuracy threshold
def testBasis2(trainBasis, basis_df, test_IDs, split_IDs, basis_metadata, cell_type_column, specification_value=0.1):
    print("Processing test data...")
    accuracies = {'top1': 0,
                  'top3': 0,
                  'unspecified': 0
    }
    matches = {}
    misses = {}
    for sample_IDs in tqdm(split_IDs):
        test_data = basis_df[sample_IDs]
        test_processed = top.process(test_data)
        test_projections = top.score(trainBasis, test_processed)
        accuracies, matches, misses = scoreProjections(accuracies, matches, misses, test_projections, basis_metadata, cell_type_column, specification_value=specification_value)
        del test_data
        del test_processed
        del test_projections
    for key, value in accuracies.items():
        print("{}: {}".format(key, value/len(test_IDs)))

    return accuracies, matches, misses


# Get the metrics for a given projection. Optionally adjust the minimum accuracy threshold
def scoreProjections(accuracies, matches, misses, projections, metadata, cell_type_column, specification_value=0.1):
    # cells with maximum projection under this value are considered "unspecified"

    predicted_labels = []
    predicted_labels_specified = []
    true_labels = []
    matches = {}
    misses = {}
    print("Scoring projection...")
    for sample_id, sample_projections in projections.items():
        types_sorted_by_projections = sample_projections.sort_values(ascending=False).index
        true_type = metadata.loc[sample_id, cell_type_column]
        true_labels += [true_type]
        top_type = types_sorted_by_projections[0]
        predicted_labels += [top_type]

        if sample_projections.max() < specification_value:
            predicted_labels_specified += ['Unspecified']
            accuracies['unspecified'] += 1
        else:
            predicted_labels_specified += [top_type]

        if top_type == true_type:
            accuracies['top1'] += 1
            if true_type not in matches:
                matches[true_type] = 1
            else:
                matches[true_type] += 1
        else:
            if true_type not in misses:
                misses[true_type] = 1
            else:
                misses[true_type] += 1
        if true_type in types_sorted_by_projections[:3]:
            accuracies['top3'] += 1

    return accuracies, matches, misses


# Get dict of cell types in the source to the counts of cell types in the basis they were most similar to
def getTopPredictedMap(projections, metadata, cell_type_column="cell_type"):
    topPredictedMap = {}
    for sample_id, sample_projections in projections.items():
        types_sorted_by_projections = sample_projections.sort_values(ascending=False).index
        true_type = metadata.loc[sample_id, cell_type_column]
        top_type = types_sorted_by_projections[0]
        # print(sample_id + ", " + true_type)
        if true_type not in topPredictedMap:
            topPredictedMap[true_type] = []
        topPredictedMap[true_type].append(top_type)
    return topPredictedMap


# Get a dict of cell types in basis to the similarity scores of each type in the source
def getMatchingProjections(projections, metadata, cell_type_column, basisKeep, sourceKeep, prefix=None):

    similarityMap = {}
    for label in basisKeep:
        similarityMap[label] = {}

    for trueLabel in sourceKeep:
        for label in similarityMap:
            if prefix is not None:
                adjustedTrueLabel = prefix + trueLabel
            else:
                adjustedTrueLabel = trueLabel
            similarityMap[label][adjustedTrueLabel] = []

    for sample_id, sample_projections in projections.items():
        trueLabel = metadata.loc[sample_id, cell_type_column]

        if trueLabel in sourceKeep:
            projectionTypes = sample_projections.index
            for label in basisKeep:
                labelIndex = projectionTypes.get_loc(label)
                similarityScore = sample_projections.iloc[labelIndex]
                if prefix is not None:
                    adjustedTrueLabel = prefix + trueLabel
                else:
                    adjustedTrueLabel = trueLabel
                similarityMap[label][adjustedTrueLabel].append(similarityScore)

    return similarityMap


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
def create_colorbar(data, label, colormap='rocket_r', ax = None):
    ax = ax or plt.gca()
    cmap = plt.get_cmap(colormap)
    scalarmap = ScalarMappable(norm=plt.Normalize(min(data), max(data)),
                               cmap=cmap)
    scalarmap.set_array([])
    plt.colorbar(scalarmap, label=label, ax = ax)
    return cmap


# Create scatter plot showing projections of each cell in a UMAP plot, for a given cell type
def plot_UMAP(projections, embedding, cell_type, ax=None, **kwargs):
    ax = ax or plt.gca()
    type_projections = np.array(projections.loc[cell_type]).T
    palette = create_colorbar(type_projections, 'Projection onto {}'.format(cell_type),
                             ax = ax)
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


# Create scatter plot showing projection scores for two cell types, with the option to
# color according to marker gene
def plot_two(projections, celltype1, celltype2,
             gene=None, gene_expressions=None, ax=None, 
             title=None, hue=None, labels=None, palette=None, markers=None, 
             minSimilarity=-0.1, maxSimilarity=0.4, legendFontSize=16, **kwargs):

    ax = ax or plt.gca()
    x = projections.loc[celltype1]
    y = projections.loc[celltype2]
    
    if gene:
        palette = create_colorbar(gene_expressions.loc[gene],
                                  '{} expression'.format(gene), ax=ax)

        plot = sns.scatterplot(x=x,
                               y=y,
                               hue=gene_expressions.loc[gene],
                               palette=palette,
                               alpha=0.5,
                               ax=ax,
                               **kwargs
                               )
        plot.legend_.remove()

    else:

        if palette is None or markers is None:
            plot = sns.scatterplot(x=x, y=y, alpha=0.5, ax=ax, hue=hue, **kwargs)
        else:
            plot = sns.scatterplot(x=x, y=y, alpha=0.5, ax=ax, hue=hue, hue_order=labels, style_order=labels, markers=markers, palette=palette, **kwargs)
        ax.legend(title="Source Labels", title_fontsize=legendFontSize, fontsize=legendFontSize, loc="upper right")
        
    if title is not None:
        ax.set_title(title)
    ax.axvline(x=0.1, color='black', linestyle='--', linewidth=0.5, dashes=(5, 10))
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=0.5, dashes=(5, 10))
    ax.set_xlim(minSimilarity, maxSimilarity)
    ax.set_ylim(minSimilarity, maxSimilarity)

    return sns.color_palette()


# Plot multiple 2D similarity plots at once based on some field, such as time
def plot_two_multiple(projections, celltype1, celltype2, annotations, subsetCategory, subsetNames, legendFontSize=16, minSimilarity=-0.1, maxSimilarity=0.4, 
                      gene=None, sourceData=None):

    # Get subplots
    subsetCount = len(subsetNames)
    dimX = math.ceil(math.sqrt(subsetCount))
    dimY = math.ceil(subsetCount / dimX)
    fig, axes = plt.subplots(dimX, dimY, figsize=(8 * dimX, 8 * dimY), layout="constrained")

    # Set up label colors and shapes
    colors = list(sns.color_palette("bright")) + list(sns.color_palette())
    markers = ["X", "o", "^", "s", "d", "p", "*",  "<", ">", "v", "H", "h", "D", "x", ".", ",", "1", "2", "3", "4", "+", "|", "_"]
    labels = [str(label) for label in set(annotations) if label != "Other"]
    labelColorMap = {}
    labelMarkerMap = {}
    for i in range(len(labels)):
        labelColorMap[labels[i]] = colors[i]
        labelMarkerMap[labels[i]] = markers[i]

    # Plot for each subset
    for i, ax in enumerate(axes.flat):
        if i >= subsetCount: 
            fig.delaxes(ax)
            continue
        subset = subsetNames[i]
        toInclude = np.logical_and(annotations!='Other', subsetCategory==subset)
        # toInclude = subsetCategory==subset
        if sourceData is not None:
            geneExpressions = sourceData.loc[:, toInclude]
        else:
            geneExpressions = None
        plot_two(
                projections.loc[:, toInclude],
                celltype1, celltype2,
                ax=ax,
                hue=annotations[toInclude],
                title=subset,
                labels=labels,
                palette=labelColorMap, markers=labelMarkerMap,
                legendFontSize = legendFontSize,
                minSimilarity = minSimilarity, maxSimilarity = maxSimilarity,
                s=60, 
                style=annotations[toInclude],
                gene=gene, gene_expressions=geneExpressions
        )

    return fig, axes


# 3D Similarity plot
def plot_three(df, x_label, y_label, z_label, names, figureTitle="Similarity Plot", legendTitle="Source Annotations"):
    # fig.add_trace(go.Scatter3d(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9], mode='markers', name='Group A'))
    nameSet = set(names)
    colorMapping = {}
    i = 0
    for name in nameSet:
        colorMapping[name] = i
        i += 1
        
    fig = go.Figure()
    for name in nameSet:
        filteredProjections = df.loc[:, names==name]
        x = filteredProjections.loc[x_label, :]
        y = filteredProjections.loc[y_label, :]
        z = filteredProjections.loc[z_label, :]
        
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5, color=colorMapping[name]),
            name=name,
            hovertemplate= x_label + ": %{x:.4f}<br>"+ y_label +": %{y:.4f}<br>"+ z_label +": %{z:.4f}<extra></extra>"
        ))
    
    fig.update_layout(
        scene=dict(
            # xaxis=dict(range=[0, 0.4]),
            # yaxis=dict(range=[0, 0.4]),
            # zaxis=dict(range=[0, 0.4]),
            aspectmode="cube",  # Ensures all axes look the same width
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ),
        title=figureTitle,
        width=800,
        height=800,
        # legend_title_font=dict(size=16, color="blue"),
        legend=dict(
            title=legendTitle,
            x=1, # x-coordinate of the legend (0 is left, 1 is right)
            y=1, # y-coordinate of the legend (0 is bottom, 1 is top)
            orientation='v', # 'h' for horizontal, 'v' for vertical
            xanchor='right', # 'auto', 'left', 'center', 'right'
            yanchor='top' # 'auto', 'top', 'middle', 'bottom'
        )
    )

    fig.add_trace(go.Scatter3d(
        x=[0.1, 0.1],
        y=[-0.2, 0.5],
        z=[0, 0],
        mode='lines',
        line=dict(dash='dash', width=5, color='red'),
        name=x_label + "=0.1"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0.1, 0.1],
        z=[-0.2, 0.5],
        mode='lines',
        line=dict(dash='dash', width=5, color='red'),
        name=y_label + "=0.1"
    ))
    fig.add_trace(go.Scatter3d(
        x=[-0.2, 0.5],
        y=[0, 0],
        z=[0.1, 0.1],
        mode='lines',
        line=dict(dash='dash', width=5, color='red'),
        name=z_label + "=0.1"
    ))

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


# Create similarity plot with two populations shown on the same axes 
def compare_populations(ax, celltype1, celltype2, scores_dict):

    color_dict = {}
    potentialColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    i = 0
    for label in scores_dict:
        color_dict[label] = potentialColors[i]
        i += 1

    # celltype1 = 'Alveolar type 1 cell'
    # celltype2 = 'Goblet cell'

    for key, projections in scores_dict.items():
        sns.scatterplot(x=projections.loc[celltype1],
                        y=projections.loc[celltype2],
                        color=color_dict[key],
                        alpha=0.25,
                        s=40,
                        ax=ax
                       )

    for key, projections in scores_dict.items():

        scores1 = projections.loc[celltype1]
        scores2 = projections.loc[celltype2]

        sns.kdeplot(x=scores1,
                    y=scores2,
                    fill=True,
                    color=color_dict[key],
                    alpha=0.2,
                    label=key,
                    ax=ax
                   )
        sns.kdeplot(x=scores1,
                    y=scores2,
                    color=color_dict[key],
                    alpha=0.6,
                    label=key,
                    ax=ax
                   )

    # ax.set_xlim(-0.2, 0.65)
    # ax.set_ylim(-0.2, 0.65)
    ax.set_xlabel(celltype1)
    ax.set_ylabel(celltype2)

    legend_elements = [Patch(facecolor=color, edgecolor=color, alpha=0.6,
                             label=key) for key, color in color_dict.items()]

    plt.legend(handles=legend_elements)


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
def similarityBoxplot(ax, trueLabels, basisLabels, similarityMap, groupLengths=None, labelIdxStartMap=None):
    num_groups = len(trueLabels)  # Number of groups
    num_boxes_per_group = len(basisLabels)  # Number of boxplots per group

    # Define positions for each group
    widths = []
    group_width = 0.8  # Controls how wide the groups are
    if True:#groupLengths is None:
        box_width = group_width / num_boxes_per_group  # Width of each boxplot
        for i in range(len(trueLabels)):
            widths.append(box_width)
    else:
        for groupLength in groupLengths:
            widths.append(group_width / groupLength)

    # Colors for each boxplot within a group
    # colors = ['skyblue', 'goldenrod', 'lightgreen', 'plum', 'blanchedalmond', 'lavender', 'salmon', 'crimson', 'midnightblue', 'maroon']
    colors = list(sns.color_palette("bright")) + list(sns.color_palette())

    medianlineprops = dict(linewidth=1.5, color='black')

    # Plot each set of boxplots
    for i in range(num_groups):
        trueLabel = trueLabels[i]
        # trueLabelCount = len(similarityMap[label].keys())
        for j in range(num_boxes_per_group):
            label = basisLabels[j]
            # print(trueLabel)
            # print(label)
            # # # # print(list(similarityMap[label].keys()))
            # print(colors[j] + "\n")
            if trueLabel in similarityMap[label]:
                currentBoxWidth = widths[i]
                pos = i + j * currentBoxWidth - (group_width / 2) + currentBoxWidth / 2  # Offset positions
                bp = ax.boxplot(similarityMap[label][trueLabel], positions=[pos], widths=currentBoxWidth, patch_artist=True, medianprops=medianlineprops, boxprops={'edgecolor': 'black'}) #, showmeans=True, meanline=True)
                for box in bp['boxes']:
                    box.set(facecolor=colors[j])
            else:
                print("True label not found")

    # Labels
    ax.set_xticks(range(num_groups))
    if labelIdxStartMap is not None:
        individualLabels = []
        indices = list(labelIdxStartMap.keys())
        nextIndex = indices[1]
        currentSource = labelIdxStartMap[0]
        sourcesUsed = 1
        for i in range(num_groups):
            if i >= nextIndex:
                sourcesUsed += 1
                currentSource = labelIdxStartMap[nextIndex]
                if sourcesUsed < len(indices):
                    nextIndex = indices[sourcesUsed]
                else:
                    nextIndex = 999999999
            individualLabels.append(currentSource + "\n" + trueLabels[i])

    else:
        individualLabels = trueLabels

    # Add vertical lines to separate groups
    y_min, y_max = ax.get_ylim()
    for i in range(1, num_groups):  # Skip first category
        group_border = i - 0.5  # Position of separator between groups
        ax.vlines(x=group_border, ymin=y_min, ymax=y_max, 
                  color='black', linestyle='dashed', linewidth=1)

    ax.set_xticklabels(individualLabels, fontsize="large", rotation=45)
    ax.set_xlabel("Source Labels", weight="bold", fontsize="large")
    ax.set_ylabel("Similarity Scores", weight="bold", fontsize="large")
    ax.legend([plt.Rectangle((0,0),1,1,facecolor=c) for c in colors[:num_boxes_per_group]], 
               basisLabels, loc="upper right", title="Basis Labels")

