set# File containing TopObject class used for loading AnnData objects and performing core scTOP operations
# Author: Eitan Vilker (with some functions written by Maria Yampolskaya)

import numpy as np
import pandas as pd
import sctop as top
import sys
sys.path.append('/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Vilker_Helper_Files/scTOP')
import scanpy as sc
import anndata as ad
from pybiomart import Server
import mygene
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.stats as sps
import os
os.environ['SCIPY_ARRAY_API'] = '1'
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
import inspect
from copy import deepcopy
import os


class TopObject:
    def __init__(self, name, manualInit=False, useAverage=False, skipProcess=False, keep=None, exclude=None, dataset="/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Vilker_Helper_Files/scTOP/DatasetInformation.csv"):
        self.name = name
        self.dataset = dataset
        self.processedData = None
        if self.dataset is not None:
            datasetInfo = pd.read_csv(self.dataset, index_col="Name", keep_default_na=False).loc[self.name, :]
            self.cellTypeColumn, self.toKeep, self.toExclude, self.filePath, self.timeColumn, self.species, self.duplicates, self.raw, self.layer, self.comments = datasetInfo
            self.raw = getTruthValue(self.raw)
            self.duplicates = getTruthValue(self.duplicates)
            self.toKeep = self.toKeep[1:-1].replace("'", "").split(", ") if type(self.toKeep) is str and len(self.toKeep) > 0 else self.toKeep
            self.toExclude = self.toExclude[1:-1].replace("'", "").split(", ") if type(self.toExclude) is str and len(self.toExclude) > 0 else self.toExclude
            if not manualInit:  # In case you want to adjust any of the parameters first
                self.setup(useAverage=useAverage, skipProcess=skipProcess, keep=keep, exclude=exclude)

        self.projections = {}
        self.basis = None
        self.combinedBases = {}
        self.PCAs = {}
        self.PCABases = {}

    # Summary of parameters upon printing object
    def __str__(self):

        # Initialize structures
        members = inspect.getmembers(self)
        attributeMap = next(val for val in members if val[0] == '__dict__')[1]
        keys = attributeMap.keys()
        toReturn = "Attributes:"

        # Iterate over each property in this TopObject and simplify for printing
        for key in keys:

            # Get actual value as well as data type in order to classify how to handle each property
            value = attributeMap[key]
            valueType = type(value)

            # Depending on data type and size, change display behavior
            if (valueType is str and value != "") or valueType is bool or valueType is np.bool:  # String, displayed as is
                toReturn += "\n" + key + ": " + str(value)
            elif valueType is list or valueType is np.ndarray:  # List, truncated
                if len(value) > 0 and type(value[0]) is np.ndarray:
                    value = value[0]
                toReturn += "\n" + key + ": [" + ", ".join(list(value[:5])) + "]..." if len(value) > 5 else "\n" + key + ": " + str(value)
            elif valueType is pd.core.frame.DataFrame:  # Pandas DataFrame, truncated
                toReturn += "\n" + key + ": " + str(list(value.columns)[:5]) + "..."
            elif valueType is pd.core.series.Series:  # Pandas Series, truncated
                toReturn += "\n" + key + ": " + str(list(value)[:5]) + "..."
            elif value is None or (valueType is str and value == ""):  # None or empty string, displayed as "None"
                toReturn += "\n" + key + ": None"
            else:
                toReturn += "\n" + key + ": " + str(valueType)  # Other, displayed as type only
        return toReturn

    # Copy the object into a new instance. Somewhat slow operation
    def copy(self):
        print("Copying...")
        copy = deepcopy(self)
        copy.setAnndata(self.anndata.copy())
        print("Done!")
        return copy

    # Initialize AnnData object, metadata, df, and process it
    def setup(self, useAverage=False, skipProcess=False, keep=False, exclude=False):

        # Load AnnData (h5ad) object
        if not hasattr(self, "anndata"):
            print("Setting AnnData object...")
            annObject = sc.read_h5ad(self.filePath)

            if self.duplicates and self.duplicates != "Other":
                print("Making variable names unique...")
                annObject.var_names_make_unique()
        
        # Set and do basic filtering for AnnData object and associated metadata, df
        self.setAnndata(annObject)
        self.filter(keep=keep, exclude=exclude)

        # Process (2-step normalize) data if desired, preserving rest of object if this fails due to memory limits
        if not skipProcess:
            try:
                self.process(useAverage=useAverage)
            except:
                print("Processing dataset for scTOP failed! The rest of the TopObject has been preserved.")
        print("Finished setup!")

    # Set key features. May need to be called whenever object is edited
    def setMetadata(self, cellTypeColumn=None):
        cellTypeColumn = self.cellTypeColumn if cellTypeColumn is None else cellTypeColumn
        self.metadata = self.anndata.obs
        self.annotations = self.metadata[cellTypeColumn]
        self.sortedCellTypes = sorted([cellType for cellType in set(self.annotations) if type(cellType) is str and cellType != "nan"])
        self.timeSortFunction = None
        self.timesSorted = None
        if self.timeColumn is not None and self.timeColumn != "":
            self.timeSortFunction = lambda time: int("".join([char for char in time if char.isdigit()])) # if numbers in string unrelated to time this won't work
            self.timesSorted = sorted([str(time) for time in set(self.metadata[self.timeColumn]) if time != "nan"], key=self.timeSortFunction)

    # Set df, with a few extra options in case there are issues with the df
    def setDF(self, layer=None):
        if self.raw and self.raw != "Other":  # Check to use raw data
            print("Using raw data...")
            self.df = pd.DataFrame(self.anndata.raw.X.toarray(), index=self.metadata.index, columns=self.anndata.raw.var_names).T

        else:
            layer = self.layer if getTruthValue(layer) != "Other" and getTruthValue(self.layer) == "Other" else None  # Check to use layer other than default
            self.df = self.anndata.to_df(layer=layer).T # Create the DataFrame from the counts of the AnnData object

        if self.duplicates and self.duplicates != "Other":  # Check if there are duplicate genes requiring consolidating by measure such as mean
            self.df = self.df.drop_duplicates().groupby(level=0).mean()

    # Set anndata object along with associated objects
    def setAnndata(self, annObject, skipProcess=True):
        print("Setting AnnData...")
        self.anndata = annObject
        print("Setting metadata and df...")
        self.setMetadata()
        try:
            self.setDF()
        except:
            print("Unable to allocate sufficient memory for df!")
            return
        if not skipProcess:
            self.process()
        elif self.processedData is not None:
            self.processedData = self.processedData.loc[:, self.processedData.columns.isin(self.df.columns)]
            self.processedData /= np.linalg.norm(self.processedData, axis=0, keepdims=True)

    # Set TopObject to include or exclude cells with certain labels. Not for gene filtering!
    def filter(self, keep=None, exclude=None, condition=None, maxSamples=None, conditionList=None, # Sample conditions
               skipProcess=True, useAverage=False, seed=1):

        # Begin filtering if at least one condition was selected
        keepTruthValue = getTruthValue(keep)
        excludeTruthValue = getTruthValue(exclude)
        if keepTruthValue or excludeTruthValue or condition is not None or maxSamples is not None or conditionList is not None:
            print("Filtering TopObject...")

            # Select individual conditions
            conditionList = [] if conditionList is None else conditionList
            if keepTruthValue:  # Filter to include only annotation categories specified. Keep can be specified list or default for TopObject
                conditionList.append(self.annotations.isin(self.toKeep if keepTruthValue != "Other" else keep))
            if excludeTruthValue:  # Filter to include all annotation categories except those specified. Exclude can be specified list or default for TopObject
                conditionList.append(~self.annotations.isin(self.toExclude if excludeTruthValue != "Other" else exclude))
            if condition is not None:  # Filter to any given condition
                conditionList.append(condition)

            # Combine all conditions except undersampling for single pass, though basis and projections must be done again
            if len(conditionList) > 0:
                combinedCondition = conditionList[0]
                for i in range(1, len(conditionList)):
                    combinedCondition = np.logical_and(combinedCondition, conditionList[i])
                self.setAnndata(self.anndata[combinedCondition])

            # Undersample some cell types based on a count maximum. Must be performed after other conditions
            if maxSamples is not None:
                rus = RandomUnderSampler(sampling_strategy=getLabelCountsMap(self.annotations, maxCount=maxSamples), random_state=seed)
                df, annotations = rus.fit_resample(self.df.T, self.annotations)
                self.setAnndata(self.anndata[self.df.columns.isin(df.index)])

            # Process data as needed
            if not skipProcess:
                self.process()
            elif self.processedData is not None:
                self.processedData = self.processedData.loc[:, self.processedData.columns.isin(self.df.columns)]
            del conditionList
            print("Finished filtering!")

    # First scTOP function, ranks and normalizes source 
    def process(self, useAverage=False, chunks=500):
        print("Processing scTOP data...")
        self.processedData = top.process(self.df, average=useAverage, chunk_size=chunks)
        return self.processedData
        print("Finished processing!")

    # Main scTOP function, computing similarity between labels in sources and basis
    def project(self, basis, projectionName, pca=None, alignGenes=False, normalize=False, returnOverlap=False):
        print("Projecting onto basis...")
        if alignGenes:
            self.setAnndata(self.anndata[:, self.df.index.isin(basis.index)])
        if self.processedData is None or alignGenes:
            self.process()
        if pca is not None:
            processedData = self.processedData.loc[pca.feature_names_in_, :].T
            processedPCA = pd.DataFrame(pca.transform(processedData), index=processedData.index)
            projection = top.score(basis, processedPCA.T)
            overlap = "N/A"
        else:
            overlap = np.intersect1d(basis.index, self.processedData.index)
            if normalize:
                processedAligned = self.processedData.loc[overlap, :]
                basisAligned = basis.loc[overlap, :]
                processedAligned /= np.linalg.norm(processedAligned, axis=0, keepdims=True)
                basisAligned /= np.linalg.norm(basisAligned, axis=0, keepdims=True)
                # processedAligned = top.process(processedAligned, average=False, chunk_size=500)
                projection = top.score(basisAligned, processedAligned)
            else:
                projection = top.score(basis, self.processedData)

        self.projections[projectionName] = projection
        print("Finished projecting! " + str(len(overlap)) + " genes were in both the source and basis.")
        if returnOverlap:
            return projection, overlap
        return projection

    # Using any dataset with well-defined clusters, set it as a basis
    def setBasis(self, holdouts=None, threshold=200, seed=None, getScores=False, usePCA=False, allowedGenes=None, basisName=None, useProcessed=False, includeCriteria=None):
        print("Setting basis...")

        # Set and filter data that will form basis
        processedData = self.process() if self.processedData is None and usePCA else None # Process dataset if not done yet and needed for PCA
        cellData = self.processedData if usePCA or useProcessed else self.df
        cellData = cellData.loc[cellData.index.isin(allowedGenes), :] if allowedGenes is not None else cellData
        cellData = cellData.loc[:, includeCriteria] if includeCriteria is not None else cellData
        annotations = self.annotations[self.df.columns.isin(cellData.columns)]

        # Using fewer than 150-200 cells leads to nonsensical results, due to noise. More cells -> less sampling error
        typeCounts = annotations.value_counts()
        typesAboveThreshold = typeCounts[typeCounts > threshold].index
        basisList = []
        trainingIDs = []

        # Set structures for PCA basis if using
        if usePCA:
            print("Performing PCA...")
            if basisName is None:
                print("Must input a basis name!")
                return None
            self.PCAs[basisName] = PCA(100)
            cellData = pd.DataFrame(self.PCAs[basisName].fit_transform(cellData.T), index=cellData.columns).T

        # Process each cell type individually
        rng = np.random.default_rng(seed=seed)
        for cellType in tqdm(typesAboveThreshold):
            cellIDs = cellData.loc[:, annotations == cellType].columns
            if holdouts is not None:
                holdouts = 0.2 if type(holdouts) is bool else holdouts
                currentIDs = rng.choice(cellIDs, size=int(len(cellIDs) * (1 - holdouts)), replace=False)
            else:
                currentIDs = cellIDs
            currentCellData = cellData.loc[:, currentIDs]
            trainingIDs += [currentIDs] # Keep track of training_IDs so that you can exclude them if you want to test the accuracy

            # Average across the cells and process them using the scTOP processing method
            processed = top.process(currentCellData, average=True, chunk_size=500) if not usePCA else currentCellData.mean(axis=1)
            basisList += [processed]

        # Merge cell types into single basis
        trainingIDs = np.concatenate(trainingIDs)
        basis = pd.concat(basisList, axis=1)
        basis.columns = typesAboveThreshold
        basis.index.name = "gene"
        print("Basis set!")

        # If testing basis quality, use holdouts and train/test data
        if holdouts is not None and holdouts:
            return basis, trainingIDs
        if usePCA:
            self.PCABases[basisName] = basis
        self.basis = basis

        # Get statistics
        self.getBasisCorrelations()
        self.getBasisPredictivity()
        if getScores:
            self.getScoreContributions()
        return basis

    # Add the desired columns of one basis to another
    def combineBases(self, otherBasis, firstKeep=None, firstExclude=None, secondKeep=None, secondExclude=None, alternateFirstBasis=None, name="Combined"):
        print("Combining bases...")
        
        # Get and set basis for this object as needed
        basis1 = alternateFirstBasis if alternateFirstBasis is not None else self.basis
        basis1 = self.setBasis() if basis1 is None else basis1

        # Get basis to be combined with
        basis2 = otherBasis if not isinstance(otherBasis, TopObject) else otherBasis.basis
        basis2 = otherBasis.setBasis() if basis2 is None else basis2
        basis2.index.name = basis1.index.name = "gene"

        # Filter bases
        basis1 = basis1[firstKeep] if firstKeep is not None else basis1
        basis1 = basis1[[col for col in basis1.columns if col not in firstExclude]] if firstExclude is not None else basis1
        basis2 = basis2[secondKeep] if secondKeep is not None else basis2
        basis2 = basis2[[col for col in basis1.columns if col not in secondExclude]] if secondExclude is not None else basis2

        # Combine bases
        combinedBasis = pd.merge(basis1, basis2, on=basis1.index.name, how="inner")
        self.combinedBases[name] = combinedBasis
        return combinedBasis

    # Test an existing basis (not combined). Optionally adjust the minimum accuracy threshold
    def testBasis(self, specificationValue=0.1, holdouts=0.2, threshold=200, seed=1, includeCriteria=None):

        # Setting basis with holdouts for testing
        basis, trainingIDs = self.setBasis(holdouts=holdouts, threshold=threshold, seed=seed, includeCriteria=includeCriteria)
        sampleIDs = self.df.columns if includeCriteria is None else self.df.columns[includeCriteria]
        _, indices, _ = np.intersect1d(sampleIDs, trainingIDs, return_indices=True) # Using intersect + delete because setdiff1d has performance issues
        testIDs = np.delete(sampleIDs, indices)
        testCount = len(testIDs)
        splitIDs = np.array_split(testIDs, 10)
        print("Processing test data...")
        accuracies = {'top1': 0,
                      'top3': 0,
                      'Unspecified': 0}
        predictions = {}
        predictions["True"] = []
        predictions["Top1"] = []
        predictions["Top3"] = []

        for sampleIds in tqdm(splitIDs):
            testData = self.df[sampleIds]
            testProcessed = top.process(testData)
            testProjections = top.score(basis, testProcessed)
            accuracies, predictions = self.scoreProjections(testProjections, accuracies, predictions, specificationValue=specificationValue)
            del testData, testProcessed, testProjections
        for key, value in accuracies.items():
            print("{}: {}".format(key, value / testCount))

        accuracies["Total test count"] = testCount
        self.testResults = (accuracies, predictions)
        return self.testResults[0]

    # Get the metrics for a given projection. Optionally adjust the minimum accuracy threshold
    def scoreProjections(self, projections, accuracies, predictions, specificationValue=0.1): # cells with maximum projection under specificationValue are considered "unspecified"

        for sampleId, sampleProjections in projections.items():
            typesSortedByProjections = sampleProjections.sort_values(ascending=False).index
            trueType = self.metadata.loc[sampleId, self.cellTypeColumn]
            topType = typesSortedByProjections[0]

            if sampleProjections.max() < specificationValue:
                accuracies['Unspecified'] += 1
                topType = 'Unspecified'

            predictions["True"].append(trueType)
            predictions["Top1"].append(topType)

            if topType == trueType:
                accuracies['top1'] += 1

            inTop3 = trueType in typesSortedByProjections[:3]
            if inTop3:
                accuracies['top3'] += 1
            predictions["Top3"].append(inTop3)

        return accuracies, predictions

    # Create correlation matrix between cell types of basis, helpful to determine if any features are overlapping
    def getBasisCorrelations(self, specificBasis=None):
        basis = self.basis if specificBasis is None else specificBasis
        corr = basis.T.dot(basis) / basis.shape[0]
        if specificBasis is None:
            self.corr = corr 
        return corr

    # Create predictivity matrix to assess impact of cell type on gene expression
    def getBasisPredictivity(self, specificBasis=None):
        basis = self.basis if specificBasis is None else specificBasis
        corr = self.corr if specificBasis is None else self.getBasisCorrelations(specificBasis=basis)
        eta = np.linalg.inv(corr).dot(basis.T) / basis.shape[0]
        self.predictivity = pd.DataFrame(eta, index=basis.columns, columns=basis.index)
        return self.predictivity

    # Create score contribution matrix displaying product of predictivity and normalized expression
    def getScoreContributions(self, specificBasis=None, subsetCategory=None, subsetName=None):
        scoreContributions = {}
        labelExpression = self.processedData if hasattr(self, "processedData") else self.process()
        if subsetCategory is not None and subsetName is not None:
            labelExpression = labelExpression.loc[:, subsetCategory == subsetName]
        predictivityMatrix = self.predictivity if specificBasis is None else self.getBasisPredictivity(specificBasis=specificBasis)
        for label in predictivityMatrix.index:
            scoreContributions[label] = {}
            commonGenes = np.intersect1d(labelExpression.index, predictivityMatrix.columns)
            scoreContributions[label] = labelExpression.loc[commonGenes].multiply(predictivityMatrix.loc[label, commonGenes], axis=0)

        if subsetName is None:
            self.scoreContributions = scoreContributions
        return scoreContributions

    # Given a projection and a cell type in the basis, change annotation in source to reflect top labels of that cell type
    def newCellTypeCatFromProjection(self, projectionName, target, newCategoryName, newLabelName=None, specificationValue=0.1, target2=None):
        newLabelName = newLabelName or target
        projection = self.projections[projectionName]
        targetCells = []
        for sample in self.df.columns:
            condition = projection.loc[:, sample].idxmax() == target and projection.loc[target, sample] > specificationValue
            condition = condition if target2 is None else condition and projection.loc[target2, sample] > specificationValue
            if condition:
                targetCells.append(sample)

        self.anndata.obs[newCategoryName] = self.annData.obs[self.cellTypeColumn]
        self.anndata.obs[newCategoryName] = self.anndata.obs[newCategoryName].cat.add_categories([newLabelName])

        for sample in targetCells:
            self.anndata.obs.loc[sample, newCategoryName] = newLabelName

        self.cellTypeColumn = newCategoryName
        self.metadata = self.anndata.obs
        self.annotations = self.metadata[self.cellTypeColumn]


    # Get set of highest variance genes for which the highest accuracy was observed classifying cell types
    def getBestGenes(self, seed=0, proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], trialCount=5, trainProportion=0.8, specificationValue=0.1, batchJob=False):
        
        # Initialize data structures
        transposedDF = self.df.T
        geneCount = len(transposedDF.columns)
        proportionTestMap = {}
        for proportion in proportions:
            proportionTestMap[proportion] = {}
        trials = []
        for i in range(trialCount):
            trials.append(seed + i)
        proportionCount = len(proportions)

        # Get train and test data stratified to have same proportion of each cell type
        for trial in trials: # For each seed
            print("Trial: " + str(trial))
            rus = RandomUnderSampler(sampling_strategy=getLabelCountsMap(self.annotations, proportion=trainProportion), random_state=trial)
            trainX, trainY = rus.fit_resample(transposedDF, self.annotations)
            trainXProcessed = top.process(trainX.T, chunk_size=500)
            trainXProcessed /= np.linalg.norm(trainXProcessed, axis=0, keepdims=True)
            inTrain = transposedDF.index.isin(trainX.index)
            print("Processing 2...")
            testX = top.process(transposedDF.loc[~inTrain, :].T, chunk_size=500)
            testX /= np.linalg.norm(testX, axis=0, keepdims=True)
            testY = self.annotations[~inTrain]

            for geneProportion in proportions:  # Get subset of genes for each proportion
                print("Proportion: " + str(geneProportion))
                # Identify genes
                selector = SelectKBest(score_func=f_classif, k=int(geneProportion * geneCount))
                trainSelected = selector.fit_transform(trainXProcessed.T, trainY)
                selectedFeatures = transposedDF.columns[selector.get_support()]

                # Set basis using identified genes and training data
                basis = setBasis(trainX.T, trainY, allowedGenes=selectedFeatures, threshold=50)

                # Use model to determine efficacy of gene subsets
                projection = top.score(basis, testX)
                results = []
                for sampleId, sampleProjections in projection.items():
                    typesSortedByProjections = sampleProjections.sort_values(ascending=False).index
                    trueType = testY.loc[sampleId]
                    topType = typesSortedByProjections[0]
                    results.append(int(trueType == topType and sampleProjections.max() > specificationValue))
                
                accuracy = sum(results) / len(results)
                if batchJob:
                    proportionTestMap[geneProportion] = [accuracy, list(selectedFeatures)]
                else:
                    proportionTestMap[geneProportion][trial] = [accuracy, selectedFeatures]
                del selector, trainSelected, selectedFeatures, basis, projection, results
            del trainX, trainY, testX, testY, trainXProcessed, rus, inTrain
        del transposedDF

        if batchJob:
            return proportionTestMap
        return bestGenesAnalysis(proportionTestMap, proportions, trials)     

    # Get ortholog genes based on Ensembl mapping to another species
    def getOrthologs(self, mapping, inplace=False):
        
        # Get genes with orthologs to other species in mapping
        print("Finding orthologs...")
        mg = mygene.MyGeneInfo()
        testGenes = mg.querymany(mapping.index, field='symbol', size=1)
        basisGenes = mg.querymany(mapping.values, field='symbol', size=1)

        # Get symbols of genes
        symbolMapping = pd.DataFrame(data={
            "test": [val['symbol'] if 'symbol' in val.keys() else val['query'] for val in testGenes], 
            "basis": [val['symbol'] if 'symbol' in val.keys() else val['query'] for val in basisGenes]})

        # Filter out genes without orthologs and set names side by side
        print("Filtering AnnData object for orthologs...")
        validMap = symbolMapping[symbolMapping["test"].isin(self.df.index)]
        orthologAligned = self.anndata[:, validMap['test']].copy()
        orthologAligned.var_names = validMap['basis']
        
        # Return, dropping duplicates if multiple target genes mapped to one basis gene
        orthologAligned = orthologAligned[:, ~orthologAligned.var_names.duplicated()].copy()
        if inplace:
            self.setAnndata(orthologAligned)
        print("Done!")
        return orthologAligned


# Gets a dict for passing in numbers of each cell type to downsampling function
def getLabelCountsMap(annotations, maxCount=None, proportion=None):
    labelCountsMap = {}
    valueCounts = annotations.value_counts()
    for label in set(annotations):
        count = int(valueCounts[label])
        labelCountsMap[label] = count if maxCount is None or count < maxCount else maxCount
    if proportion:
        for label in labelCountsMap.keys():
            labelCountsMap[label] = int(labelCountsMap[label] * proportion)
    return labelCountsMap


# Add or update an entry to a summary file containing metadata regarding datasets
def addDataset(summaryFile, name, filePath=None, cellTypeColumn=None, toKeep=None, toExclude=None, timeColumn=None, species=None, duplicates=None, raw=None, layer=None, comments=None):
    summaryFileInfo = pd.read_csv(summaryFile, index_col="Name", keep_default_na=False)
    possibleEntries = [cellTypeColumn, toKeep, toExclude, filePath, timeColumn, species, duplicates, raw, layer, comments]
    entryCount = len(possibleEntries)
    alreadyPresent = name in summaryFileInfo.index
    newEntry = summaryFileInfo.loc[name, :].copy() if alreadyPresent else pd.Series([None] * entryCount)

    for i in range(entryCount):
        entry = possibleEntries[i]
        if entry is not None and entry != "":
            newEntry.iat[i] = entry
    newEntry.name = name
    newEntry = pd.DataFrame(newEntry).T
    if alreadyPresent:
        summaryFileInfo.update(newEntry)
    else:
        newEntry.columns = summaryFileInfo.columns
        summaryFileInfo = pd.concat([summaryFileInfo, newEntry], ignore_index=False)
    print("Writing updated file...")
    summaryFileInfo.to_csv(summaryFile, index_label="Name")
    return summaryFileInfo


# Remove entry from dataset
def deleteDataset(summaryFile, name):
    summaryFileInfo = pd.read_csv(summaryFile, index_col="Name", keep_default_na=False)
    summaryFileInfo = summaryFileInfo.drop([name])
    summaryFileInfo.to_csv(summaryFile, index_label="Name")
    return summaryFileInfo


# User-friendly way to update or add entry to the summary file
def dynamicAddDataset(summaryFile=None):
    try:
        if summaryFile is None:
            summaryFile = processInput("Enter the file path of a csv containing entries formatted for scTOP:", isFile=True, isRequired=True)
        name = processInput("Assign name. Enter a name to describe your dataset (if updating existing entry, choose the same name):", isRequired=True)
        filePath = processInput("Assign filePath. Enter the file path corresponding to the anndata object (.h5ad) for your dataset:", isFile=True)
        cellTypeColumn = processInput("Assign cellTypeColumn. Enter the title of the column containing cell type annotations:")
        toKeep = processInput("Assign toKeep. Press Y to enter a list of cell types that you may filter to later or press Enter to skip:", followUpMessage="Enter a cell type to include in filtering or press Enter to continue", isList=True)
        toExclude = processInput("Assign toExclude. Press Y to enter a list of cell types that you may filter out later or press Enter to skip:", followUpMessage="Enter a cell type to exclude in filtering or press Enter to continue", isList=True)
        timeColumn = processInput("Assign timeColumn. Enter the title of the column containing times samples were collected or press Enter to skip:")
        species = processInput("Assign species. Enter the (singular) name of species or press Enter to skip:")
        duplicates = processInput("Assign duplicates. Enter Y if the dataset has duplicate genes; otherwise enter N or press Enter to skip:", isBool=True)
        raw = processInput("Assign raw. Enter Y if using the raw values stored in the anndata object instead; otherwise enter N or press Enter to skip:", isBool=True)
        layer = processInput("Assign layer. Enter the name of a specific layer (typically counts, data, or scaled_data) to use or press Enter to skip:")
        comments = processInput("Assign comments. Enter any additional comments you would like or press Enter to skip:")
        addDataset(summaryFile, name, filePath=filePath, cellTypeColumn=cellTypeColumn, toKeep=toKeep, toExclude=toExclude, timeColumn=timeColumn, species=species, duplicates=duplicates, raw=raw, layer=layer, comments=comments)
    except:
        print("Quit early!")


# Checks user input and processes based on type
def processInput(message, followUpMessage=None, isFile=False, isList=False, isBool=False, isRequired=False):
    while True:
        userInput = input(message)
        truthValue = getTruthValue(userInput)

        if userInput == "Q":
            sys.exit()
        elif userInput == "":
            if isRequired:
                print("A value must be entered here")
            else:
                return None
        elif isFile:
            if os.path.exists(userInput):
                return userInput
            else:
                print("No file found at path: " + userInput)
        elif isBool:
            if type(truthValue) is bool:
                return truthValue
            else:
                print("Enter a valid true/false value")
        elif isList:
            entryList = []
            if not truthValue:
                break
            else:
                if truthValue != "Other":
                    while True:
                        entry = input(followUpMessage)
                        if entry == "Q":
                            sys.exit()
                        elif entry == "":
                            return entryList
                        entryList.append(entry)
        else:
            break

    return userInput


# Converts input to Boolean True or False, or "Other" or None if inapplicable
def getTruthValue(val):
    if val is None or val == "":
        return None
    if type(val) is bool:
        return val
    if type(val) is str:
        val = val.upper()
        if val == "Y" or val == "YES" or val == "T" or val == "TRUE":
            return True
        if val == "N" or val == "NO" or val == "F" or val == "FALSE":
            return False
        return "Other"
    return "Other"

# Using any dataset with well-defined clusters, set it as a basis
def setBasis(cellData, annotations, holdouts=None, threshold=200, seed=None, getScores=False, usePCA=False, allowedGenes=None, basisName=None):
    print("Setting basis...")
    # Count the number of cells per type
    typeCounts = annotations.value_counts()

    # Using fewer than 150-200 cells leads to nonsensical results, due to noise. More cells -> less sampling error
    typesAboveThreshold = typeCounts[typeCounts > threshold].index
    basisList = []
    trainingIDs = []
    cellData = cellData.loc[cellData.index.isin(allowedGenes), :] if allowedGenes is not None else cellData

    if usePCA:
        print("Performing PCA...")
        if basisName is None:
            print("Must input a basis name!")
            return None
        basisPCA = PCA(100)
        cellData = pd.DataFrame(basisPCA[basisName].fit_transform(cellData.T), index=cellData.columns).T

    rng = np.random.default_rng(seed=seed)
    for cellType in tqdm(typesAboveThreshold):
        cellIDs = cellData.loc[:, annotations == cellType].columns
        if holdouts is not None:
            holdouts = 0.2 if type(holdouts) is bool else holdouts
            currentIDs = rng.choice(cellIDs, size=int(len(cellIDs) * (1 - holdouts)), replace=False)
        else:
            currentIDs = cellIDs
        currentCellData = cellData.loc[:, currentIDs]
        trainingIDs += [currentIDs] # Keep track of training_IDs so that you can exclude them if you want to test the accuracy

        # Average across the cells and process them using the scTOP processing method
        processed = top.process(currentCellData, average=True, chunk_size=500) if not usePCA else currentCellData.mean(axis=1)
        basisList += [processed]

    trainingIDs = np.concatenate(trainingIDs)
    basis = pd.concat(basisList, axis=1)
    basis.columns = typesAboveThreshold
    basis.index.name = "gene"
    print("Basis set!")
    toReturn = [basis]
    if holdouts is not None and holdouts:
        toReturn.append(trainingIDs)

    if usePCA:
        toReturn.append(basisPCA)

    if len(toReturn) == 1:
        return basis
    return toReturn


def bestGenesAnalysis(proportionTestMap, proportions, trials):
    # Find proportion with highest average accuracy
    print("Finding best proportions...")
    bestProportion = 1
    bestAccuracy = 0
    geneProportionFrame = pd.DataFrame(index=[str(val) for val in proportions] + ["Average Trial Accuracy"], columns=[str(val) for val in trials] + ["Average Proportion Accuracy"])
    # return geneProportionFrame, proportions, proportionTestMap
    for geneProportion in proportions:  # For each proportion of genes
        accuracies = []
        for trial in proportionTestMap[geneProportion]:  # For each seed
            accuracies.append(proportionTestMap[geneProportion][trial][0])
        averageAccuracy = sum(accuracies) / len(accuracies)
        geneProportionFrame.loc[str(geneProportion)] = accuracies + [averageAccuracy]
        
        # Replace current best gene proportion if superior
        if averageAccuracy > bestAccuracy:
            bestProportion, bestAccuracy = (geneProportion, averageAccuracy)
    geneProportionFrame.loc["Average Trial Accuracy"] = geneProportionFrame.mean()

    # Get number of times each gene was included in the subset selected
    genesSelectedMap = {}
    for trial in proportionTestMap[bestProportion]:  # For each seed
        for gene in proportionTestMap[bestProportion][trial][1]:  # For each gene
            genesSelectedMap[gene] = 1 if gene not in genesSelectedMap.keys() else genesSelectedMap[gene] + 1

    genesSelectedFrame = pd.DataFrame.from_dict(genesSelectedMap, orient='index', columns=["Successes"])
    return genesSelectedFrame, geneProportionFrame.astype(float)


# Get EnsemblMart server for finding gene orthologs between species
def getEnsemblMart(speciesNames=["human", "mouse"]):
    potentialConfig = {
        'human':      {'dataset': 'hsapiens_gene_ensembl',  'prefix': 'hsapiens'},
        'chimpanzee': {'dataset': 'ptroglodytes_gene_ensembl','prefix': 'ptroglodytes'},
        'gorilla':    {'dataset': 'ggorilla_gene_ensembl',   'prefix': 'ggorilla'},
        'macaque':    {'dataset': 'mmulatta_gene_ensembl',   'prefix': 'mmulatta'},
        'marmoset':   {'dataset': 'cjacchus_gene_ensembl',   'prefix': 'cjacchus'},
        'mouse':      {'dataset': 'mmusculus_gene_ensembl',  'prefix': 'mmusculus'},
        'opossum':    {'dataset': 'mdomestica_gene_ensembl', 'prefix': 'mdomestica'},
        'platypus':   {'dataset': 'oanatinus_gene_ensembl',  'prefix': 'oanatinus'}
    }
    ensemblConfig = {species: potentialConfig[species] for species in speciesNames}    
    server = Server('http://www.ensembl.org')
    ensemblMart = server.marts['ENSEMBL_MART_ENSEMBL']
    return ensemblMart, ensemblConfig


# Get ortholog gene mapping between any two species, with the reference first
def getOrthologMapping(basisSpecies, targetSpecies, ensemblMart, ensemblConfig):
    """Fetches 1:1 orthologs: Target IDs -> Basis IDs."""
    if basisSpecies == targetSpecies:
        return None
    
    source_prefix = ensemblConfig[basisSpecies]['prefix']
    target_dataset_name = ensemblConfig[targetSpecies]['dataset']
    homolog_attr = f"{source_prefix}_homolog_ensembl_gene"
    
    try:
        dataset = ensemblMart.datasets[target_dataset_name]
        df = dataset.query(attributes=['ensembl_gene_id', homolog_attr], use_attr_names=True)
        df = df.dropna().drop_duplicates()
        
        df = df.drop_duplicates(subset=['ensembl_gene_id'], keep='first')
        df = df.drop_duplicates(subset=[homolog_attr], keep='first')
        
        return df.set_index('ensembl_gene_id')[homolog_attr]
    except Exception as e:
        print(f"  Warning: Could not fetch mapping for {targetSpecies}->{basisSpecies}: {e}")
        return pd.Series(dtype=str)

