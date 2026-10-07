# File containing TopObject class used for loading AnnData objects and performing core scTOP operations
# Author: Eitan Vilker (with some functions written or inspired by Maria Yampolskaya and Huan Souza)

import numpy as np
import pandas as pd
import sctop as top
import sys
import scanpy as sc
import anndata as ad
from pybiomart import Server
import mygene
from scipy.sparse import csr_matrix
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


# Object-oriented structure for containing AnnData objects, scTOP processed data and projections, relevant functions, and integration with visualization and statistical tools
class TopObject:
    def __init__(self, identifier, annObject=None, cellTypeColumn=None, manualInit=False, useAverage=False, skipProcess=False, keep=None, exclude=None, maxSamples=None, keepFull=[], datasetCollection="/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/OutsidePaperObjects/DatasetCollection.csv"):
        self.identifier = identifier
        self.datasetCollection = datasetCollection
        self.processed = None

        if annObject is not None:
            if cellTypeColumn is None:
                print("You must enter a value for cellTypeColumn")
            else:
                self.anndata, self.cellTypeColumn = (annObject, cellTypeColumn)
                self.toKeep = self.toExclude = self.filePath = self.timeColumn = self.species = self.duplicates = self.raw = self.layer = self.comments = None
                self.setup(useAverage=useAverage, skipProcess=skipProcess, keep=keep, exclude=exclude, maxSamples=maxSamples, keepFull=keepFull)
        elif self.datasetCollection is not None:
            dataset = pd.read_csv(self.datasetCollection, index_col="Name", keep_default_na=False).loc[self.identifier, :]
            self.cellTypeColumn, self.toKeep, self.toExclude, self.filePath, self.timeColumn, self.species, self.duplicates, self.raw, self.layer, self.comments = dataset
            self.cellTypeColumn = cellTypeColumn or self.cellTypeColumn
            self.raw = getTruthValue(self.raw)
            self.duplicates = getTruthValue(self.duplicates)
            self.toKeep = self.toKeep[1:-1].replace("'", "").split(", ") if type(self.toKeep) is str and len(self.toKeep) > 0 else self.toKeep
            self.toExclude = self.toExclude[1:-1].replace("'", "").split(", ") if type(self.toExclude) is str and len(self.toExclude) > 0 else self.toExclude
            if not manualInit:  # In case you want to adjust any of the parameters first
                self.setup(useAverage=useAverage, skipProcess=skipProcess, keep=keep, exclude=exclude, maxSamples=maxSamples, keepFull=keepFull)

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
        copy.setAnndata(self.anndata.copy(), df=self.df.copy())
        print("Done!")
        return copy

    # Initialize AnnData object, metadata, df, and process it
    def setup(self, useAverage=False, skipProcess=False, keep=False, exclude=False, maxSamples=None, keepFull=[]):

        # Load AnnData (h5ad) object
        if not hasattr(self, "anndata"):
            print("Setting AnnData object...")
            annObject = sc.read_h5ad(self.filePath)

            if self.raw and self.raw != "Other":  # Check to use raw data
                annObject = ad.AnnData(X=annObject.raw.X, obs=annObject.obs, var=annObject.raw.var, uns=annObject.uns)

            if self.duplicates and self.duplicates != "Other":
                print("Making variable names unique...")
                annObject.var_names_make_unique()
        else:
            annObject = self.anndata
        
        # Set and do basic filtering for AnnData object and associated metadata, df
        self.setAnndata(annObject, keep=keep, exclude=exclude, maxSamples=maxSamples, keepFull=keepFull)

        # Check if there are duplicate genes and consolidate by measure such as mean
        if self.duplicates and self.duplicates != "Other":
            print("Consolidating duplicates...")
            self.df = self.df.drop_duplicates().groupby(level=0).mean()
            self.anndata = ad.AnnData(X=csr_matrix(self.df.T), obs=self.metadata, var=pd.DataFrame(self.df.index, index=self.df.index), uns=self.anndata.uns)

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
            self.timeSortFunction = lambda time: int("".join([char for char in time if char.isdigit()]) or 0) # if numbers in string unrelated to time this won't work
            self.timesSorted = sorted([str(time) for time in set(self.metadata[self.timeColumn]) if time != "nan"], key=self.timeSortFunction)

    # Set df, with a few extra options in case there are issues with the df
    def setDF(self, layer=None):
        layer = self.layer if getTruthValue(layer) != "Other" and getTruthValue(self.layer) == "Other" else None  # Check to use layer other than default
        self.df = self.anndata.to_df(layer=layer).T # Create the DataFrame from the counts of the AnnData object
        return self.df

    # Set anndata object along with associated objects
    def setAnndata(self, annObject, skipProcess=True, keep=None, exclude=None, maxSamples=None, keepFull=[], df=None):
        print("Setting AnnData...")
        self.anndata = annObject
        print("Setting metadata and df...")
        self.setMetadata()
        if not self.filter(keep=keep, exclude=exclude, maxSamples=maxSamples, keepFull=keepFull):
            try:
                self.df = self.setDF() if df is None else df
            except:
                print("Unable to allocate sufficient memory for df!")
                return
        if not skipProcess:
            self.process()
        elif self.processed is not None and len(self.processed.index) >= len(self.df.index):
            self.processed = self.processed.loc[self.processed.index.isin(self.df.index), self.processed.columns.isin(self.df.columns)]
            self.processed.index = self.df.index

    # Set TopObject to include or exclude cells with certain labels. Not for gene filtering! Return True if filtering occurred
    def filter(self, keep=None, exclude=None, condition=None, maxSamples=None, keepFull=[], conditionList=None,
               skipProcess=True, useAverage=False, seed=0, annotations=None, df=None, inplace=True):
        
        annotations = self.annotations if annotations is None else annotations
        filtered = False
        
        # Begin filtering if at least one condition was selected
        keepTruthValue = False if type(keep) is bool and not getTruthValue(self.toKeep) else getTruthValue(keep)
        excludeTruthValue = False if type(exclude) is bool and not getTruthValue(self.toExclude) else getTruthValue(exclude)

        if keepTruthValue or excludeTruthValue or condition is not None or maxSamples is not None or conditionList is not None:
            filtered = True

            # Select individual conditions
            conditionList = [] if conditionList is None else conditionList
            if keepTruthValue:  # Filter to include only annotation categories specified. Keep can be specified list or default for TopObject
                conditionList.append(annotations.isin(self.toKeep if type(keep) is bool else keep))
            if excludeTruthValue:  # Filter to include all annotation categories except those specified. Exclude can be specified list or default for TopObject
                conditionList.append(~annotations.isin(self.toExclude if type(exclude) is bool else exclude))
            if condition is not None:  # Filter to any given condition
                conditionList.append(condition)

            # Combine all conditions except undersampling for single pass, though basis and projections must be done again
            if len(conditionList) > 0:
                combinedCondition = conditionList[0]
                for i in range(1, len(conditionList)):
                    combinedCondition = np.logical_and(combinedCondition, conditionList[i])
                annotations = annotations[combinedCondition]

            # Undersample some cell types based on a count maximum. Must be performed after other conditions
            if maxSamples is not None:
                annotations = downsample(annotations, maxSamples, keepFull=keepFull, seed=seed)

        if not (inplace and df is None):
            df = self.df if df is None else df
            return df.loc[:, annotations.index], annotations
        if not filtered:
            return False

        # Process data as needed
        self.setAnndata(self.anndata[annotations.index])
        if not skipProcess:
            self.process()
        elif self.processed is not None:
            self.processed = self.processed.loc[:, self.processed.columns.isin(self.df.columns)]
        print("Finished filtering!")
        return True

    # First scTOP function, ranks and normalizes source 
    def process(self, useAverage=False, chunks=500, df=None, annotations=None, maxSamples=None, keepFull=[], condition=None, setProcessed=True, seed=1):
        df = self.df if df is None else df
        df, annotations = self.filter(annotations=annotations, df=df, condition=condition, maxSamples=maxSamples, keepFull=keepFull, seed=seed, inplace=False)
        
        print("Processing scTOP data...")
        processed = top.process(df, average=useAverage, chunk_size=chunks)
        if setProcessed:
            self.processed = processed
        else:
            return processed, annotations
        print("Done processing!")
        return processed

    # Main scTOP function, computing similarity between labels in sources and basis
    def project(self, basis, projectionName, pca=None, alignGenes=False, normalize=False, returnOverlap=False):
        print("Projecting onto basis...")
        if alignGenes:
            self.setAnndata(self.anndata[:, self.df.index.isin(basis.index)])
        if self.processed is None or alignGenes:
            self.process()
        if pca is not None:
            processed = self.processed.loc[pca.feature_names_in_, :].T
            processedPCA = pd.DataFrame(pca.transform(processed), index=processed.index)
            projection = top.score(basis, processedPCA.T)
            overlap = "N/A"
        else:
            overlap = np.intersect1d(basis.index, self.processed.index)
            if normalize:
                processedAligned = self.processed.loc[overlap, :]
                basisAligned = basis.loc[overlap, :]
                processedAligned /= np.linalg.norm(processedAligned, axis=0, keepdims=True)
                basisAligned /= np.linalg.norm(basisAligned, axis=0, keepdims=True)
                # processedAligned = top.process(processedAligned, average=False, chunk_size=500)
                projection = top.score(basisAligned, processedAligned)
            else:
                projection = top.score(basis, self.processed)

        self.projections[projectionName] = projection
        print("Finished projecting! " + str(len(overlap)) + " genes were in both the source and basis.")
        if returnOverlap:
            return projection, overlap
        return projection

    # Using any dataset with well-defined clusters, set it as a basis
    def setBasis(self, holdouts=None, threshold=200, seed=1, getScores=False, usePCA=False, allowedGenes=None, basisName=None, useProcessed=False, includeCriteria=None, maxSamples=None, annotationColumn=None):
        print("Setting basis...")

        # Set and filter data that will form basis
        processed = self.process() if self.processed is None and usePCA else None # Process dataset if not done yet and needed for PCA
        cellData = self.processed if usePCA or useProcessed else self.df
        cellData = cellData.loc[cellData.index.isin(allowedGenes), :] if allowedGenes is not None else cellData
        # cellData = cellData.loc[:, includeCriteria] if includeCriteria is not None else cellData
        annotations = self.annotations if annotationColumn is None else self.metadata[annotationColumn]
        # annotations = annotations[self.df.columns.isin(cellData.columns)]
        holdouts = 0.2 if type(holdouts) is bool and holdouts else holdouts

        # Undersample some cell types based on a count maximum
        # if maxSamples is not None:
        maxSamples = maxSamples if holdouts is None or maxSamples is None else int(maxSamples / (1 - holdouts)) # Adjust based on removed holdouts so amount removed is as specified
        cellData, annotations = self.filter(df=cellData, annotations=annotations, condition=includeCriteria, maxSamples=maxSamples, seed=seed, inplace=False)
            # rus = RandomUnderSampler(sampling_strategy=getLabelCountsMap(annotations, maxCount=maxSamples), random_state=seed)
            # samples, annotations = rus.fit_resample(np.array(annotations.index).reshape(-1, 1), annotations)
            # cellData = cellData.loc[:, [sample[0] for sample in samples]]
            # annotations.index = cellData.columns

        # Using fewer than 150-200 cells leads to nonsensical results, due to noise. More cells -> less sampling error
        typeCounts = annotations.value_counts()
        typesAboveThreshold = typeCounts[typeCounts >= threshold].index
        basisList = []
        trainingIDs = []

        # Set structures for PCA basis if using
        if usePCA:
            print("Performing PCA...")
            if basisName is None:
                print("Must input a basis name!")
                return None
            cellData = pd.DataFrame(self.PCAs[basisName].fit_transform(cellData.T), index=cellData.columns).T

        # Process each cell type individually
        rng = np.random.default_rng(seed=seed)
        for cellType in tqdm(typesAboveThreshold):
            cellIDs = cellData.loc[:, annotations == cellType].columns
            currentIDs = cellIDs if holdouts is None else rng.choice(cellIDs, size=int(len(cellIDs) * (1 - holdouts)), replace=False)
            currentCellData = cellData.loc[:, currentIDs]
            trainingIDs += [currentIDs] # Keep track of trainingIDs so that you can exclude them if you want to test the accuracy

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
    def combineBases(self, otherBasis, firstKeep=None, firstExclude=None, secondKeep=None, secondExclude=None, alternateFirstBasis=None, name="Combined", labelOrigin=False):
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

        if not set(basis1.columns).isdisjoint(set(basis2.columns)) or labelOrigin:
            basis1.columns = [self.identifier + " " + col for col in basis1.columns]
            basis2.columns = [otherBasis.identifier + " " + col if isinstance(otherBasis, TopObject) else col for col in basis2.columns]

        # Combine bases
        combinedBasis = pd.merge(basis1, basis2, on=basis1.index.name, how="inner")
        self.combinedBases[name] = combinedBasis
        return combinedBasis

    # Test an existing basis (not combined). Optionally adjust the minimum accuracy or sample counts thresholds
    def testBasis(self, specificationValue=0.1, holdouts=0.2, threshold=200, seed=1, includeCriteria=None, annotationColumn=None, allowedGenes=None, 
                  maxBasisSamples=None, maxTestSamples=None, trialCount=1):
        accuracies = {'top1': 0, 'top3': 0, 'Unspecified': 0}
        predictions = {"True": [], "Top1": [], "Top3": []}
        seed0 = seed
        threshold = threshold if maxBasisSamples is None or threshold < maxBasisSamples else maxBasisSamples - 1

        # Multiple trials if desired
        for i in range(trialCount):
            print("Trial: " + str(i + 1))
            
            # Setting basis with holdouts for testing
            basis, trainingIDs = self.setBasis(holdouts=holdouts, threshold=threshold, seed=seed, includeCriteria=includeCriteria, allowedGenes=allowedGenes, maxSamples=maxBasisSamples, annotationColumn=annotationColumn)
            IDs = self.df.columns if includeCriteria is None else self.df.columns[includeCriteria]
            _, indices, _ = np.intersect1d(IDs, trainingIDs, return_indices=True) # Using intersect + delete because setdiff1d has performance issues
            testIDs = np.delete(IDs, indices)
    
            # Undersample some cell types based on a count maximum
            if maxTestSamples is not None:
                annotations = self.annotations if annotationColumn is None else self.metadata[annotationColumn]
                annotations = annotations[testIDs]
                rus = RandomUnderSampler(sampling_strategy=getLabelCountsMap(annotations, maxCount=maxTestSamples), random_state=seed)
                samples, annotations = rus.fit_resample(np.array(annotations.index).reshape(-1, 1), annotations)
                testIDs = [sample[0] for sample in samples]
        
            # Predict labels for subsets of test IDs for efficiency
            print("Processing test data...")
            splitIDs = np.array_split(testIDs, 10)
            for currentIDs in tqdm(splitIDs):
                currentProcessed = top.process(self.df[currentIDs])
                currentProjections = top.score(basis, currentProcessed)
                accuracies, predictions = self.scoreProjections(currentProjections, accuracies, predictions, specificationValue=specificationValue, annotationColumn=annotationColumn)
                del currentProcessed, currentProjections

            seed = 101 * seed0 * i + 9 * (i + seed0 + 1) + seed0  # Arbitrary function to prevent collisions in seed numbers

        # Output results summary
        testCount = len(testIDs) * trialCount
        for key, value in accuracies.items():
            print("{}: {}".format(key, value / testCount))

        # Save results to TopObject
        accuracies["Total test count"] = testCount
        self.testResults = (accuracies, predictions)
        return self.testResults[0]

    # Get the metrics for a given projection. Optionally adjust the minimum accuracy threshold
    def scoreProjections(self, projections, accuracies, predictions, specificationValue=0.1, annotationColumn=None): # cells with maximum projection under specificationValue are considered "unspecified"

        annotationColumn = annotationColumn or self.cellTypeColumn
        for sampleId, sampleProjections in projections.items():
            typesSortedByProjections = sampleProjections.sort_values(ascending=False).index
            trueType = self.metadata.loc[sampleId, annotationColumn]
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
    def getBasisCorrelations(self, basis=None, metric=None):
        basisCopy = self.basis if basis is None else basis
        if metric is None or metric == "dot":
            corr = basisCopy.T.dot(basisCopy)
        elif metric == "pearson":
            corr = basisCopy.corr()
        else:
            print("Enter valid metric!")
            return None
        if basis is None:
            self.corr = corr 
        return corr

    # Create predictivity matrix to assess impact of cell type on gene expression
    def getBasisPredictivity(self, basis=None):
        basisCopy = self.basis if basis is None else basis
        corr = self.corr if basis is None else self.getBasisCorrelations(basis=basis)
        eta = np.linalg.inv(corr).dot(basisCopy.T)
        predictivity = pd.DataFrame(eta, index=basisCopy.columns, columns=basisCopy.index)
        self.predictivity = predictivity if basis is None else None
        return predictivity

    # Create score contribution matrix displaying product of predictivity and normalized expression
    def getScoreContributions(self, basis=None, subsetCategory=None, subsetName=None, includeCriteria=None):

        # Get label expressions
        scoreContributions = {}
        predictivityMatrix = self.predictivity if basis is None else self.getBasisPredictivity(basis=basis)
        geneExpressions = self.processed if includeCriteria is None else self.processed.loc[:, includeCriteria]
        # geneExpressions = geneExpressions if subsetCategory is None or subsetName is None else geneExpressions.loc[:, subsetCategory == subsetName]

        # Multiply predictivity by expression for each basis label
        for label in predictivityMatrix.index:
            scoreContributions[label] = {}
            commonGenes = np.intersect1d(geneExpressions.index, predictivityMatrix.columns)
            scoreContributions[label] = geneExpressions.loc[commonGenes].multiply(predictivityMatrix.loc[label, commonGenes], axis=0)

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

        self.anndata.obs[newCategoryName] = self.anndata.obs[self.cellTypeColumn]
        self.anndata.obs[newCategoryName] = self.anndata.obs[newCategoryName].cat.add_categories([newLabelName])

        for sample in targetCells:
            self.anndata.obs.loc[sample, newCategoryName] = newLabelName

        self.cellTypeColumn = newCategoryName
        self.metadata = self.anndata.obs
        self.annotations = self.metadata[self.cellTypeColumn]


    # Get set of highest variance genes for which the highest accuracy was observed classifying cell types
    def getBestGenes(self, seed=0, proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], trialCount=5, trainProportion=0.8, specificationValue=0.1, batchJob=False):
        
        # Initialize data structures
        geneCount = len(self.df.index)
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
            trainX, trainY = downsample(self.annotations, None, df=self.df, proportion=trainProportion, seed=trial)

            trainXProcessed = top.process(trainX, chunk_size=500)
            trainXProcessed /= np.linalg.norm(trainXProcessed, axis=0, keepdims=True)
            inTrain = self.df.columns.isin(trainX.columns)
            print("Processing again...")
            testX = top.process(self.df.loc[:, ~inTrain], chunk_size=500)
            testX /= np.linalg.norm(testX, axis=0, keepdims=True)
            testY = self.annotations[~inTrain]

            for geneProportion in proportions:  # Get subset of genes for each proportion
                print("Proportion: " + str(geneProportion))
                # Identify genes
                selector = SelectKBest(score_func=f_classif, k=int(geneProportion * geneCount))
                trainSelected = selector.fit_transform(trainXProcessed.T, trainY)
                selectedFeatures = self.df.index[selector.get_support()]

                # Set basis using identified genes and training data
                basis = setBasis(trainX, trainY, allowedGenes=selectedFeatures, threshold=50)

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
            del trainX, trainY, testX, testY, trainXProcessed, inTrain

        if batchJob:
            return proportionTestMap
        return bestGenesAnalysis(proportionTestMap, proportions, trials)

    # Filter dataset to genes that maximize accurate identification of cell types given a proportion
    def filterBestGenes(self, proportion, processed=None, startFromDF=False, inplace=True, includeCriteria=None, annotations=None, maxSamples=None, seed=1):

        # If filtering an unprocessed large dataset
        if startFromDF or (processed is None and (self.processed is None or self.processed.shape != self.df.shape)):
            processed, annotations = self.process(setProcessed=inplace, annotations=annotations, condition=includeCriteria, maxSamples=maxSamples, seed=seed)
        # If we already have processed data, filter as normal
        else:
            processed = self.processed if processed is None else processed
            processed, annotations = self.filter(df=processed, annotations=annotations, condition=includeCriteria, maxSamples=maxSamples, seed=seed, inplace=False)

        # Use ANOVA to get the genes that maximize reclassification
        genes = processed.index
        selector = SelectKBest(score_func=f_classif, k=int(proportion * len(genes)))
        trainSelected = selector.fit_transform(processed.T, annotations)
        selectedFeatures = list(genes[selector.get_support()])
        if not inplace:
            return processed.loc[selectedFeatures], selectedFeatures
        self.setAnndata(self.anndata[:, genes.isin(selectedFeatures)])

    # Get ortholog genes based on Ensembl mapping to another species
    def getOrthologs(self, mapping=None, inplace=True):

        mapping = pd.read_csv("/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Differentiation/objects/HumanMouseOrthologs.csv") if mapping is None else mapping

        # Filter out genes without orthologs and set names side by side
        print("Filtering AnnData object for orthologs...")
        validMap = mapping[mapping['test'].isin(self.df.index)]
        orthologAligned = self.anndata[:, validMap['test']].copy()
        oldNames = orthologAligned.var_names
        orthologAligned.var_names = validMap['basis']
        
        # Drop duplicates
        orthologAligned = orthologAligned[:, ~orthologAligned.var_names.duplicated()].copy()
        validMap = mapping[mapping['basis'].isin(orthologAligned.var_names)]
        validMap = mapping[mapping['test'].isin(oldNames)]
        orthologAligned = orthologAligned[:, validMap['basis']].copy()

        # Set aligned data and return
        if inplace:
            if self.processed is not None:
                self.processed = self.processed.loc[validMap['test'], :]
                self.processed.index = validMap['basis']
            self.setAnndata(orthologAligned)

        print("Done!")
        return orthologAligned

    # Combine df and annotations of two TopObjects   ### TODO: Check that optimization works, remove extraneous code
    def mergeWithOther(self, topObject, inplace=False, includeCriteriaSelf=None, includeCriteriaOther=None, indexName="gene"):
    
        # Filter and align TopObjects
        print("Aligning dfs...")
        if inplace:
            self.df = self.df if includeCriteriaSelf is None else self.df.loc[:, includeCriteriaSelf]
            topObject.df = topObject.df if includeCriteriaOther is None else topObject.df.loc[:, includeCriteriaOther]
            self.df.index.name = topObject.df.index.name = indexName
        if not inplace:
            dfSelf = self.df if includeCriteriaSelf is None else self.df.loc[:, includeCriteriaSelf]
            dfOther = topObject.df if includeCriteriaOther is None else topObject.df.loc[:, includeCriteriaOther]
            dfSelf.index.name = dfOther.index.name = indexName

        # Get metadata
        annotations = pd.concat([self.annotations if includeCriteriaSelf is None else self.annotations[includeCriteriaSelf], topObject.annotations if includeCriteriaOther is None else topObject.annotations[includeCriteriaOther]])

        # Replace dataset with new, combined data
        print("Merging TopObjects...")
        combinedObject = self if inplace else self.copy()
        combinedObject.timeColumn = None
        if inplace:
            self.df = pd.merge(self.df, topObject.df, on=indexName, how="inner")
            combinedObject.setAnndata(
                ad.AnnData(self.df.T, var=pd.DataFrame(self.df.index, index=self.df.index), 
                        obs=pd.DataFrame({self.cellTypeColumn: annotations[annotations.index.isin(self.df.columns)]}, index=self.df.columns)), 
                df=self.df
            )
        else:
            combinedDF = pd.merge(dfSelf, dfOther, on=indexName, how="inner")
            del dfSelf, dfOther
            combinedObject.setAnndata(
                ad.AnnData(combinedDF.T, var=pd.DataFrame(combinedDF.index, index=combinedDF.index), 
                        obs=pd.DataFrame({self.cellTypeColumn: annotations[annotations.index.isin(combinedDF.columns)]}, index=combinedDF.columns)), 
                df=combinedDF
            )
        combinedObject.processed = None
        print("Done merging!")
    
        # Return combined object if not merging inplace
        if not inplace:
            return combinedObject


# Gets a dict for passing in numbers of each cell type to downsampling function
def getLabelCountsMap(annotations, maxCount=None, proportion=None, keepFull=[]):
    labelCountsMap = {}
    valueCounts = annotations.value_counts()
    labelCountsMap = {label: int(valueCounts[label]) if maxCount is None or int(valueCounts[label]) < maxCount or label in keepFull else maxCount for label in set(annotations)}

    # If desired, set each label to a fraction of its actual count
    labelCountsMap = {label: labelCountsMap[label] if proportion is None else int(labelCountsMap[label] * proportion) for label in labelCountsMap.keys()}
    return labelCountsMap


# Undersample cell types based on a count maximum
def downsample(annotations, maxCount, df=None, proportion=None, keepFull=[], seed=1):
    labelCountsMap = getLabelCountsMap(annotations, maxCount=maxCount, keepFull=keepFull, proportion=proportion)
    samples = []
    for label in labelCountsMap.keys():
        rng = np.random.default_rng(seed) # Set here so rng is not order-dependent
        currentAnno = annotations[annotations == label]
        samples += list(currentAnno.index[rng.choice(len(currentAnno), size=labelCountsMap[label], replace=False)])
    return annotations[samples] if df is None else (df.loc[:, samples], annotations[samples])


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
    elif type(val) is bool:
        return val
    elif type(val) is str:
        val = val.upper()
        if val == "Y" or val == "YES" or val == "T" or val == "TRUE":
            return True
        if val == "N" or val == "NO" or val == "F" or val == "FALSE":
            return False
        return "Other"
    elif type(val) is list:
        if len(val) == 0 or (len(val) == 1 and val[0] == ""):
            return False
        # return True
    return "Other"


# Using any dataset with well-defined clusters, set it as a basis (duplicated to work outside TopObjects, and likely redundant with new scTOP code)
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


# Given the results of testing for best classifying genes, report best genes
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
    # server = Server('http://www.ensembl.org', use_cache = False)
    server = Server(host="http://may2025.archive.ensembl.org/") #https can break and maybe trailing slash
    ensemblMart = server.marts['ENSEMBL_MART_ENSEMBL']
    return ensemblMart, ensemblConfig, server


# Get ortholog gene mapping between any two species, with the reference first
def getOrthologMapping(basisSpecies, targetSpecies, ensemblMart, ensemblConfig):
    """Fetches 1:1 orthologs: Target IDs -> Basis IDs."""
    if basisSpecies == targetSpecies:
        return None

    if ensemblMart is None or ensemblConfig is None:
        #ensemblMart, ensemblConfig = 
        pass
    
    source_prefix = ensemblConfig[basisSpecies]['prefix']
    target_dataset_name = ensemblConfig[targetSpecies]['dataset']
    homolog_attr = f"{source_prefix}_homolog_ensembl_gene"
    
    try:
        print("Generating mapping...")
        dataset = ensemblMart.datasets[target_dataset_name]
        df = dataset.query(attributes=['ensembl_gene_id', homolog_attr], use_attr_names=True)
        df = df.dropna().drop_duplicates()
        df = df.drop_duplicates(subset=['ensembl_gene_id'], keep='first')
        df = df.drop_duplicates(subset=[homolog_attr], keep='first')
        mapping = df.set_index('ensembl_gene_id')[homolog_attr]
        
        # Get genes with orthologs to other species in mapping
        print("Finding orthologs...")
        mg = mygene.MyGeneInfo()
        testGenes = mg.querymany(mapping.index, field='symbol', size=1)
        basisGenes = mg.querymany(mapping.values, field='symbol', size=1)

        # Get symbols of genes
        symbolMapping = pd.DataFrame(data={
            "test": [val['symbol'] if 'symbol' in val.keys() else val['query'] for val in testGenes], 
            "basis": [val['symbol'] if 'symbol' in val.keys() else val['query'] for val in basisGenes]})

        return symbolMapping

    except Exception as e:
        print(f"  Warning: Could not fetch mapping for {targetSpecies}->{basisSpecies}: {e}")
        return pd.Series(dtype=str)


## ========================= ##
## Functions for loading and writing AnnData and basis objects ##
## ========================= ##

# Function to load a basis given a file location or basis name and summary csv
def loadBasis(file=None, basisCollection=None, basisName=None, geneIndex="gene", basisKeep=None):
    if file is None:
        if basisCollection is None or basisName is None:
            print("Must enter either a filename or a file containing multiple bases and the name of the basis you want")
            return None
        file = pd.read_csv(basisCollection, index_col="Name").loc[basisName, "File"]

    # Handle more complicated h5 case; bases are usually small enough that a csv is fine though
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
        idx = col.upper().find("CELL")
        if idx != -1:
            newCols[col] = col[:idx - 1]
    basis = basis.rename(columns=newCols)

    if basisKeep is not None:
        basis = basis[[colName for colName in basis.columns if colName in basisKeep]]
    print("Loaded " + basisName + " basis!")
    return basis


# Converts files in raw format (straight from GEO usually) to AnnData. Set geneHeader to None if no header
def rawToAnnData(countsPath, genesPath, metadataPath, barcodesPath=None,
                 matrix=False, transposeCounts=True, geneSeparator="\t", metadataSeparator="\t", barcodesSeparator="\t", metadataIndexColumn=None, geneHeader="infer", barcodesHeader="infer", geneColumnIdx=0, skipMetadataRow=None, skipCountsRow=None):
    # Set counts
    print("Setting counts...")
    counts = sc.read_mtx(countsPath) if matrix or countsPath.endswith("mtx") else ad.AnnData(pd.read_csv(countsPath))
    try:
        annObject = counts.T if transposeCounts else counts
    except:
        print("You may need to set transposeCounts to True")
        return None

    # Set metadata
    if metadataPath is not None:
        try:
            print("Setting metadata...")
            metadata = pd.read_csv(metadataPath, sep=metadataSeparator) if skipMetadataRow is None else pd.read_csv(metadataPath, sep=metadataSeparator, skiprows=[skipMetadataRow])
            metadata.replace(np.nan, '', inplace=True)
            if barcodesPath is not None:
                metadata.index = pd.read_csv(barcodesPath, sep=barcodesSeparator, header=barcodesHeader)
            else:
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
        genes.rename(columns={genes.columns[geneColumnIdx]: "name"}, inplace=True)
        genes.set_index("name", inplace=True)
        print("Adding genes to AnnData...")
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


# Use run-length encoding and sparse matrix to store processed data as small as possible. Note: Still very inefficient relative to raw data because processing removes 0s
def writeProcessed(processed, outFile):
    print("Converting to polars...")
    polarsProc = pl.from_pandas(processed)

    print("Applying RLE...")
    rleResults = {}
    for col in polarsProc.columns:
        unnested = polarsProc[col].rle().struct.unnest()
        rleResults[f"{col}_len"] = unnested["len"]
        rleResults[f"{col}_value"] = unnested["value"]

    print("Padding with 0s...")
    maxLen = max(s.len() for s in rleResults.values())
    rlePadded = {name: s.extend_constant(0, maxLen - s.len()) if s.len() < maxLen else s for name, s in rleResults.items()}
    rleStacked = np.column_stack([rlePadded[name].to_numpy() for name in rlePadded])

    print("Converting to sparse and writing...")
    mmwrite(outFile, csr_matrix(rleStacked))


# Decompress sparse processed data and reverse run-length encoding
def readProcessed(genes, colNames, fileName):
    print("Reading matrix and sending to array...")
    denseMatrix = sc.read_mtx(fileName).X.toarray()
    colPairs = [[col + "_len", col + "_value"] for col in colNames]
    colPairs = [key for sublist in colPairs for key in sublist]

    print("")
    reconstructed = {}
    for col in colNames:
        lenCol, valCol = (denseMatrix[:, colPairs.index(f"{col}_len")], denseMatrix[:, colPairs.index(f"{col}_value")])
        mask = lenCol != 0
        lens, vals = (lenCol[mask].astype(int), valCol[mask])
        reconstructed[col] = np.repeat(vals, lens)
    
    return pd.DataFrame(reconstructed, index=genes)