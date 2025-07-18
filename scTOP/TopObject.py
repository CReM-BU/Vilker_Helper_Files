# File containing TopObject class used for loading AnnData objects and performing core scTOP operations
# Author: Eitan Vilker (with some functions written by Maria Yampolskaya)

import numpy as np
import pandas as pd
import sctop as top
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import inspect
from copy import deepcopy
import os
import sys


class TopObject:
    def __init__(self, name, manualInit=False, useAverage=False, skipProcess=False, dataset="/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Vilker_Helper_Files/scTOP/DatasetInformation.csv"):
        self.name = name
        self.dataset = dataset
        if self.dataset is not None:
            datasetInfo = pd.read_csv(self.dataset, index_col="Name", keep_default_na=False).loc[self.name, :]
            self.cellTypeColumn, self.toKeep, self.toExclude, self.filePath, self.timeColumn, self.duplicates, self.raw, self.layer = datasetInfo
            self.raw = getTruthValue(self.raw)
            self.duplicates = getTruthValue(self.duplicates)
        else:
            # Prompt to manually add dataset info
            pass
        if type(self.toKeep) is str and len(self.toKeep) > 0:
            self.toKeep = self.toKeep[1:-1].replace("'", "").split(", ")
        if type(self.toExclude) is str and len(self.toExclude) > 0:
            self.toExclude = self.toExclude[1:-1].replace("'", "").split(", ")

        self.projections = {}
        self.basis = None
        if not manualInit:  # In case you want to adjust any of the parameters first
            self.setup(useAverage=useAverage, skipProcess=skipProcess)

    # Summary of parameters upon printing object
    def __str__(self):
        attributeMap = inspect.getmembers(self)[2][1]
        keys = attributeMap.keys()
        toReturn = "Attributes:"
        for key in keys:
            value = attributeMap[key]
            valueType = type(value)
            if valueType is str or valueType is bool or valueType is np.bool:
                toReturn += "\n" + key + ": " + str(value)
            elif valueType is list or valueType is np.ndarray:
                if len(value) > 0 and type(value[0]) is np.ndarray:
                    value = value[0]
                if len(value) > 5:
                    toReturn += "\n" + key + ": [" + ", ".join(list(value[:5])) + "]..."
                else:
                    toReturn += "\n" + key + ": " + str(value)
            elif valueType is pd.core.frame.DataFrame:
                toReturn += "\n" + key + ": " + str(list(value.columns)[:5]) + "..."
            elif valueType is pd.core.series.Series:
                toReturn += "\n" + key + ": " + str(list(value)[:5]) + "..."
            elif value is None:
                toReturn += "\n" + key + ": None"
            else:
                toReturn += "\n" + key + ": " + str(valueType)
        return toReturn

    # Copy the object into a new instance
    def copy(self):
        return deepcopy(self)

    # Initialize AnnData object, metadata, df, and process it
    def setup(self, useAverage=False, skipProcess=False):
        if not hasattr(self, "annObject"):
            print("Setting AnnData object...")
            self.annObject = sc.read_h5ad(self.filePath)
        if self.duplicates and self.duplicates != "":
            self.annObject.var_names_make_unique()
        self.setMetadata()
        self.setDF()
        if not skipProcess:
            self.processDataset(useAverage=useAverage)

    # Set AnnData object to include or exclude cells with certain labels
    def filterAnnObject(self, toKeep=None, toExclude=None, keep=False, exclude=False):
        print("Filtering AnnData...")
        if keep:
            if toKeep is None or toKeep == "":
                toKeep = self.toKeep
            self.annObject = self.annObject[self.annotations.isin(toKeep)]
        if exclude:
            if toExclude is None or toExclude == "":
                toExclude = self.toExclude
            self.annObject = self.annObject[~self.annotations.isin(toExclude)]
        self.setMetadata()

    # Set key features. May need to be called whenever object is edited
    def setMetadata(self):
        print("Setting metadata...")
        self.metadata = self.annObject.obs
        self.annotations = self.metadata[self.cellTypeColumn]
        self.filteredAnnotations = [label if (label in self.toKeep and label not in self.toExclude) else "Other" for label in self.annotations]
        self.timeSortFunction = None
        self.timesSorted = None
        if self.timeColumn is not None and self.timeColumn != "":
            self.timeSortFunction = lambda time: int("".join([char for char in time if char.isdigit()])) # if numbers in string unrelated to time this won't work
            self.timesSorted = sorted([str(time) for time in set(self.metadata[self.timeColumn])], key=self.timeSortFunction)

    # Set df, with a few extra options in case there are issues with the df
    def setDF(self, layer=None):
        print("Setting df...")
        if self.raw is not None and self.raw and self.raw != "":
            print(self.raw)
            self.df = pd.DataFrame(self.annObject.raw.X.toarray(), index = self.metadata.index, columns = self.annObject.raw.var_names).T
        else:
            if layer is None or layer == "":
                layer = self.layer
                if layer == "":
                    layer = None
            self.df = self.annObject.to_df(layer=layer).T
        if self.duplicates and self.duplicates != "":
            self.df = self.df.drop_duplicates().groupby(level=0).mean()

    # First scTOP function, ranks and normalizes source 
    def processDataset(self, useAverage=False):
        print("Processing scTOP data...")
        self.processedData = top.process(self.df, average=useAverage)
        print("Done!")

    # Main scTOP function, computing similarity between labels in sources and basis
    def projectOntoBasis(self, basis, projectionName):
        print("Projecting onto basis...")
        projection = top.score(basis, self.processedData)
        self.projections[projectionName] = projection
        print("Finished projecting!")
        return projection

    # Using any dataset with well-defined clusters, set it as a basis
    def setBasis(self, testing=False):
        print("Setting basis...")
        # Count the number of cells per type
        typeCounts = self.annotations.value_counts()

        # Using fewer than 150-200 cells leads to nonsensical results, due to noise. More cells -> less sampling error
        threshold = 200 # only use cell types with at least this many cells (but use all cells for training)
        types_above_threshold = typeCounts[typeCounts > threshold].index
        basisList = []
        trainingIDs = []
        rng = np.random.default_rng()
        for cell_type in tqdm(types_above_threshold):
            cell_IDs = self.metadata[self.annotations == cell_type].index
            if testing:
                current_IDs = rng.choice(cell_IDs, size=threshold, replace=False) # This line is for using only the threshold number of cells for the reference basis. This can be useful for testing the accuracy of the basis, but it performs notably worse in accuracy metrics compared to using all possible cells.
            else:
                current_IDs = cell_IDs
            cell_data = self.df[current_IDs]
            trainingIDs += [current_IDs] # Keep track of training_IDs so that you can exclude them if you want to test the accuracy

            # Average across the cells and process them using the scTOP processing method
            processed = top.process(cell_data, average=True)
            basisList += [processed]

        trainingIDs = np.concatenate(trainingIDs)
        basis = pd.concat(basisList, axis=1)
        basis.columns = types_above_threshold
        basis.index.name = "gene"
        print("Basis set!")
        if testing:
            return basis, trainingIDs
        self.basis = basis
        return basis

    # Add the desired columns of a smaller basis to a primary basis
    def combineBases(self, otherBasis, firstKeep=None, firstRemove=None, secondKeep=None, secondRemove=None, name="Combined"):
        print("Combining bases...")
        if self.basis is None:
            self.setBasis()
        if isinstance(otherBasis, TopObject):
            if otherBasis.basis is None:
                otherBasis.setBasis()
            basis2 = otherBasis.basis
        else:
            basis2 = otherBasis
        basis1 = self.basis
        if basis1.index.name is None:
            basis1.index.name = "gene"
        basis2.index.name = basis1.index.name
        if not hasattr(self, "combinedBases"):
            self.combinedBases = {}
        if firstKeep is not None:
            basis1 = basis1[firstKeep]
        if firstRemove is not None:
            basis1 = basis1[[col for col in basis1.columns if col not in firstRemove]]
        if secondKeep is not None:
            basis2 = basis2[secondKeep]
        if secondRemove is not None:
            basis2 = basis2[[col for col in basis1.columns if col not in secondRemove]]
        combinedBasis = pd.merge(basis1, basis2, on=basis1.index.name, how="inner")
        self.combinedBases[name] = combinedBasis
        return combinedBasis

    # Test an existing basis (not combined). Optionally adjust the minimum accuracy threshold
    def testBasis(self, specification_value=0.1):

        # Setting basis with holdouts for testing
        basis, trainingIDs = self.setBasis(testing=True)
        test_IDs = np.setdiff1d(self.df.columns, trainingIDs)
        testCount = len(test_IDs)
        splitIDs = np.array_split(test_IDs, 10) # From Maria- "I split this test dataset because it's very large and took up a lot of memory -- you don't need to do this if you have enough memory to test the entire dataset at once"

        print("Processing test data...")
        accuracies = {'top1': 0,
                      'top3': 0,
                      'unspecified': 0}
        matches = {}
        misses = {}
        for sample_IDs in tqdm(splitIDs):
            test_data = self.df[sample_IDs]
            test_processed = top.process(test_data)
            test_projections = top.score(basis, test_processed)
            accuracies, matches, misses = self.scoreProjections(accuracies, matches, misses, test_projections, specification_value=specification_value)
            del test_data
            del test_processed
            del test_projections
        for key, value in accuracies.items():
            print("{}: {}".format(key, value / testCount))

        accuracies["Total test count"] = testCount
        self.testResults = (accuracies, matches, misses)
        return self.testResults

    # Get the metrics for a given projection. Optionally adjust the minimum accuracy threshold
    def scoreProjections(self, accuracies, matches, misses, projections, specification_value=0.1): # cells with maximum projection under specification_value are considered "unspecified"

        predicted_labels = []
        predicted_labels_specified = []
        true_labels = []
        matches = {}
        misses = {}
        print("Scoring projection...")
        for sample_id, sample_projections in projections.items():
            types_sorted_by_projections = sample_projections.sort_values(ascending=False).index
            true_type = self.metadata.loc[sample_id, self.cellTypeColumn]
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

    # Create correlation matrix between cell types of basis, helpful to determine if any features are overlapping
    def getBasisCorrelations(self):
        print("Getting correlations...")
        if self.basis is None:
            print("Basis must be set first!")
            return None
        self.corr = self.basis.corr()
        print("Done!")
        return self.corr


def addDataset(summaryFile, name, filePath=None, cellTypeColumn=None, toKeep=None, toExclude=None, timeColumn=None, duplicates=None, raw=None, layer=None):
    summaryFileInfo = pd.read_csv(summaryFile, index_col="Name", keep_default_na=False)
    possibleEntries = [cellTypeColumn, toKeep, toExclude, filePath, timeColumn, duplicates, raw, layer]
    entryCount = len(possibleEntries)
    alreadyPresent = name in summaryFileInfo.index
    if alreadyPresent:
        newEntry = summaryFileInfo.loc[name, :].copy()
    else:
        newEntry = pd.Series([None] * entryCount)

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
    summaryFileInfo.to_csv(summaryFile, index_label="Name")
    return summaryFileInfo


# User-friendly way to update or add entry to the summary file
def dynamicAddDataset(summaryFile=None):
    if summaryFile is None:
        summaryFile = processInput("Enter the file path of a csv containing entries formatted for scTOP:", isFile=True, isRequired=True)
    name = processInput("Assign name. Enter a name to describe your dataset (if updating existing entry, choose the same name):", isRequired=True)
    filePath = processInput("Assign filePath. Enter the file path corresponding to the anndata object (.h5ad) for your dataset:", isFile=True)
    cellTypeColumn = processInput("Assign cellTypeColumn. Enter the title of the column containing cell type annotations:")
    toKeep = processInput("Assign toKeep. Press Y to enter a list of cell types that you may filter to later or press Enter to skip:", followUpMessage="Enter a cell type to include in filtering or press Enter to continue", isList=True)
    toExclude = processInput("Assign toExclude. Press Y to enter a list of cell types that you may filter out later or press Enter to skip:", followUpMessage="Enter a cell type to exclude in filtering or press Enter to continue", isList=True)
    timeColumn = processInput("Assign timeColumn, Enter the title of the column containing times samples were collected or press Enter to skip:")
    duplicates = processInput("Assign duplicates. Enter Y if the dataset has duplicate genes; otherwise enter N or press Enter to skip:", isBool=True)
    raw = processInput("Assign raw. Enter Y if using the raw values stored in the anndata object instead; otherwise enter N or press Enter to skip:", isBool=True)
    layer = processInput("Assign layer. Enter the name of a specific layer (typically counts, data, or scaled_data) to use or press Enter to skip:")
    addDataset(summaryFile, name, filePath=filePath, cellTypeColumn=cellTypeColumn, toKeep=toKeep, toExclude=toExclude, timeColumn=timeColumn, duplicates=duplicates, raw=raw, layer=layer)


# Checks user input and processes based on type
def processInput(message, followUpMessage=None, isFile=False, isList=False, isBool=False, isRequired=False):
    while True:
        userInput = input(message)
        if userInput == "Q":
            # return "Quit"
            sys.exit()
        elif userInput == "":
            if isRequired:
                print("A value must be entered here")
            else:
                return None
        elif isFile:
            if userInput == "Q":
                # return "Quit"
                sys.exit()
            elif os.path.exists(userInput):
                return userInput
            else:
                print("No file found at path: " + userInput)
        elif isBool:
            truthValue = getTruthValue(userInput)
            if truthValue is not None:
                userInput = truthValue
                break
        else:
            break

    if isList and userInput != "":
        entryList = []
        if userInput:
            while True:
                entry = input(followUpMessage)
                if entry == "Q":
                    # return "Quit"
                    sys.exit()
                elif entry == "":
                    break
                entryList.append(entry)
        userInput = entryList
    return userInput


# Allows user to escape if desired
def checkEscape(userInput):
    if userInput == "Q":
        pass


# Converts user input to Boolean True or False 
def getTruthValue(val):
    val = val.upper()
    if val == "Y" or val == "YES" or val == "T" or val == "TRUE":
        return True
    if val == "N" or val == "NO" or val == "F" or val == "FALSE":
        return False
    else:
        print("Enter a valid true/false value")
        return None


# Queries user for values to add to list if input is True
def getValueList(userInput, message):
    if getTruthValue(userInput):
        entryList = []
        while True:
            entry = input(message)
            if entry == "":
                break
            entryList.append(entry)
    else:
        entryList = []
    return entryList


# Asks for file until valid path provided 
def getValidatedFile(file, message):
    while True:
        # file = input("Enter the file path of a csv containing entries formatted for scTOP:")
        file = input(message)
        if os.path.exists(file):
            break
        else:
            print("No file found at path: " + file)