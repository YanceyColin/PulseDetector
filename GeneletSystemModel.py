#### Genelet System Model #####
#                             #
#      By: Colin Yancey       #
#      Created June 2022      #
#    Last Edited: 05/10/2023  #
#        Schulman Lab         #
#  Johns Hopkins University   #
#                             #
###############################

'''
Built using the General Genelet Model (GGM) from Schaffter et al. 2022 (https://www.nature.com/articles/s41557-022-01001-3)
This work intends to improve upon the existing model by rewriting the mass-action-kinetics ODE construction algorithms
and standardizing many useful methods for examining various circuit topologies and parameters. This code also integrates
aptlet (ARTIST) modeling (aptamers that work with genelets) and adds optional reversible reactions for future design space.
'''

import numpy as np
import scipy.integrate as spi
from sympy import symbols, Matrix, Transpose, matrix_multiply_elementwise as MME
import sys
import re
from datetime import date

import ArrayPlots as AP
import GeneralUse as GU

# Dictionary of rate constants that the model defaults to using if not overwritten
# (Units), function, (GGM rate constant name from Schaffter et al. 2022 Supplementary Information if applicable)
generalRateConstants = {
    "CoactivatorP": {
        "kVal": 0.0056, # (uL/U RNAP/s) Coactivator production (kpc)
        "List": "OCE"
    },
    "RepressorP": {
        "kVal": 0.0056, # (uL/U RNAP/s) Repressor production (kpr)
        "List": "OCE"
    },
    "CSilencerP": {
        "kVal": 0.0056, # (uL/U RNAP/s) Coactivator Silencer production (kpi)
        "List": "OCE"
    },
    "RSilencerP": {
        "kVal": 0.0056, # (uL/U RNAP/s) Repressor Silencer production (kpi)
        "List": "OCE"
    },

    "BlockerNode": {
        "kVal": 10000*1e-9, # (1/nM/s) Spontaneous binding of blocker to free node genelet (kgb)
        "List": "G"
    },
    "BlockerNodeReverse": {
        "kVal": 0, # (1/s) Spontaneous unbinding of blocker from node genelet
        "List": "G"
    },
    "BlkCoactDisplacement": {
        "kVal": 5000*1e-9, # (1/nM/s) Blocker sequestered via coactivator strand displacement (kgbc)
        "List": "G"
    },
    "BlkCoact": {
        "kVal": 10000*1e-9, # (1/nM/s) Free blocker sequestered via coactivator (kbc)
        "List": "OCE"
    },
    "BlkCoactReverse": {
        "kVal": 0, # (1/s) Spontaneous unbinding of blocker and coactivator
        "List": "OCE"
    },

    "ActivatorNode": {
        "kVal": 10000*1e-9, # (1/nM/s) Spontaneous binding of activator to free node genelet (kga)
        "List": "G"
    },
    "ActivatorNodeReverse": {
        "kVal": 0, # (1/s) Spontaneous unbinding of activator from node genelet
        "List": "G"
    },
    "ActRepressDisplacement": {
        "kVal": 5000*1e-9, # (1/nM/s) Activator sequestered via repressor strand displacement (kgar)
        "List": "G"
    },
    "ActRepress": {
        "kVal": 10000*1e-9, # (1/nM/s) Free activator sequestered via repressor (kar)
        "List": "OCE"
    },
    "ActRepressReverse": {
        "kVal": 0, # (1/s) Spontaneous unbinding of activator and repressor
        "List": "OCE"
    },

    "ActBlkDisplacement": {
        "kVal": 5000*1e-9, # (1/nM/s) Activator -> Blocker strand displacement on node genelet (kgab)
        "List": "G"
    },

    "RSilencing": {
        "kVal": 10000*1e-9, # (1/nM/s) Repressor sequestered via R-Silencer (kir)
        "List": "OCE"
    },
    "RSilencingReverse": {
        "kVal": 0, # (1/s) Spontaneous unbinding of repressor from R-Silencer
        "List": "OCE"
    },

    "CSilencing": {
        "kVal": 10000*1e-9, # (1/nM/s) Coactivator sequestered via C-Silencer (kir)
        "List": "OCE"
    },
    "CSilencingReverse": {
        "kVal": 0, # (1/s) Spontaneous unbinding of coactivator from C-Silencer
        "List": "OCE"
    },

    "RNaseHDegrade": {
        "kVal": 0.03363, # (uL/U Rnase H/s) Degradation rate of RNase H (kd_H)
        "List": "OCE"
    },
    "RNaseADegrade": {
        "kVal": 0.0003, # (1/s) Degradation rate of RNase A (kd_A)
        "List": "OCE"
    },


    # Aptlet kinetics
    "AptletProtein": {
        "kVal": 0.5, # (1/nM/s) Active aptlet silenced by target protein binding
        "List": "A"
    },
    "AptletProteinReverse": {
        "kVal": 0.32, # (1/s) Spontaneous unbinding of protein from aptlet
        "List": "A"
    }
}

# Dictionary of possible unit choices and their associated regularizing value
unitChoices = {
    "M": 1e9, # Molar
    "mM": 1e6, # Millimolar
    "uM": 1e3, # Micromolar
    "nM": 1, # Nanomolar (if concentrations aren't in a reasonable nM-magnitude regime, ODE solver may diverge)
    "sec": 1, # Seconds (time-regularizing math inversed from concentration-regularizing math)
    "min": 60, # Minutes
    "hr": 3600, # Hours,
    "U/uL": 1, # Units per microliter
    "U/mL": 1e-3 # Units per milliliter
}

# List of possible RNA products to be produced by a given genelet.
# "C" indicates a Coactivator.
# "R" indicates a Repressor.
# "CS" indicates a C-Silencer.
# "RS" indicates a R-Silencer.
geneletOutputs = ["C","R","CS","RS"]

# List of possible external systems interfacing with the genelet network
# "A" indicates an aptlet node
externalOpts = ["A"]

# Create a dictionary that maps each output to its respective index
geneletOutputIndex = {}
for i,x in enumerate(geneletOutputs):
    geneletOutputIndex[x] = i


####### ----- HELPER FUNCTIONS ----- #######

# Assembles a dictionary of class properties that allows for genelets and external options to all be recorded.
# actVec: The activator concentration List in order of the lowest to highest genelet index in a given network
# blkVec: The blocker concentration list "                                                                  "
# aptProteinVec: The protein concentration list for a network that contains aptlets
# unitChoice: The unit used for the concentrations of the given class properties (converts to nM)
def createGeneralClassProperties(actVec, blkVec, aptProteinVec=[], unitChoice="nM"):
    actLen = len(actVec)
    if actLen!=len(blkVec):
        sys.exit("Activator and Blocker lists must be of equal length.")

    return {
        "G":{ # Genelets
            "Activator": np.array(actVec, dtype=np.dtype(float))*unitChoices[unitChoice], # Concentration of activator in nM
            "Blocker": np.array(blkVec, dtype=np.dtype(float))*unitChoices[unitChoice] # Concentration of blocker in nM
        },
        "A":{ # Aptlets
            "Protein": np.array(aptProteinVec, dtype=np.dtype(float))*unitChoices[unitChoice] # Concentration of aptlet-specific protein in nM
        }
    }


# Creates an array of linearly interpolated time values to be used in a simulation run.
# maxTime: Maximum time point in the interpolation
# minTime: Minimum time point in the interpolation
# dataPointNum: The number of data points to be taken, including the minimum and maximum time data points
# unitChoice: The unit used for the max time value (converted to seconds for the simulation)
def getTimeValues(maxTime, minTime=0, dataPointNum=1001, unitChoice="hr"):
    return np.linspace(minTime, maxTime, dataPointNum)*unitChoices[unitChoice]


# Filters a set of domains by a class (node genelet "G", etc.) and/or a number (corresponding to an OCE).
# domainsList: A list of strings that have a set of characters followed by a set of numbers
# classStr: The class string to filter by
# numStr: The number string to filter by
def filterDomains(domainsList, classStr="\D+", numStr="\d+"):
    return [i for i,x in enumerate(domainsList) if re.search("^"+classStr+numStr+"$", x)]


# Returns a given domain address's corresponding class and number that is original compressed as a single string.
# domainAddress: A set of characters followed by a number, indicating the class and index, respectively, of a given domain
def getClassAndNumber(domainAddress):
    if domainAddress=="":
        return ["", None]
    else:
        return [re.search("\D+", domainAddress).group(), int(re.search("\d+", domainAddress).group())]


# Returns a list of the grouped input and output domains corresponding to a given class.
# inputDomains: A domains list (see filter domains) for the inputs of a genelet network
# outputDomains: A domains list (see filter domains) for the outputs of a genelet network
# classStr: The class whose pairs are being extracted from the domains lists
def getClassPairs(inputDomains, outputDomains, classStr):
    return [
        [
            getClassAndNumber(inputDomains[i]),
            getClassAndNumber(outputDomains[i])
        ] for i in filterDomains(inputDomains, classStr=classStr)
    ]


# Create a set of differential equations corresponding to a genelet network's current state that explains how it will evolve in time.
# tVal: The time at which the instanteous slopes of each variable are being extracted
# dVals: The state of the genelet network prior to the timestep
# inputMat: Input matrix detailing which inputs match with which orthogonal circuit elements (OCEs)
# conMats: Connectivity matricies for the genelet network as well as each additional external component interfacing with the network
# nCParsed: The node concentrations parsed into individual concentrations for each class of component to help resolve the vector math
# classProperties: Class properties of the genelet network
# rxnRates: The dictionary of reaction rate vectors in accordance with the dimensions of the genelet network
# RNaseH and RNaseA: Concentrations of RNase H and A in solution respectively. They are in units of U/uL.
def GenerateGeneletEquations(tVal, dVals,
    inputMat, conMats,
    nCParsed, classProperties,
    rxnRates,
    RNAP, RNaseH, RNaseA
):
    ngCount, oceCount = inputMat.shape # Node genelet count, orthogonal circuit element count

    # Sort the input domain which is constrained to the dVals list (constraint of solve_ivp) into a set of vectors
    freeCoact      = dVals[0          :   oceCount] # Free coactivator
    freeRepress    = dVals[oceCount   : 2*oceCount] # Free repressor
    freeCSilencer  = dVals[2*oceCount : 3*oceCount] # Free C-Silencer
    freeRSilencer  = dVals[3*oceCount : 4*oceCount] # Free R-Silencer

    BlkCoactComp   = dVals[4*oceCount : 5*oceCount] # Blocker-coactivator complex
    ActRepressComp = dVals[5*oceCount : 6*oceCount] # Activator-repressor complex
    CSilenceComp    = dVals[6*oceCount : 7*oceCount] # Coactivator-C-Silencer complex
    RSilenceComp    = dVals[7*oceCount : 8*oceCount] # Repressor-R-Silencer complex

    btmIndex = 8*oceCount
    BlkNGComp = dVals[btmIndex            :   btmIndex+ngCount] # Activator-node genelet complex
    ActNGComp = dVals[btmIndex+ngCount    : btmIndex+2*ngCount] # Blocker-node genelet complex

    freeAptlet = dVals[btmIndex+2*ngCount : btmIndex+2*ngCount+len(nCParsed["A"])] # Aptlet unbound from corresponding protein

    # Some helpful components to compress the math (not part of the list of diff eqs)
    freeBlocker = classProperties["G"]["Blocker"] - BlkCoactComp - inputMat.T@BlkNGComp
    freeActivator = classProperties["G"]["Activator"] - ActRepressComp - inputMat.T@ActNGComp

    freeNG = nCParsed["G"] - BlkNGComp - ActNGComp
    ActNGToBlkNG = rxnRates["ActBlkDisplacement"]*ActNGComp*(inputMat@freeBlocker)

    netBlkCoact = rxnRates["BlkCoact"]*freeBlocker*freeCoact - rxnRates["BlkCoactReverse"]*(inputMat.T@BlkNGComp)
    netActRepress = rxnRates["ActRepress"]*freeActivator*freeRepress - rxnRates["ActRepressReverse"]*(inputMat.T@ActNGComp)

    blkCoactDisplace = freeCoact*(inputMat.T@(rxnRates["BlkCoactDisplacement"]*BlkNGComp))
    actRepressDisplace = freeRepress*(inputMat.T@(rxnRates["ActRepressDisplacement"]*ActNGComp))

    netCSilencing = rxnRates["CSilencing"]*freeCoact*freeCSilencer - rxnRates["CSilencingReverse"]*CSilenceComp
    netRSilencing = rxnRates["RSilencing"]*freeRepress*freeRSilencer - rxnRates["RSilencingReverse"]*RSilenceComp

    rnaseACoef = rxnRates["RNaseADegrade"]*RNaseA # Degrades RNA in solution
    rnaseAandHCoef = rnaseACoef + rxnRates["RNaseHDegrade"]*RNaseH # RNase H degrades RNA in RNA-DNA duplexes. RNA in these duplexes will also be degraded by RNase A
    outputProduction = {} # 
    for outputType in geneletOutputs:
        outputProduction[outputType] = conMats["G"][:, geneletOutputIndex[outputType], :].T@ActNGComp
        for objClass in externalOpts:
            if conMats[objClass].ndim==3:
                if objClass=="A": # Iterate through each different class to assign their unique production method
                    outputProduction[outputType] += conMats[objClass][:, geneletOutputIndex[outputType], :].T@freeAptlet

    aptletProteinComp = nCParsed["A"] - freeAptlet

    # Assemble the next iteration of diff eq's
    dFreeCoact = -blkCoactDisplace - netBlkCoact - netCSilencing - rnaseACoef*freeCoact + RNAP*rxnRates["CoactivatorP"]*outputProduction["C"]
    dFreeRepress = -actRepressDisplace - netActRepress - netRSilencing - rnaseACoef*freeRepress + RNAP*rxnRates["RepressorP"]*outputProduction["R"]
    dFreeCSilencer = -netCSilencing - rnaseACoef*freeCSilencer + RNAP*rxnRates["CSilencerP"]*outputProduction["CS"]
    dFreeRSilencer = -netRSilencing - rnaseACoef*freeRSilencer + RNAP*rxnRates["RSilencerP"]*outputProduction["RS"]

    dBlkCoactComp = netBlkCoact + blkCoactDisplace - rnaseAandHCoef*BlkCoactComp
    dActRepressComp = netActRepress + actRepressDisplace - rnaseAandHCoef*ActRepressComp
    dCSilenceComp = netCSilencing - rnaseACoef*CSilenceComp
    dRSilenceComp = netRSilencing - rnaseACoef*RSilenceComp

    dBlkNGComp = -rxnRates["BlkCoactDisplacement"]*BlkNGComp*(inputMat@freeCoact) + ActNGToBlkNG + rxnRates["BlockerNode"]*freeNG*(inputMat@freeBlocker) - rxnRates["BlockerNodeReverse"]*BlkNGComp
    dActNGComp = -rxnRates["ActRepressDisplacement"]*ActNGComp*(inputMat@freeRepress) - ActNGToBlkNG + rxnRates["ActivatorNode"]*freeNG*(inputMat@freeActivator) - rxnRates["ActivatorNodeReverse"]*ActNGComp
    dFreeAptlet = -rxnRates["AptletProtein"]*freeAptlet*(classProperties["A"]["Protein"] - aptletProteinComp) + rxnRates["AptletProteinReverse"]*aptletProteinComp

    return np.concatenate([dFreeCoact, dFreeRepress, dFreeCSilencer, dFreeRSilencer, dBlkCoactComp, dActRepressComp, dCSilenceComp, dRSilenceComp, dBlkNGComp, dActNGComp, dFreeAptlet])


####### ----- GENELET NETWORK CLASS ----- #######

class GeneletNetwork:
    # Initializes the genelet network class and determines the network topology.
    # inputDomains: List of strings that have letter and number identifiers for each input in the network
        # Each input domain has an integer. This is its associated input domain and all elements that interact directly will share this integer.
            # If prefixed with a G, this refers to a genelet input domain.
            # If prefixed with an A, this refers to an aptlet input domain.
    # outputDomains: Corresponding outputs that are produced upon activation of the corresponding inputDomain
        # Each output domain has a prefix, which must be a string in geneletOutputs.
        # Each output domain has an integer. This is its associated output domain and all elements that interact directly will share this integer.
    # areSpecificDomains: Details whether the indexes used correspond with specifically characterized input or output domains
    def __init__(self, inputDomains, outputDomains, areSpecificDomains=False):
        if len(inputDomains)!=len(outputDomains):
            sys.exit("Genelet network must have input and output domains of equal length.")

        self.InputDomains, self.OutputDomains = inputDomains, outputDomains

        self.DomainList = {"G": getClassPairs(inputDomains, outputDomains, "G")}
        self.OCEs = [getClassAndNumber(y) for y in set(["".join([x[0][0],str(x[0][1])]) for x in self.DomainList["G"]])] # Orthogonal circuit elements
        self.OCEs.sort(key = lambda x: x[1]) # Order them in ascending order to match the anticipated order of classProperties variables.

        # Generates a 2D matrix that one-hot-encodes each the OCEs and assigns each node genelet to one by their respective input domain.
        self.InputMatrix = np.array([[(z[1]==x[0][1] and 1) or 0 for z in self.OCEs] for x in self.DomainList["G"]])

        # Generates a 3D matrix which associates an upstream node's activity with the production of a downstream genelet that affects an entire domain.
        # The first dimension goes through each node to assign the node responsible for the output.
        # The second dimension designates separate channels for each possible node output.
        # The third dimension indicates the particular OCE the output is directed at.
        def createConnectivityMatrix(classPairs):
            tempVar = np.array([[(y!=x[1][0] and [0]*len(self.OCEs)) or [(z[1]==x[1][1] and 1) or 0 for z in self.OCEs] for y in geneletOutputs] for x in classPairs])
            return tempVar

        self.ConnectivityMatricies = {"G": createConnectivityMatrix(self.DomainList["G"])}
        self.NodeConcentrationsEncoder = {"G": filterDomains(inputDomains, "G")}

        for externalClass in externalOpts:
            self.DomainList[externalClass] = getClassPairs(inputDomains, outputDomains, externalClass)
            self.ConnectivityMatricies[externalClass] = createConnectivityMatrix(getClassPairs(inputDomains, outputDomains, externalClass))
            self.NodeConcentrationsEncoder[externalClass] = filterDomains(inputDomains, classStr=externalClass)

        self.RxnRates = {} # Set up reaction rates as vectors
        for x in generalRateConstants.keys():
            vecType = generalRateConstants[x]["List"]
            if vecType == "OCE":
                self.RxnRates[x] = np.full(len(self.OCEs), generalRateConstants[x]["kVal"])
            else:
                self.RxnRates[x] = np.full(len(self.DomainList[vecType]), generalRateConstants[x]["kVal"])

        if areSpecificDomains:
            pass # Change chosen reaction rates to genelet-specific ones

    # Sets the initial conditions of the given genelet network.
    # initialState: List of strings that records the initial condition of each node in order of the input and output domain lists.
        # For genelets, these can be as follows:
            # "ON" (node genelet is preannealed to activator)
            # "OFF" (node genelet starts off free in solution) **DEFAULT**
            # "BLK" (node genelet is preannealed to blocker)
        # For aptlets, N/A
    # nodeConcentrations: List of values that are the concentrations of each node in order of the input and output domain lists.
    # classProperties: Dictionary of the relevant class properties for the given network (Key: Name for each input domain's class, such as "G" or "A").
    # ...Property lists are ordered by ascending number following the domain's class identifier (i.e. G1's Activator, then G2's, then G3's, etc.)
        # For genelets, this will include the following lists:
            # Activator concentration (Key: Activator)
            # Blocker concentration (Key: Blocker)
        # For aptlets, this will include the following lists:
            # Protein concentration (Key: Protein)
    # standardBLK: Additional blocker is added to the blocker initialized in classProperties equivalent to 1.5x the total concentration of all initially blocked node genelets in the OCE family. Standard protocol from Shaffter et. al
    # unitChoice: Units of the given initial condition values. All units are converted to nM for the simulation.
    # RNAP, RNaseH, and RNaseA: Concentrations of RNA polymerase, RNase H, and RNase A in solution respectively. They are in units of U/uL.
    def setInitialConditions(self, initialState=[], nodeConcentrations=[], classProperties=[], standardBLK=True, unitChoice="nM", RNAP=3.57, RNaseH=8.92e-3, RNaseA=0):
        if initialState!=[]:
            self.InitialState = initialState
        elif not hasattr(self, "InitialState"):
            print("The initial state must be defined in the first set of initial conditions.")
            exit()

        if nodeConcentrations!=[]:
            self.NodeConcentrations = np.array(nodeConcentrations, dtype=np.dtype(float))*unitChoices[unitChoice]
        elif not hasattr(self, "NodeConcentrations"):
            print("The node concentrations must be defined in the first set of initial conditions.")
            exit()

        if classProperties!=[]:
            self.ClassProperties = classProperties
            for i, x in enumerate(self.OCEs):
                oceConcentration = sum(self.NodeConcentrations[[y for y in filterDomains(self.InputDomains, classStr="G", numStr=str(x[1])) if self.InitialState[y]=="BLK"]])
                if standardBLK:
                    self.ClassProperties["G"]["Blocker"][i] += 1.5*oceConcentration
                elif self.ClassProperties["G"]["Blocker"][i] < oceConcentration:
                    print("Not enough blocker provided in solution to allow for all G"+str(x[1])+" domains to be blocked")
                    exit()
        elif not hasattr(self, "ClassProperties"):
            print("The class properties must be defined in the first set of initial conditions.")
            exit()

        self.RNAP   = RNAP
        self.RNaseH = RNaseH
        self.RNaseA = RNaseA

        ngCount, oceCount = self.InputMatrix.shape # Orthogonal circuit element count
        ngRange = range(ngCount)

        self.InitialConditions = np.concatenate([
            [0]*oceCount, # freeCoact
            [0]*oceCount, # freeRepress
            [0]*oceCount, # freeCSilencer
            [0]*oceCount, # freeRSilencer
            [0]*oceCount, # BlkCoactComp
            [0]*oceCount, # ActRepressComp
            [0]*oceCount, # CSilenceComp
            [0]*oceCount, # RSilenceComp
            [self.NodeConcentrations[i] if self.InitialState[i]=="BLK" else 0 for i in self.NodeConcentrationsEncoder["G"]], # BlkNGComp (can be blocked at start)
            [self.NodeConcentrations[i] if self.InitialState[i]=="ON" else 0 for i in self.NodeConcentrationsEncoder["G"]], # ActNGComp (can be activated at start)
            self.NodeConcentrations[self.NodeConcentrationsEncoder["A"]] # freeAptlet (all free at start)
        ], dtype=np.dtype(float))

    # Simulates the expected kinetics of the given genelet network.
    # timeList: A list containing all of the time stamps to be evaluated and recorded
        # The first and last time stamps are the initial and final times that the solution will integrate over, using the established initial conditions.
    # resetRun: If set to True, the initial conditions will be reset to their default values.
        # If set to False, the initial conditions will be the endpoint of the prior simulation (this endpoint can be edited before execution of another simulation).
    # unitChoice: Sets the solution's units used in creating OutputConcentrations. Defaults to nM.
    def simulate(self, timeList, resetRun=True, unitChoice="nM"):
        if not hasattr(self, "NodeConcentrations"):
            sys.exit("Initial conditions must be set before a simulation can start.")

        nCParsed = {}
        for key, val in self.NodeConcentrationsEncoder.items():
            nCParsed[key] = self.NodeConcentrations[val]

        rates = lambda tVal, dVals: GenerateGeneletEquations(tVal, dVals,
            self.InputMatrix, self.ConnectivityMatricies,
            nCParsed, self.ClassProperties,
            self.RxnRates,
            self.RNAP, self.RNaseH, self.RNaseA
        )

        rawSolution = spi.solve_ivp(rates, [timeList[0], timeList[-1]], self.InitialConditions if resetRun else self.CurrentConditions, method="LSODA", t_eval=timeList)
        if resetRun or not hasattr(self, "TimeList"):
            self.TimeList = rawSolution.t/3600 # Timestamps of saved data
        else:
            self.TimeList = np.append(self.TimeList[:-1], rawSolution.t/3600)
        rawSolution.y /= unitChoices[unitChoice] # Normalize the solution concentrations to the designated units

        ngCount, oceCount = self.InputMatrix.shape # Node genelet count, orthogonal circuit element count
        if resetRun or not hasattr(self, "OutputConcentrations"):
            self.OutputConcentrations = {} # Create a dictionary of relevant concentrations for easy look-ups

        for i in range(oceCount):
            oceNum = str(self.OCEs[i][1])
            GU.safeAppend(self.OutputConcentrations, "C"+oceNum,     rawSolution.y[             i], 1)
            GU.safeAppend(self.OutputConcentrations, "R"+oceNum,     rawSolution.y[oceCount   + i], 1)
            GU.safeAppend(self.OutputConcentrations, "CS"+oceNum,    rawSolution.y[2*oceCount + i], 1)
            GU.safeAppend(self.OutputConcentrations, "RS"+oceNum,    rawSolution.y[3*oceCount + i], 1)
            GU.safeAppend(self.OutputConcentrations, "Blk-C"+oceNum, rawSolution.y[4*oceCount + i], 1)
            GU.safeAppend(self.OutputConcentrations, "Act-R"+oceNum, rawSolution.y[5*oceCount + i], 1)
            GU.safeAppend(self.OutputConcentrations, "CS-C"+oceNum,  rawSolution.y[6*oceCount + i], 1)
            GU.safeAppend(self.OutputConcentrations, "RS-R"+oceNum,  rawSolution.y[7*oceCount + i], 1)

        for i in range(ngCount):
            cP = self.DomainList["G"][i] # Class pair list
            ngName = cP[0][0]+str(cP[0][1])+"->"+cP[1][0]+ ("" if cP[1][0]=="" else str(cP[1][1]))
            GU.safeAppend(self.OutputConcentrations, "Blk: "+ngName, rawSolution.y[8*oceCount +           i], 1)
            GU.safeAppend(self.OutputConcentrations, "Act: "+ngName, rawSolution.y[8*oceCount + ngCount + i], 1)

        aptletCount = len(self.DomainList["A"])
        for i in range(aptletCount):
            aptNum = str(self.DomainList["A"][0][i][1])
            GU.safeAppend(self.OutputConcentrations, "Apt"+aptNum, rawSolution.y[8*oceCount+2*ngCount + i], 1)

        self.CurrentConditions = rawSolution.y[:, -1] # Take the last element of every solution element

    # Changes the indicated value if a new value is provided. See plot1Variable for additional info on parameters.
    def modifyValue(self, valueType, valueInd=None, newValue=None, unitChoice="nM"):
        targName = ""
        if valueType=="RNAP" or valueType=="RNaseH" or valueType=="RNaseA":
            curVal = self.__dict__[valueType]
            if newValue!=None:
                self.__dict__[valueType] = newValue*unitChoices[unitChoice]
            return curVal, ""
        elif valueInd==None:
            sys.exit("ERROR: valueType determined to be a list, therefore valueInd must be provided.")

        if valueType=="Node":
            targ = self.NodeConcentrations
            targName = self.InputDomains[valueInd]+("" if self.OutputDomains[valueInd]=="" else " -> "+self.OutputDomains[valueInd])
        elif valueType=="Activator":
            targ = self.ClassProperties["G"]["Activator"]
            targName = "G"+str(self.OCEs[valueInd][1])
        elif valueType=="Blocker":
            targ = self.ClassProperties["G"]["Blocker"]
            targName = "G"+str(self.OCEs[valueInd][1])
        elif valueType=="Protein":
            targ = self.ClassProperties["A"]["Protein"]
            targName = "A"+str(self.DomainList["A"][0][valueInd][1])
        else:
            sys.exit("ERROR: Indicated value type is unrecognizable.")

        curVal = targ[valueInd]
        if newValue!=None:
            targ[valueInd] = newValue*unitChoices[unitChoice]

        self.setInitialConditions(RNAP=self.RNAP, RNaseH=self.RNaseH)
        return curVal, targName

    # Generates a set of plots along a change of one variable.
    # valueType: Name of list (or variable) where the value is stored
    # valueInd: Index of the value if the value is in a list, otherwise None
    # newValues: The values to be substituted sequentially for each simulation
    # timeList: The chosen time values that will be plotted (list of numbers from lowest to highest time value)
    # plotOutputs: The outputs of interest to be plotted from each simulation's solution list. The main variable (a list) contains lists of strings, each of which detail the outputs on the particular plot (grouped by column).
    # plotFileName: The designated path + file name of the saved PNG from the simulations
    # specialOpts: Optional array of modifications to be made in the final plotting. See ArrayPlots.py for additional information.
    # spikes: Optional parameter indicating abrupt modifications induced in the target simulations. See "createSpikeArray()" for more details.
    # timeUnits: Units used for timeList and the plots
    # concUnits: Units used for newValues and the plots
    def plot1Variable(self, valueType, valueInd, newValues, timeList, plotOutputs, plotFileName="plotFile", specialOpts=[], spikes=None, timeUnits="hr", concUnits="nM"):
        defaultVal, targName = self.modifyValue(valueType, valueInd)
        curPlotData = [[[] for _ in range(len(newValues))] for _ in range(len(plotOutputs))]

        newTab = []

        for i, x in enumerate(newValues):
            self.modifyValue(valueType, valueInd, x, concUnits)
            self.simulate(timeList, unitChoice=concUnits)
            for j, plotOutput in enumerate(plotOutputs):
                for indX in plotOutput:
                    curPlotData[j][i].append([self.OutputConcentrations[indX]]) # It is in its own table because it is isolated from other data for purpose of AP coloring

        self.modifyValue(valueType, valueInd, defaultVal, concUnits)
        specialOpts.append(["Legend", AP.offsetLegends(len(newValues), plotOutputs)])
        AP.generatePlots(timeList/unitChoices[timeUnits], curPlotData,
            plotXTitles=[str(GU.roundToSigFig(x,5))+" "+concUnits for x in newValues], globalXTitle=valueType+("" if valueInd==None else " ("+targName+")"), localXTitle="Time ("+timeUnits+")",
            localYTitle="Concentration ("+concUnits+")",
            plotFileName=plotFileName,
            colorHues = AP.makeHueArray(
                {
                    "Type" : "RowBased",
                    "GroupData" : [len(x) for x in plotOutputs]
                },
                len(plotOutputs),
                len(newValues),
                unique=True
            ),
            specialOpts=specialOpts
        )
        self.saveGeneletParams(plotFileName)

    # Generates a set of plots while changing two variables with each permutation of changes displayed in the grid.
    # See plot1Variable counterparts for info on each variable; plotOutputs is no longer a list of lists as it is just the list of outputs displayed in each subplot of the grid (each simulation is only allocated one subplot)
    def plot2Variables(
        self,
        valueType1, valueInd1, newValues1,
        valueType2, valueInd2, newValues2,
        timeList, plotOutput, plotFileName="plotFile", specialOpts=[],
        spikes=None, timeUnits="hr", concUnits1="nM", concUnits2="nM", outputUnits="nM",
        fixedY=None
    ):
        defaultVal1, targName1 = self.modifyValue(valueType1, valueInd1)
        defaultVal2, targName2 = self.modifyValue(valueType2, valueInd2)
        curPlotData = [[[] for _ in range(len(newValues1))] for _ in range(len(newValues2))]

        for i, x1 in enumerate(newValues1):
            self.modifyValue(valueType1, valueInd1, x1, concUnits1)
            for j, x2 in enumerate(newValues2):
                self.modifyValue(valueType2, valueInd2, x2, concUnits2)
                self.simulate(timeList, unitChoice=outputUnits)
                for indX in plotOutput:
                    curPlotData[j][i].append([self.OutputConcentrations[indX]]) # It is in its own table because it is isolated from other data for purpose of AP coloring

        self.modifyValue(valueType1, valueInd1, defaultVal1)
        self.modifyValue(valueType2, valueInd2, defaultVal2)
        specialOpts.append(["Legend", AP.offsetLegends(len(newValues1), [plotOutput])])
        AP.generatePlots(timeList/unitChoices[timeUnits], curPlotData,
            plotXTitles=[str(GU.roundToSigFig(x, 5))+" "+concUnits1 for x in newValues1], globalXTitle=valueType1+("" if targName1=="" else " ("+targName1+")"), localXTitle="Time ("+timeUnits+")",
            plotYTitles=[str(GU.roundToSigFig(x, 5))+" "+concUnits2 for x in newValues2], globalYTitle=valueType2+("" if targName2=="" else " ("+targName2+")"), localYTitle="Concentration ("+outputUnits+")",
            plotFileName=plotFileName,
            specialOpts=specialOpts,
            fixedY=fixedY
        )
        self.saveGeneletParams(plotFileName)


    # Generates a TXT file that stores all of the necessary information needed to recreate the genelet network.
    # fileName: The designated path + file name of the saved TXT
    # timeStamp: Records the date the file was created at the top of the txt; set to False to omit this information
    def saveGeneletParams(self, fileName="plotFile", timeStamp=True):
        paramsTxt = ''.join([
            "" if not timeStamp else str(date.today())+"\n\n"
            "RNAP (U/uL):     "+str(GU.roundToSigFig(self.RNAP, 5))+"\n",
            "RNaseH (U/uL):   "+str(GU.roundToSigFig(self.RNaseH, 5))+"\n\n"
            "Input Domains:   "+str(self.InputDomains)+"\n",
            "Output Domains:  "+str(self.OutputDomains)+"\n",
            "Initial State:   "+str(self.InitialState)+"\n",
            "Node Conc. (nM): "+str(self.NodeConcentrations)+"\n\n",
            "Activator Conc. (nM): "+str(self.ClassProperties["G"]["Activator"])+"\n",
            "Blocker Conc. (nM):   "+str(self.ClassProperties["G"]["Blocker"])+"\n\n",
            "" if len(self.ClassProperties["A"]["Protein"])==0 else "Protein Conc. (nM): "+str(self.ClassProperties["A"]["Protein"])
        ])

        with open(fileName+".txt", 'w') as f:
            f.write(paramsTxt)


# Demonstrate a simple IFFL and its activity controlled by the initial concentration of a protein
if __name__=="__main__":
    #Initialize a genelet network
    newGN = GeneletNetwork(
        ["G2", "G2", "G3", "G1"],
        ["C1", "C3", "R1", ""]
    )

    # Set the genelet network's initial conditions
    newGN.setInitialConditions(
        ["OFF","OFF","BLK","BLK"],
        [25, 5, 25, 25],
        createGeneralClassProperties([250, 250, 250], [0, 0, 0])
    )

    # Plot the activity of three genelets vs. time in separate graphs, and do this for three variations of A1 concentration.
    newGN.plot1Variable(
        "Activator", 0, [125, 250, 500],
        getTimeValues(4),
        [["Act: G2->C1"], ["Act: G3->R1"], ["Act: G1->"]],
        plotFileName="StandardIFFL"
    )
