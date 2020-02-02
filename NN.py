import random
import math
import numpy
import copy

MAXSIGMOID = 40

def sigmoid(x):
    if x >MAXSIGMOID:
        return 1
    elif x < -MAXSIGMOID:
        return 0
    else:
        return 1/(1 + numpy.exp(-x))

class NeuralNetwork:
    def __init__(self, nbInput, nbHiddenLayer, nbNeuronPerHL, nbOutput):
        """Cette fonction doit intialiser un réseau avec des poids nul (ou laissés non définis)"""
        self.poidsInitialise = False
        self.inputsFed = False
        self.computed = False
        
        self.nbInput = nbInput
        self.nbHiddenLayer = nbHiddenLayer
        self.nbNeuronPerHL = nbNeuronPerHL
        self.nbOutput = nbOutput

        self.sigmoidVect = numpy.vectorize(sigmoid)

        self.weights=[]
        self.shapes=[]
        
        #input
        if (nbHiddenLayer>0):
            inputWeights = numpy.matrix([[0]*nbNeuronPerHL for i in range(nbInput+1)]) #+1 pour le biais
        else:
            inputWeights = numpy.matrix([[0]*nbOutput for i in range(nbInput+1)]) #+1 pour le biais
        self.weights.append(inputWeights)
        self.shapes.append(inputWeights.shape)

        #hidden
        for h in range(self.nbHiddenLayer-1):
            hiddenWeights = numpy.matrix([[0]*nbNeuronPerHL for i in range(nbNeuronPerHL+1)]) #+1 pour le biais
            self.weights.append(hiddenWeights)
            self.shapes.append(hiddenWeights.shape)

        #output
        outputWeights = numpy.matrix([[0]*nbOutput for i in range(nbNeuronPerHL+1)]) #+1 pour le biais
        self.weights.append(outputWeights)
        self.shapes.append(outputWeights.shape )

        self.inputArray = numpy.array([0]*(nbInput+1))
        self.ouputArray = numpy.array([0]*nbOutput)

    def createRandomWeight(self, moy, cible):
        """En moyenne, le poids tiré sera: moy, et en moyenne, un neuron de la couche suivante aura en entree la valeur +/- cible. Une cible plus grande que MAXSIGMOID est inutile (cf Sigmoid)"""
        for i in range(len(self.weights)):
            nbDeSommation = self.weights[i].shape[0]
            #En moyenne, l'input sera de 0.5. On veut nbSommation*(0.5*weight) = cible 
            ecartType = cible/nbDeSommation/0.5 /3 #On divise par 3 pour avoir 95% de chance que la valeur tiree respecte notre equation
            self.weights[i] = numpy.random.normal(moy,ecartType,self.shapes[i])

        self.poidsInitialise = True
        self.computed = False #Il faut recompute

    def setWeight(self, weights):
        self.weights  = copy.deepcopy(weights)
        self.poidsInitialise = True
        self.computed = False #Il faut recompute

    def setInputs(self,myInputArray):
        """Cette fonction donne l'array d'entree"""
        self.inputArray = numpy.append(myInputArray,1) #pour le biais. Numpy genere une copy de l'array
        self.inputsFed = True
        self.computed = False #Il faut recompute
        

    def compute(self):
        """Cette fonction propage les input jusqu'a la sortie"""
        if (self.poidsInitialise and self.inputsFed):
            progressArray = copy.deepcopy(self.inputArray)
            progressArray = self.sigmoidVect(progressArray)
            for w in self.weights:
                progressArray = numpy.dot(progressArray,w)
                progressArray = self.sigmoidVect(progressArray)
                progressArray = numpy.append(progressArray,1) #pour le biais. Numpy genere une copy de l'array
            self.ouputArray = copy.deepcopy(progressArray)[:-1]
            self.computed = True
        else:
            print("Error can't compute: The NN has weights (" + str(self.poidsInitialise) + ") and inputs (" + str(self.inputsFed) +")" )

    def getOutputs(self):
        """Cette fonction recupere la sortie"""
        if (self.computed):
            return self.ouputArray
        else:
            print("Error can't get ouptut: The NN has not computed before")

    def reset(self):
        self.inputsFed = False
        self.computed = False


def crossbreedNN(parent1,parent2,taux_mutation):
    """Fonction qui crée un nouveau NN dont le cerveau a la moitié des poids de du parent1, et la moitié du parent2. Puis mute l'enfant d'un pourcentage taux_mutation"""
    out = NeuralNetwork(parent1.nbInput,parent1.nbHiddenLayer,parent1.nbNeuronPerHL,parent1.nbOutput)
    childWeights = []
    for i in range(len(parent1.weights)):
        rnd = numpy.random.randint(0,2,parent1.shapes[i])
        rndMut = numpy.random.randint(0,2,parent1.shapes[i])
        childWeights.append((rnd*parent1.weights[i] + (1-rnd)*parent2.weights[i]))
        avg = numpy.average(numpy.abs(childWeights[-1]))
        childWeights[-1] = childWeights[-1] + (rndMut*numpy.random.normal(0,taux_mutation,parent1.shapes[i]))*avg # <-- L'amplitude des mutations varie avec la moyenne (abs) des poids de la matrice
    out.setWeight(childWeights)
    return out


class Genetique:
    """ La fonction de crossBreeding genere un individu a partir de deux autres"""
    def __init__(self, nbParents, nbIndivParPop, taux_mutation, evaluateFCT=None, crossBreedingFCT=None, populationTest = None):
        self.nbParents=nbParents
        self.nbIndivParPop = nbIndivParPop
        self.taux_mutation = taux_mutation
        
        self.populationTest = populationTest
        self.evaluateFCT = evaluateFCT
        self.crossBreedingFCT = crossBreedingFCT

        self.evaluateFCT_ok = evaluateFCT!=None
        self.crossBreedingFCT_ok = crossBreedingFCT!=None
        self.populationTest_ok = populationTest!=None

    def setEvaluationFCT(self, fct):
        """La fonction d'evaluation permet de récupérer la fitness d'un individu.
        Elle prend en parametre l'individu a tester et une seed pour le random
        Elle renvoie un triplet, dont la premiere valeur est la fitness 
        """
        self.evaluateFCT = fct
        self.evaluateFCT_ok = True

    def setCrossBreedingFCT(self, fct):
        self.crossBreedingFCT = fct
        self.crossBreedingFCT_ok = True

    def setTestPopulation(self, pop):
        self.populationTest = pop
        self.populationTest_ok = True

    def getIndexOfTopN(self, array, n = None):
        indiceChampion = None
        if n==None:
            n = self.nbParents
        if len(array)<=n:
            return [i for i in range(len(array))]
        else:
            out = []
            alreadyTaken = [False]*len(array)
            for pick in range(n):
                index = None
                max = None
                for k,v in enumerate(array):
                    if(index==None and not alreadyTaken[k]):
                        max = v[0]
                        index = k
                    elif (index!=None and v[0]>max and not alreadyTaken[k]):
                        max = v[0]
                        index = k
                alreadyTaken[index] = True
                out.append(index)
                if indiceChampion==None:
                    indiceChampion = index
        print("Champion : ",array[indiceChampion][0]," antiFit ",array[indiceChampion][1]," pena ",array[indiceChampion][2])
        return out

    def doOneStep(self, quiet = True):
        if self.evaluateFCT_ok and self.crossBreedingFCT_ok and self.populationTest_ok:
            random.seed()
            seed = random.randint(0,100000)
            fitnesses = [self.evaluateFCT(individu,seed) for individu in self.populationTest] # une liste de triple (fitness,[antifit1,antifit2...],[pena1,pena2,..])
            parents = [self.populationTest[i] for i in self.getIndexOfTopN(fitnesses)]
            #print(parents[0].weights)
            newGen = [copy.deepcopy(i) for i in parents]
            while len(newGen)<self.nbIndivParPop:
                i = random.randint(0,self.nbParents-1)
                j = random.randint(0,self.nbParents-1)
                newIndividu = self.crossBreedingFCT(parents[i],parents[j],self.taux_mutation)
                newGen.append(newIndividu)
            self.populationTest = copy.deepcopy(newGen)
            for indiv in self.populationTest:
                indiv.reset()
        else:
            print("Can't do a step in this condition")
        return seed



class GenetiqueForNN(Genetique):
    def __init__(self, nbParents, nbIndivParPop, taux_mutation, evaluateFCT=None, populationTest = None):
        super().__init__(nbParents,nbIndivParPop,taux_mutation, evaluateFCT,crossbreedNN,populationTest)

    def setCrossBreedingFCT(self,fct):
        print("CrossBreedingFCT already set for NN")
