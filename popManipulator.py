import pickle
import os
import pygame
import numpy
import time
import copy
import NN

def getPop(index = None):
    if (not os.path.isdir('saves')):
        os.makedirs('saves')
    files = sorted(os.listdir('saves'))
    if index==None:
        print("AvailablePops:")
        show = [str(i) + " -> " + str(files[i]) +"\n" for i in range(len(files))]
        print("".join(show))
        choix = input("which one (number)\n")
    else:
        choix = str(index)
    file = open(os.path.join("saves",files[int(choix)]),"rb")
    pop = pickle.load(file)
    file.close()
    return pop

def getLabel(neuralNet):
    out = "_" + str(neuralNet.weights[0].shape[0]-1)+"-"
    for k in range(len(neuralNet.weights)):
        out+= str(neuralNet.weights[k].shape[1])
        out+="-"
    out = out[:-1]

    return out

def savePop(pop):
    if (not os.path.isdir('saves')):
        os.makedirs('saves')
    choix = input("Custom name = (leaveblank for autogeneration) (start with + for append to autogeneration\n")
    if choix=="":
        name = time.strftime("%Y-%m-%d_%Hh%M" + getLabel(pop[0]),time.gmtime())+"_size"+str(len(pop))
    elif choix[0]=="+":
        name = time.strftime("%Y-%m-%d_%Hh%M" + getLabel(pop[0]),time.gmtime())+"_size"+str(len(pop)) + choix[1:]
    else:
        name = choix
    file = open(os.path.join('saves',name+".txt"),"wb")
    pickle.dump(pop,file)
    file.close()

def color(weight,maxWeightAmplitude):
    weightScaleColor = 255/maxWeightAmplitude
    R = int(min(255,max(128 + weightScaleColor*weight,0)))
    G = 128
    B = int(min(255,max(128 - weightScaleColor*weight,0)))
    return (R,G,B)

def drawSynapse(surface,p1,p2,weight,maxWeightAmplitude):
    weightScaleThick = 15/maxWeightAmplitude
    pygame.draw.line(surface, color(weight,maxWeightAmplitude), p1, p2, max(1,min(15,int(weightScaleThick*abs(weight)))))

def getTextSurface(text,width,height,color,colorBGN=(255,255,255)):
    font = pygame.font.Font(None,height)
    out = pygame.Surface((width,height))
    out.fill((255,255,255))
    rectOut = out.get_rect()

    text = font.render(text, True, color,colorBGN) 
    rectText = text.get_rect(center=rectOut.center)

    out.blit(text,rectText)

    return out



def getNNRepresentation(neuralNet,size):
    out = pygame.Surface(size)
    out.fill((255,255,255))
    width = out.get_width()
    height = out.get_height()
    radius = int(min(width,height)/(numpy.max(neuralNet.shapes)+1)/2)
    nbTotLayers = len(neuralNet.weights)+1
    posCenters=[]
    maxsPos = []
    maxsNeg = []
    for weight in neuralNet.weights:
        maxsPos.append(numpy.amax(weight))
        maxsNeg.append(numpy.amin(weight))
    maxPos = max([abs(i) for i in maxsPos])
    maxNeg = max([abs(i) for i in maxsNeg])
    maxWeightAmplitude = max(maxPos,maxNeg)
    
    for k in range(nbTotLayers):
        thisLayerX = int(width/(nbTotLayers+1)*(k+1))
        thisNbOfNeuron = getNbNeuronLayer(neuralNet,k,True)
        posCenters.append([ (thisLayerX , int(height/(thisNbOfNeuron+1)*(i+1))) for i in range(thisNbOfNeuron)])
        for center in posCenters[-1]:
            pygame.draw.circle(out, (0,0,0), center, radius, 2)
        if k!=len(neuralNet.weights):
            pygame.draw.circle(out,(180,180,180),posCenters[-1][-1],radius, 2)#le biais

    for k in range(nbTotLayers-1):
        thisNbOfNeuron = getNbNeuronLayer(neuralNet,k,True)
        for i,thisCenter in enumerate(posCenters[k]):
            for j,nextCenter in enumerate(posCenters[k+1][:-1]):
                drawSynapse(out,thisCenter,nextCenter,neuralNet.weights[k][i][j],maxWeightAmplitude)
        for i,thisCenter in enumerate(posCenters[-2]):
            drawSynapse(out,thisCenter,posCenters[-1][-1],neuralNet.weights[-1][i][-1],maxWeightAmplitude)

    labelMax = getTextSurface('{:10.2f}'.format(maxPos),out.get_width()/5,40,color(maxPos,maxWeightAmplitude))
    labelMaxRect = labelMax.get_rect(topright = out.get_rect().topright)
    out.blit(labelMax,labelMaxRect)
    labelMin = getTextSurface("-"+'{:10.2f}'.format(maxNeg),out.get_width()/5,40,color(-maxNeg,maxWeightAmplitude))
    labelMinRect = labelMin.get_rect(bottomright = out.get_rect().bottomright)
    out.blit(labelMin,labelMinRect)


    
    return out

def printWeights(neuralNet):
    for k,weight in enumerate(neuralNet.weights):
        print("Affichage de la matrice n *",k)
        print("[",end="")
        for i in range(len(weight)):
            if i!=0:
                print(" [",end="")
            else:
                print("[",end="")
            for j in range(len(weight[0])):
                if j!=len(weight[0])-1:
                    print('{:10.2f},  '.format(weight[i,j]),end="")
                else:
                    print('{:10.2f}'.format(weight[i,j]),end="")
            if i!=len(weight)-1:
                print("],")
            else:
                print("]",end="")
        print("]")


def getNbNeuronLayer(neuralNet,k,includeBiais):
    if includeBiais:
        includeBiais = 1
    else:
        includeBiais = 0

    if k==0:
        return neuralNet.shapes[0][0] - 1 + includeBiais
    elif k==len(neuralNet.weights):
        return neuralNet.shapes[k-1][1]
    else:
        return neuralNet.shapes[k-1][1] + includeBiais

def convertNN(neuralNet,formatString):
    assert formatString.count("-") == (getLabel(neuralNet)[1:]).count("-"), "I can't add/remove a whole layer yet"
    l = formatString.split("-")
    nbHiddenLayer = 0
    nbNeuronPerHiddenLayer = 0
    if len(l)>=3:
        nbHiddenLayer = len(l)-2
        nbNeuronPerHiddenLayer = int(l[1])
        compareTo = l[1]
        for i in l[1:-1]:
            assert i == compareTo , "I can't have hidden layers of different sizes"

    targetNbNeuronsLayerWithoutBias = [int(s) for s in formatString.split("-")]
    actualNbNeuronsLayerWithoutBias = [int(s) for s in (getLabel(neuralNet)[1:]).split("-")]
    outWeights = copy.deepcopy(neuralNet.weights)
    for k in range(len(neuralNet.weights)):
        diffEntree = targetNbNeuronsLayerWithoutBias[k] - actualNbNeuronsLayerWithoutBias[k]
        if diffEntree<0: #reduction de l'entree de la matrice
            lastLigne = copy.deepcopy(neuralNet.weights[k][-1,:])
            outWeights[k] = copy.deepcopy(neuralNet.weights[k][:targetNbNeuronsLayerWithoutBias[k]+1])
            outWeights[k][-1,:] = lastLigne
        elif diffEntree>0: #augmentation de l'entree de la matrice
            lastLigne = copy.deepcopy(neuralNet.weights[k][-1,:])
            for p in range(len(neuralNet.weights[k][-1,:])):
                neuralNet.weights[k][-1,p] = 0
            outWeights[k] = numpy.pad(neuralNet.weights[k],((0,diffEntree),(0,0)), mode = 'constant') # def lignes de 0 ont été rajoutées. Mais on veut garder les valeurs de la dernière ligne (biais)
            (outWeights[k])[-1,:] = lastLigne
        else:
            outWeights[k] = copy.deepcopy(neuralNet.weights[k])

        diffSortie = targetNbNeuronsLayerWithoutBias[k+1] - actualNbNeuronsLayerWithoutBias[k+1]
        if diffSortie<0: #reduction de la sortie de la matrice
            outWeights[k] = copy.deepcopy(outWeights[k][:,:targetNbNeuronsLayerWithoutBias[k+1]])
        elif diffSortie>0: #augmentation de la sortie de la matrice
            outWeights[k] = numpy.pad(outWeights[k],((0,0),(0,diffSortie)), mode = 'constant')

    out = NN.NeuralNetwork(outWeights[0].shape[0]-1,nbHiddenLayer,nbNeuronPerHiddenLayer,outWeights[-1].shape[1])
    out.setWeight(copy.deepcopy(outWeights))
    return out

        

def init(width,height):
    pygame.init()
    screen = pygame.display.set_mode((width, height))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    screen.blit(background,(0,0))
    return screen

def quickGraph(neuralNet):
    screen = init(800,800)
    screen.blit(getNNRepresentation(neuralNet,screen.get_size()),(0,0))
    pygame.display.flip()

if __name__ == "__main__":
    pop = getPop()
    for individu in pop:
        quickGraph(individu)
        time.sleep(0.3)