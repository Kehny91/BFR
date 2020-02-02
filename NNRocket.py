import NN
import BFR
import numpy
import testRocketNN
import math
import time
import multiprocessing
import copy
import pickle
import os
import datetime
import random
import popManipulator

NBINDIVPOP = 100 #nbIndiv/pop
DT = 0.03
STEPS = 1000
genetique = NN.GenetiqueForNN(10,NBINDIVPOP,0.5)
NBESSAI = 1

def getInputFromRocket(rocket):
    v = rocket.getVelocity() #On veut quasi-saturer a 100m/s
    pos = rocket.getPosition() #On veut quasi-saturer a 2000m/s
    theta = rocket.getTheta() #On veut quasi-saturer a 3.14rad
    w = rocket.getW() #On veut quasi-saturer a 3.14rad/s
    quasiSaturation = NN.MAXSIGMOID/2
    return numpy.array([pos.x/2000*quasiSaturation, pos.y/2000*quasiSaturation, v.x/100*quasiSaturation, v.y/100*quasiSaturation, math.sin(theta)*quasiSaturation, math.cos(theta)*quasiSaturation, w/3.14*quasiSaturation])


def evalLanding(neuralNet,seed):
    random.seed(seed)

    amplitudePosition = 50
    amplitudeTheta = 15 * BFR.toRad
    amplitudeVitesse = 25
    amplitudeW = 15 * BFR.toRad

    antifitness = [0]
    penalty = [0,0]

    for essai in range(NBESSAI):
    
        myRocket = BFR.RocketClassique(10,3,30000,1500000,4*BFR.toRad,460,5.0,4,
                                theta=BFR.pi/2 + (random.random()*2-1)*amplitudeTheta,
                                x=testRocketNN.PHY_WIDTH/2 + (random.random()*2-1)*amplitudePosition,
                                y=4*testRocketNN.PHY_HEIGHT/5 + (random.random()*2-1)*amplitudePosition,
                                vx=(random.random()*2-1)*amplitudeVitesse/4,
                                vy=(random.random()*(-1))*amplitudeVitesse,
                                w=(random.random()*2-1)*amplitudeW)
        throttle = 0
        gimbal = 0


        for i in range(STEPS):
            neuralNet.setInputs(getInputFromRocket(myRocket))
            neuralNet.compute()    
            [throttle, gimbal] = neuralNet.getOutputs()
            gimbal = gimbal*2 - 1
            myRocket.compute(DT,throttle,gimbal)
            if myRocket.getPosition().y - myRocket.mainFrame.dx/2<0:
                penalty[0] += (-20*(myRocket.getPosition().y- myRocket.mainFrame.dx/2))*DT
            seuil = 10
            if myRocket.getPosition().y - myRocket.mainFrame.dx/2>0 and myRocket.getPosition().y - myRocket.mainFrame.dx/2<seuil: #On est a seuil m du sol
                penalty[1] -= 20*(seuil-(myRocket.getPosition().y- myRocket.mainFrame.dx/2))*DT
            #Statique Pos
            antifitness[0] += (math.sqrt((myRocket.getPosition().x-testRocketNN.WIDTH/2/testRocketNN.SCALE)**2 + (myRocket.getPosition().y - myRocket.mainFrame.dx/2)**2)) * DT *0.1
    return (-sum(antifitness)-sum(penalty),antifitness,penalty)


def evalHover(neuralNet,seed):
    random.seed(seed)

    amplitudePosition = 30
    amplitudeTheta = 15 * BFR.toRad
    amplitudeVitesse = 20
    amplitudeW = 15 * BFR.toRad

    antifitness = [0,0,0]
    penalty = [0]

    for essai in range(NBESSAI):
    
        myRocket = BFR.RocketClassique(10,3,30000,1500000,4*BFR.toRad,460,5.0,4,
                                theta=BFR.pi/2 + (random.random()*2-1)*amplitudeTheta,
                                x=testRocketNN.WIDTH/2/testRocketNN.SCALE + (random.random()*2-1)*amplitudePosition,
                                y=testRocketNN.WIDTH/2/testRocketNN.SCALE+ (random.random()*2-1)*amplitudePosition,
                                vx=(random.random()*2-1)*amplitudeVitesse,
                                vy=(random.random()*2-1-0.2)*amplitudeVitesse,
                                w=(random.random()*2-1)*amplitudeW)

        throttle = 0
        gimbal = 0

        for i in range(STEPS):
            neuralNet.setInputs(getInputFromRocket(myRocket))
            neuralNet.compute()    
            [throttle, gimbal] = neuralNet.getOutputs()
            gimbal = gimbal*2 - 1
            myRocket.compute(DT,throttle,gimbal)
            if myRocket.getTheta()<0:
                penalty[0] += 1000 *abs(math.sin(myRocket.getTheta()))*DT
            #Dynamique Ang
            antifitness[0] += abs(myRocket.getW())*DT * 2
            antifitness[1] += abs(BFR.normalize(myRocket.getTheta()-math.pi/2))*DT*5
            #Dynamique Pos
            antifitness[2] += myRocket.getVelocity().norm() * DT
    return (-sum(antifitness)-sum(penalty),antifitness,penalty)

if __name__ == "__main__":
    #genetique.setEvaluationFCT(evalHover)
    genetique.setEvaluationFCT(evalLanding)
    
    choix = input("Do you want to use an old pop ? (Y/N)\n")
    if choix=="Y":
        pop = popManipulator.getPop()
    else:
        pop = []
        for i in range(NBINDIVPOP):
            pop.append(NN.NeuralNetwork(7,2,7,2))
            pop[-1].createRandomWeight(0,NN.MAXSIGMOID)

    genetique.setTestPopulation(pop)

    process = None
    for i in range(600):
        print("Gen ",i)
        seed = genetique.doOneStep(quiet=False)
        if process !=None:
            process.join()

        random.seed(seed)

        amplitudePosition = 50
        amplitudeTheta = 15 * BFR.toRad
        amplitudeVitesse = 25
        amplitudeW = 15 * BFR.toRad
                
        myRocket = BFR.RocketClassique(10,3,30000,1500000,4*BFR.toRad,460,5.0,4,
                                    theta=BFR.pi/2 + (random.random()*2-1)*amplitudeTheta,
                                    x=testRocketNN.PHY_WIDTH/2 + (random.random()*2-1)*amplitudePosition,
                                    y=4*testRocketNN.PHY_HEIGHT/5 + (random.random()*2-1)*amplitudePosition,
                                    vx=(random.random()*2-1)*amplitudeVitesse/4,
                                    vy=(random.random()*(-1))*amplitudeVitesse,
                                    w=(random.random()*2-1)*amplitudeW)

        copyChampion=copy.deepcopy(genetique.populationTest[0])
        process = multiprocessing.Process(target=testRocketNN.testRocketNN, args=(DT,myRocket,STEPS,copyChampion))
        process.start()

    choix = input("Do you want to save this pop ? (Y/N)\n")
    if choix=="Y":
        popManipulator.savePop(genetique.populationTest)

