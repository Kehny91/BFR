#Librairie définissant une Rocket

from math import *
import random

toRad=pi/180
toDeg=180/pi
g_0=9.81
rho_0=1.225
TURBULENCE=0  #N ou N.m

def moduloF(x, modulo):
    if x>=0:
        return x - (x//modulo)*modulo
    else:
        return x - (x//modulo+1)*modulo

def normalize(angle):
    """ Permet de renvoyer un angle quelconque dans l'intervalle -pi pi"""
    angle = moduloF(angle, 2*pi)

    if -pi<angle and angle<=pi:
        return angle
    elif angle<=-pi:
        return normalize(angle + 2*pi)
    elif angle>pi:
        return normalize(angle - 2*pi)
    else:
        assert False,"angle not real: " + str(angle)

class Vector:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __add__(self,other):
        return Vector( self.x+other.x , self.y+other.y )

    def __sub__(self,other):
        return Vector( self.x-other.x , self.y-other.y )

    def __mul__(self,other):
        """ATTENTION produit scalaire avec un autre vecteur, mais multiplication (.) avec un scalaire"""
        if type(other)==Vector:
            return self.x*other.x + self.y*other.y
        else:
            try:
                return Vector( self.x*other , self.y*other )
            except:
                None

    def __pow__(self,other):
        """produit vectoriel"""
        return self.x*other.y - self.y*other.x
    
    def angle(self):
        return atan2( self.y , self.x )

    def norm(self):
        return sqrt((self.x**2) + (self.y**2))

    def unitaire(self):
        if self.x==0 and self.y==0:
            return Vector(1,0) #arbitraire
        else:
            n=self.norm()
            return Vector(self.x/n,self.y/n)

    def rotate(self,angle):
        x=self.x*cos(angle)-self.y*sin(angle)
        y=self.x*sin(angle)+self.y*cos(angle)
        return Vector(x,y)

    def __str__(self):
        return "( "+str(self.x)+" , "+str(self.y)+" )"

def directeur(angle):
    return Vector(cos(angle),sin(angle))

class Attachement:
    def __init__(self,x,y,theta,mass,father):
        """Attention, lors de l'initialisation de l'attachement, la position par rapport father est alors figée !"""
        self.pos=Vector(x,y)
        self.v=Vector(0,0)
        self.theta=theta
        self.w=0
        self.mass=mass
        self.father=father
        if father!=None:
            father.addAttachement(self)

class Aileron(Attachement):
    def __init__(self,x,y,theta,CzA,S,Cx0,allongement,qualite,mass=0,decrochage=15*toRad,father=None):
        super().__init__(x,y,theta,mass,father)
        self.CzA=CzA     #dCz/dAlpha
        self.S=S         #Surface d'aile projetee
        self.Cx0=Cx0
        self.allongement=allongement
        self.qualite=qualite
        self.decrochage=decrochage

    def Cz(self,Alpha):
        return self.CzA*sin(Alpha)

    def lift(self,v,rho,Alpha):
        if -1*self.decrochage<=Alpha and Alpha<=self.decrochage:            #VOL NORMAL
            return 0.5*rho*self.S*(v**2)*self.Cz(Alpha)
        elif -pi+self.decrochage<=Alpha and Alpha<=pi-self.decrochage:      #DECROCHAGE
            return 0    
        else:                                                               #BACKWARD FLIGHT
            return -1*0.5*rho*self.S*(v**2)*self.Cz(Alpha)

    def drag(self,v,rho,Alpha):
        return 0.5*rho*self.S*(v**2)*((self.Cx0) + (self.Cz(Alpha)**2)/(pi*self.allongement*self.qualite))

    def vectorAeroForce(self,rho=rho_0):   #Vecteur V du deplacement de l'aileron dans l'air statique
        """Renvoie le vecteur des forces aero sur l'aileron"""
        alpha=(self.theta-self.v.angle())
        v=self.v.norm()
        vU=self.v.unitaire()
        vUN=vU.rotate(pi/2)
        return vUN*self.lift(v,rho,alpha) - vU*self.drag(v,rho,alpha)
        
def constrain(x,m,M):
    if x<=m:
        return m
    elif x<=M:
        return x
    else:
        return M

class Thruster(Attachement):
    def __init__(self,x,y,theta,maxThrust,maxGimbalSweep,ISP,mass,maxWGimbal = 45.0 * toRad,maxDThrottle = 1.00/0.5, father=None): #max 45.0 deg par seconde, et 100% de throttle en 500ms
        super().__init__(x,y,theta,mass,father)
        self.maxThrust=maxThrust
        self.maxGimbalSweep=maxGimbalSweep
        self.maxWGimbal = maxWGimbal
        self.maxDThrottle = maxDThrottle
        self.lastGimbalAngle = 0
        self.lastThrottle = 0
        self.ISP=ISP

    def thrust(self,throttle,gimbal,dt):
        requestedGimbalAngle = constrain(gimbal,-1,1)*self.maxGimbalSweep
        reachableMinGimbal = max( - self.maxGimbalSweep , self.lastGimbalAngle - self.maxWGimbal*dt) # Le max qu'on peut atteindre en un temps dt
        reachableMaxGimbal = min(   self.maxGimbalSweep , self.lastGimbalAngle + self.maxWGimbal*dt) # Le min qu'on peut atteindre en un temps dt

        requestedThrottle = throttle
        reachableMinThrottle = max(0, self.lastThrottle - self.maxDThrottle*dt)
        reachableMaxThrottle = min(1, self.lastThrottle + self.maxDThrottle*dt)
        self.lastGimbalAngle = constrain(requestedGimbalAngle,reachableMinGimbal,reachableMaxGimbal)
        self.lastThrottle = constrain(requestedThrottle,reachableMinThrottle,reachableMaxThrottle)
        return directeur((self.theta + self.lastGimbalAngle)) * (self.maxThrust*self.lastThrottle)
    
    def massFlow(self,throttle):
        return self.maxThrust * throttle / (self.ISP*g_0)

    def getGimbalAngle(self):
        return self.lastGimbalAngle

    def getThrottle(self):
        return self.lastThrottle

class RigidBody(Attachement):
    """Un Rigid Body gère les collisions. Il est referencé par son centre de masse/geometrie"""
    def __init__(self,dx,dy,theta,x,y,mass,father=None):
        super().__init__(x,y,theta,mass,father)
        self.dx=dx
        self.dy=dy
        self.attachements=[]
        self.isRoot=(father==None)

    def addAttachement(self,device):
        self.attachements.append(  (device , device.pos-self.pos , device.theta))

    def computeGeometry(self):
        """Actualise l'attitude des attachements en fonction """
        if self.isRoot:
            for (dev,deltaPos,deltaTheta) in self.attachements:
                dev.theta=normalize(self.theta+deltaTheta)
                dev.pos=self.pos + deltaPos.rotate(self.theta)
                dev.v=self.v+(deltaPos.rotate(self.theta+pi/2).unitaire())*(deltaPos.norm()*self.w)
        else:
            print("can't compute geometry of non root piece")

    def move(self,dx,dy,dtheta):
        if self.isRoot:
            self.pos=self.pos+Vector(dx,dy)
            self.theta=normalize(self.theta+dtheta)
            self.computeGeometry()
        else:
            print("can't move an attached piece")

    def setPosition(self,x,y,theta):
        if self.isRoot:
            self.pos=Vector(x,y)
            self.theta=normalize(theta)
            self.computeGeometry()
        else:
            print("can't set the position of an attached piece")

    def setVelocity(self,Vx,Vy,W):
        if self.isRoot:
            self.v = Vector(Vx,Vy)
            self.w = W
            self.computeGeometry()
        else:
            print("can't set velocity of an attached piece")


    def getCG_Mass_J(self):
        num=self.pos*self.mass
        denom=self.mass
        j=self.dx*self.dy*((self.dx**2)+(self.dy*2))/12 #Moment d'inertie d'un rectangle
        for (dev,deltaPos,deltaTheta) in self.attachements:
            num=num + dev.pos*dev.mass
            denom+= dev.mass
            j+=(deltaPos.norm()**2)*dev.mass # approximation: les attachements sont des points
        return (num*(1/denom) , denom,j)  #approximation: le centre de masse est au centre geo
        
    

class RocketClassique:
    ID = 1
    def __init__(self, dx, dy, mass, maxThrust, maxGimbalSweep, ISP, aileronCzA, aileronTotalS, x=0, y=0, theta=0, vx=0, vy=0, w=0):
        self.uniqueID = RocketClassique.ID
        RocketClassique.ID+=1
        self.mainFrame=RigidBody(dx, dy, 0, 0, 0, mass, father=None)
        self.thruster=Thruster(-dx/2,0,0,maxThrust,maxGimbalSweep,ISP,mass/10,father=self.mainFrame) #Le thruster est en bas de la frame
        self.fins=Aileron(-dx/2,0,0,aileronCzA,aileronTotalS,aileronCzA/10,1.0,0.5,mass=0,decrochage=15*toRad,father=self.mainFrame) #Les ailerons sont en bas de la frame et ont 10 de finesse
        self.mainFrame.setPosition(x,y,theta)
        self.mainFrame.setVelocity(vx,vy,w)
        self.last_throttle = 0
        self.last_gimbal = 0
        self.pos0 = Vector(x,y)
        self.theta0 = theta
        self.v0 = Vector(vx,vy)
        self.w0 = w

    def goToIntialState(self):
        self.mainFrame.setPosition(self.pos0.x,self.pos0.y,self.theta0)
        self.mainFrame.setVelocity(self.v0.x,self.v0.y,self.w0)
        
    def move(self,dx,dy,dtheta):
        self.mainFrame.move(dx,dy,dtheta)

    def setPosition(self,x,y,theta):
        self.mainFrame.setPosition(x,y,theta)

    def setVelocity(self,Vx,Vy,W):
        self.mainFrame.setVelocity(Vx,Vy,W)

    def getPosition(self):
        return self.mainFrame.pos
    
    def getTheta(self):
        return self.mainFrame.theta

    def getVelocity(self):
        return self.mainFrame.v
    
    def getW(self):
        return self.mainFrame.w

    def compute(self,dt,throttle,gimbal):
        self.mainFrame.computeGeometry()
        forces=Vector(0,0)
        moments=0
        for (dev,deltaPos,deltaTheta) in self.mainFrame.attachements:
            if (type(dev)==Thruster):
                force=dev.thrust(throttle,gimbal,dt)
            elif (type(dev)==Aileron):
                force=dev.vectorAeroForce()
            else:
                force=Vector(0,0)
            forces=forces+force
            moments+=deltaPos.rotate(dev.theta)**force
        #moments au centre de la frame.
        (cg,m,j)=self.mainFrame.getCG_Mass_J()
        moments=moments+(self.mainFrame.pos-cg)**forces #+ random.random()*TURBULENCE #transfert des moments au CG (cf formule de BABAR)
        forces=forces+Vector(0,-1*g_0*m) #+ (Vector(1,0)*random.random()+Vector(0,1)*random.random())*TURBULENCE #ajout de la gravité

        #bilan et integration
        self.mainFrame.v=self.mainFrame.v + forces*(dt/m)
        self.mainFrame.w=self.mainFrame.w + moments*(dt/j)
        self.mainFrame.pos=self.mainFrame.pos+self.mainFrame.v*dt
        self.mainFrame.theta=normalize(self.mainFrame.theta+self.mainFrame.w*dt)