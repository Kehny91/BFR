import BFR
import NNRocket
import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import math
import numpy
import testRocket
import popManipulator
import sys

WIDTH = 1300
HEIGHT = 900
SCALE = 4 #pix/m
SPRITESCALE = 1.5*SCALE #pix/m

SPEEDMULT = 2

PHY_WIDTH = WIDTH/SCALE
PHY_HEIGHT = HEIGHT/SCALE

pygame.font.init()
font = pygame.font.SysFont('arial', 32)

#print("Width = " + str(PHY_WIDTH) + "m \t Height = " + str(PHY_HEIGHT) + "m\n")

def load_image(name, width, height):
    fullname = os.path.join('sprites', name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error as message:
        print('Cannot load image:', name)
        raise SystemExit(message)
    image = image.convert()
    image.set_colorkey(image.get_at((0,0)), pygame.RLEACCEL)
    image = pygame.transform.scale(image,(width,height))
    return image

#Dans le monde physique, le 0,0 est en bas a gauche avec X horizontal et Y vertical
def phy2pix(vector):
    return (vector.x*SCALE , HEIGHT-vector.y*SCALE)

def pix2phy(coord):
    (x,y) = coord
    return Vector(x/SCALE,(HEIGHT-y)/SCALE)

def blitRocketPositionned(image0, rocket, surf):
    image = pygame.transform.rotate(image0,rocket.mainFrame.theta*BFR.toDeg)
    rect = image.get_rect(center=phy2pix(rocket.mainFrame.pos))
    surf.blit(image,rect)


#initialisation
screen = 0
myTheta0RocketImage = 0
background = 0

#affichage
def update(dt,rocket,throttle,gimbal):
    global screen
    global myTheta0RocketImage
    global background
    rocket.compute(dt,throttle,gimbal)
    screen.blit(background,(0,0))
    blitRocketPositionned(myTheta0RocketImage,rocket,screen)
    affichageThrottle = testRocket.getSurfaceThrottle(rocket.thruster.getThrottle(),throttle,20,100)
    affichageThrottleRect = affichageThrottle.get_rect(topleft = (40,40))
    screen.blit(affichageThrottle,affichageThrottleRect)

    affichageGimbal = testRocket.getSurfaceGimbal(rocket.thruster.getGimbalAngle(),gimbal*rocket.thruster.maxGimbalSweep,rocket.thruster.maxGimbalSweep,100,100)
    affichageGimbalRect = affichageGimbal.get_rect(topleft = affichageThrottleRect.topright)
    screen.blit(affichageGimbal,affichageGimbalRect)
    text = font.render("POS : ( "+ str(math.floor(rocket.mainFrame.pos.x))+", "+str(math.floor(rocket.mainFrame.pos.y))+")", True, (0, 0,0))
    textrect = text.get_rect()
    textrect.move_ip(0,200)
    screen.blit(text,textrect)
    seuil = 10
    platform_width = 30
    if rocket.getPosition().y - rocket.mainFrame.dx/2>0 and rocket.getPosition().y - rocket.mainFrame.dx/2< seuil and abs(rocket.getPosition().x- WIDTH/2/SCALE/2) < platform_width/2: #On est a seuil m du sol
        text = font.render("BONUS")
        textrect = text.get_rect()
        textrect.move_ip(0,300)
        screen.blit(text,textrect)

    pygame.display.flip()

def testRocketNN(dt,rocket,steps,neuralNet):
    global screen
    global myTheta0RocketImage
    global background
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    myTheta0RocketImage =  load_image("rocket.gif",int(rocket.mainFrame.dx*SPRITESCALE),int(rocket.mainFrame.dy*SPRITESCALE))
    pygame.display.set_caption('Test Rocket')
    pygame.mouse.set_visible(0)
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))
    pygame.draw.line(background,(0,0,0),phy2pix(BFR.Vector(0,10)),phy2pix(BFR.Vector(PHY_WIDTH,10)),2)

    NNrep = popManipulator.getNNRepresentation(neuralNet,(250,300))
    NNrepRect = NNrep.get_rect(topright=(WIDTH,0))
    background.blit(NNrep,NNrepRect)

    rocket.goToIntialState()
    neuralNet.reset()
    for i in range(steps):
        top = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit(0)
        neuralNet.setInputs(NNRocket.getInputFromRocket(rocket))
        neuralNet.compute()
        [throttle,gimbal] = neuralNet.getOutputs()
        gimbal = gimbal*2 - 1
        update(dt,rocket,throttle,gimbal)
        while (time.time()-top<dt/SPEEDMULT):
            time.sleep(0.0005)

    background.fill((180, 180, 180))
    screen.blit(background,(0,0))
    pygame.display.flip()


    
