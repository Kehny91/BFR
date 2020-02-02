import pygame
import BFR
import NNRocket
import time
import os
import numpy

WIDTH = 600
HEIGHT = 800
SCALE = 4 #pix/m
SPRITESCALE = SCALE #pix/m

SPEEDMULT = 2

PHY_WIDTH = WIDTH/SCALE
PHY_HEIGHT = HEIGHT/SCALE

print("Width = " + str(PHY_WIDTH) + "m \t Height = " + str(PHY_HEIGHT) + "m\n")

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
pygame.font.init()
font = pygame.font.SysFont('arial', 32)

def getSurfaceThrottle(throttle,width,height):
    out = pygame.Surface((width,height))
    out.fill((0,0,0))
    rect = out.get_rect()

    inner = pygame.Surface((width-4,height-4))
    inner.fill((255,255,255))
    innerRect = inner.get_rect(center=rect.center)

    out.blit(inner,innerRect)

    bar = pygame.Surface((width-4,int((height-4)*throttle)))
    bar.fill((255,120,120))
    barRect = bar.get_rect(bottomleft=innerRect.bottomleft)
    out.blit(bar,barRect)
    
    return out
"""
def getSurfaceGimbal(gimbalAngleRad,width,height):
    out = pygame.Surface((width,height))
    out.fill((0,0,0))
    rect = out.get_rect()

    posBase = (int(width/2),5)
    pygame.draw.circle(out, (0,0,0), (int(width/2),5), 10)

    posTarget = posBase[0] - 
    
    
    return out"""


#affichage
def update(dt,rocket,throttle,gimbal):
    global screen
    global myTheta0RocketImage
    global background
    rocket.compute(dt,throttle,gimbal)
    screen.blit(background,(0,0))
    blitRocketPositionned(myTheta0RocketImage,rocket,screen)
    #text = font.render("Throttle = "+str( throttle ), True, (0, 0,0))
    #textrect = text.get_rect()
    affichageThrottle = getSurfaceThrottle(throttle,20,100)
    screen.blit(affichageThrottle,(40,40))
    #text = font.render("Gimbal = "+str( gimbal ), True, (0, 0,0))
    #textrect = text.get_rect()
    #textrect.move_ip(0,40)
    #screen.blit(text,textrect)
    pygame.display.flip()

def testRocketNN(dt,rocket,steps,neuralNet):
    global screen
    global myTheta0RocketImage
    global background
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    myTheta0RocketImage =  load_image("fusee.jpg",rocket.mainFrame.dx*SPRITESCALE,rocket.mainFrame.dy*SPRITESCALE)
    pygame.display.set_caption('Test Rocket')
    pygame.mouse.set_visible(0)
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))
    pygame.draw.line(background,(0,0,0),phy2pix(BFR.Vector(0,10)),phy2pix(BFR.Vector(PHY_WIDTH,10)),2)
    

    rocket.goToIntialState()
    neuralNet.reset()
    for i in range(steps):
        top = time.time()
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


    
