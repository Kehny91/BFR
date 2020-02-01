import pygame
import BFR
import time
import os

WIDTH = 600
HEIGHT = 800
SCALE = 4 #pix/m
SPRITESCALE = 2*SCALE #pix/m

PHY_WIDTH = WIDTH/SCALE
PHY_HEIGHT = HEIGHT/SCALE

DT = 0.03
TIMESCALE = 1 #the more, the slower will be the animation

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
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
myTheta0RocketImage =  load_image("fusee.jpg",10*SPRITESCALE,3*SPRITESCALE)
pygame.display.set_caption('Test Rocket')
pygame.mouse.set_visible(0)
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((250, 250, 250))

#modele
myRocket = BFR.RocketClassique(10,3,30000,1500000,4*BFR.toRad,460,5.0,4,PHY_WIDTH/2,PHY_HEIGHT/2,BFR.pi/2)

throttle = 0.05
gimbal = 0

def autopilot(fusee):
    gainThrottle = 5.0
    gainPAngulaire = 1.5
    gainKAngulaire = 0.4
    gainPCorrectionVx = 0.08
    crossGain = 0.1
    errorVY = max(0,0 - fusee.getVelocity().y)
    errorAng = BFR.pi/2 - fusee.getTheta()
    errorVX = fusee.getVelocity().x

    throttle_loc = gainThrottle*errorVY
    gimbal_loc = -1*gainPAngulaire*errorAng -1*gainPCorrectionVx*errorVX + gainKAngulaire*fusee.getW()
    throttle_loc += abs(gimbal_loc)*crossGain

    return (throttle_loc,gimbal_loc)



useAutopilot = -1

#affichage
def update(dt):
    global throttle
    global gimbal
    if useAutopilot==1:
        (throttle,gimbal) = autopilot(myRocket)
    myRocket.compute(dt,throttle,gimbal)
    screen.blit(background,(0,0))
    blitRocketPositionned(myTheta0RocketImage,myRocket,screen)
    pygame.display.flip()


stop = False
while not stop:
    top = time.time()
    update(DT/TIMESCALE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stop = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            stop = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            throttle = 1
        elif event.type == pygame.KEYUP and event.key == pygame.K_UP:
            throttle = 0.05
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            gimbal = 1
        elif event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
            gimbal = 0
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
            gimbal = -1
        elif event.type == pygame.KEYUP and event.key == pygame.K_LEFT:
            gimbal = 0
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            myRocket.goToIntialState()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
            useAutopilot*=-1
            throttle = 0.05
            gimbal = 0
    while (time.time()-top<DT):
        time.sleep(0.0005)
