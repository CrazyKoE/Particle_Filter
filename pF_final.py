import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import numpy as np
import random
import operator
import math
from PIL import Image, ImageDraw
from functools import reduce
from math import *

generate_range = 1000 #The number of particles
move_times = 30 #The number of movement of the robot
#Define the Robot
class robot:
    def __init__(self):
        self.x = random.randint(-30,30)
        self.y = random.randint(-30,30)
        self.orientation = random.random() * 2.0 * pi
        #Set the initial location and the direction
        self.forward_noise = 0.0
        self.turn_noise    = 0.0
        self.sense_noise   = 0.0
    
    def set_robot(self, new_x, new_y, new_orientation):
		#Set the location and the direction
        if new_x < -30 or new_x >= 30:
            raise ValueError('X coordinate out of bound')
        if new_y < -30 or new_y >= 30:
            raise ValueError('Y coordinate out of bound')
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0..2pi]')
        self.x = int(new_x)
        self.y = int(new_y)
        self.orientation = int(new_orientation)
        
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # Makes it possible to change the noise parameters
        # This is often useful in particle filters
        # Set the noise of the robot
        self.forward_noise = float(new_f_noise)
        self.turn_noise    = float(new_t_noise)
        self.sense_noise   = float(new_s_noise)

    def move(self, turn, forward):
        #The turn and move forward
        if forward < 0:
            raise ValueError('Robot cannot move backwards')  
        
        # turn, and add randomness to the turning command
        orientation = self.orientation + int(turn) + int(random.gauss(0.0, self.turn_noise))
        orientation %= 2 * pi
        
        # move, and add randomness to the motion command
        dist = int(forward) + random.gauss(0.0, self.forward_noise)
        x = int(self.x + (cos(orientation) * dist))
        y = int(self.y + (sin(orientation) * dist))
        x += 30
        x %= 60
        x -= 30    # cyclic truncate
        y += 30
        y %= 60
        y -= 30
        
        # set particle
        res = robot()
        res.set_robot(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

#Generate the true picture which the robot can see
def DrawingtheCircle(center=np.array([0,0])):
    #Cut the circle area which the drone can see
    img_cov = np.ones((100,100,3),dtype = int)
    for i in range(100):
        for j in range(100):
            distance = ((int(center[0]) - (int(center[0]) - 50 + i))**2 + 
                        (int(center[1]) - (int(center[1]) - 50 + j))**2)**0.5
            if(distance <= 50):
                img_cov[i][j] = img1[int(center[0])-50+i][int(center[1])-50+j]
    return img_cov

#Method 2. Generate the feature sequence of the picture
#Compare the feature sequence and get the similarity percentage of the pictures
def hashImg(imgInput):
    #List of the pixel of the picture
    pixelStorage = []
    #Store the features of the picture
    hashImg = ''
    #Resize the picture to increase the claculate speed
    width,height = 10,10
    imgInput     = imgInput.resize((width,height))
    for y in range(imgInput.height):
        tempStorage = []
        for x in range(imgInput.width):
            pos = x,y
            colorArray = imgInput.getpixel(pos)
            color=sum(colorArray)/3
            tempStorage.append(int(color))
        pixelStorage.append(tempStorage)
    for y in range(imgInput.height):
        #Calculate the average value of each row
        avg=sum(pixelStorage[y])/len(pixelStorage[y])
        #Generate the feature sequence, if the pixel is grater than the average
        #the feature is 1, or the feature is 0
        for x in range(imgInput.width):
            if pixelStorage[y][x]>=avg:
                hashImg+='1'
            else:
                hashImg+='0'
                
    return hashImg

def similar(img1Input,img2Input):
    hash1=hashImg(img1Input)
    hash2=hashImg(img2Input)
    differnce=0
    for i in range(len(hash1)):
        differnce+=abs(int(hash1[i])-int(hash2[i]))
    similar = 1 - differnce/len(hash1)
    return similar

#Method 1. Compare the reference picture with the true picture of the robot
#The bigger the number of the results is, the more different the pictures are.
def pictureCompare(center = np.array([0,0]), x = 0, y = 0):
    #Compare and calculate the similarity of the pictures
    imageZero = img.crop((center[0]-50,center[1]-50,center[0]+50,center[1]+50))
    imageComparing = img.crop((x-50,y-50,x+50,y+50))
    # result = similar(imageZero, imageComparing)
    # return result
    histogram1 = imageZero.histogram()
    histogram2 = imageComparing.histogram()

    results = math.sqrt(reduce(operator.add, list(map(lambda a,b: (a-b)**2,histogram1, histogram2)))/len(histogram1))


    #Method 2.
    #results = similar(imageZero, imageComparing)

    return results

#Calculate the weights
def weightsCal(center,x_input,y_input):
    result = []
    
    for i in range(len(x_input)):
        #divide 20 and times 2 to seperate the features, to increase the weight
        #speed up the preocess of particle resample
        if(pictureCompare(center = center, x = int(x_input[i]), y = int(y_input[i])) < 100):
            result.append(pictureCompare(center = center, x = int(x_input[i]), y = int(y_input[i]))/10)
        else:
            result.append(pictureCompare(center = center, x = int(x_input[i]), y = int(y_input[i]))*2)
    result /= np.sum(result)
    result -= 1/len(x_input)
    for i in range(len(x_input)):
        result[i] = abs(result[i])
    result /= np.sum(result)
    #Method 2. 
    #weights = []
    # for i in range(len(result)):
    #     if(result[i]>0.83):
    #         weights.append(result[i])
    #     else:
    #         weights.append(result[i])
    # weights /= np.sum(weights)
    #np_weights = np.array(weights)
    np_weights = np.array(result)
    return np_weights

#Particle Movement
def predict(particle, u, std, dt=1.):
    #Move according to input u (heading change, velocity)
    #with noise Q (std heading change, std velocity)

    N = len(particle)
    # update heading of the particles
    particle[:,2] += u[0] + (np.random.randn(N) * std[0])
    particle[:,2] %= 2 * pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particle[:, 0] += np.cos(particle[:, 2]) * dist
    #make it never pass the bound.
    particle[:, 0] += 30
    particle[:, 0] %= 60
    particle[:, 0] -= 30

    particle[:, 1] += np.sin(particle[:, 2]) * dist
    particle[:, 1] += 30
    particle[:, 1] %= 60
    particle[:, 1] -= 30
    
    return particle

#particle Resample
def particleResample(particle, weights):
    N = len(particle)
    particle_new = np.empty((len(particle),3))
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, np.random.random(N))
    # resample according to indexes
    particle_new[:] = particle[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights) # normalize  
    return particle_new

#Import the image
plt.figure(1)
img = Image.open('MarioMap.png')
#img2= Image.open('BayMap.png')
img1 = np.array(Image.open('MarioMap.png'))
plt.imshow(img1, origin=None, extent=[-30,30,-30,30])
width,height,n = img1.shape

#Generate particles
particle_plot = np.empty((generate_range,3))
particle = np.empty((generate_range,3))

particle[:,0] = np.random.uniform(0, 3000, generate_range)
particle[:,1] = np.random.uniform(0, 3000, generate_range)
particle[:,2] = np.random.uniform(0, 2*pi, generate_range)
particle[:,2] %= 2 * np.pi
#weights = np.empty(generate_range)
particle_plot[:,0] = particle[:,0]/50-30
particle_plot[:,1] = -particle[:,1]/50+30
particle_plot[:,2] = particle[:,2]
plt.plot(particle_plot[:,0],particle_plot[:,1] ,'.', color = 'red')

#Generate the robot
myrobot = robot()
fig = plt.figure(num = 2, figsize=(14,7))   
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

plt.ion()

for i in range(move_times):
    ax1.cla()
    ax1.imshow(img1, origin=None, extent=[-30,30,-30,30])
    ax1.set_xlim(-30,30)
    ax1.set_ylim(-30,30)
    
    myrobot = myrobot.move(0.0,1.0)
    myrobot.set_noise(0.05,0.1,5.0)#set the Gaussian noise
    ax1.plot(myrobot.x,myrobot.y,'.',color = 'magenta', markersize = 60)
    ax1.plot(particle_plot[:,0],particle_plot[:,1] ,'.', color = 'red')

    particle_plot = predict(particle_plot, u=(0.0, 3.0), std=(.2, .05))#std is the noise
    center_1 = np.array([-(myrobot.y+30)*50,(myrobot.x-30)*50])
    center = np.array([(myrobot.x+30)*50,-(myrobot.y-30)*50])

    img_cov = DrawingtheCircle(center= center_1)
    ax2.imshow(img_cov)

    particle[:,0] = (particle_plot[:,0]+30)*50
    particle[:,1] = -(particle_plot[:,1]-30)*50
    particle[:,2] = particle_plot[:,2]

    #Particle filter
    weights = weightsCal(center,particle[:,0],particle[:,1])

    particle_old = particle

    particle = np.empty((generate_range,2))
    particle = particleResample(particle_old, weights)

    for j in range(len(particle)):
        particle_plot[j][0] = particle[j][0]/50 - 30
        particle_plot[j][1] = -particle[j][1]/50 + 30
    plt.pause(0.2)
plt.ioff()

plt.show()