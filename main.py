# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:56:52 2017

@author: mkeranen

This script measures the diameter of a feature in an image using 3 points
selected on the circumference of the feature and 3 points selected on the
circumference of a known dimension reference feature.
"""


#Import
import cv2
from win32api import GetSystemMetrics
from sympy import Symbol, nsolve
import mpmath
import math


mpmath.mp.dps = 15
rightClicks = []
pi = 3.14159

#Functions
def is_number(num):
    """For detecting if parameter is a number. If 'num' cannot be typecast
    to float, assume that it's not a number."""
    try:
        float(num)
        return True
    except ValueError:
        return False

def get_sample_info():
    """
    Gets info from user about the sample being measured. Includes:
    -Serial Number
    -Descriptor
    -Reference object dimension
    
    Takes no parameters, returns sample serial number, whether the sample is
    the inlet or the outlet, and the dimension of the reference object.
    """
    while True:
        try:
            serialNumber = input('Enter the sample serial number being measured: ')
        except ValueError:
            print('Invalid. Please re-enter serial number.')
            continue
        if serialNumber.isnumeric() == False:
            print('Serial number must consist of only numeric characters. Re-enter serial number.')
            continue
        if len(serialNumber) != 8:
            print('Serial number must be 8 digits long. Ex: 07412345. Re-enter serial number.')
            continue
        else:
            break
    
    while True:
        try:
            descriptor = input('Enter short (1 or 2 word) description of sample: ')
        except ValueError:
            print('Invalid. Please re-enter descriptor.')
            continue
        if len(descriptor) > 15:
            print('Please shorten descriptor length to 15 or fewer characters. Please re-enter descriptor.')
            continue
        else:
            break
        
    while True:
        try:
            referenceDimension = input('Enter the measured dimension of the reference object in inches: ')
        except ValueError:
            print('Invalid. Please re-enter reference dimension.')
            continue
        if is_number(referenceDimension) == False:
            print('Dimension must be a number only. Re-enter reference dimension.')
            continue
        else:
            referenceDimension = float(referenceDimension)
            break

    return serialNumber, descriptor, referenceDimension

def get_picture_to_analyze():
    """    
    This function gets the picture to be used for measurement.
    
    Takes no parameters, returns path to image. Future versions may include
    picture taking/saving functionality
    """
    
    #Validate user input. Require working image path.
    while True:
        try:
            imagePath = str(input('Enter path to image, including extension: '))
        except ValueError:
            print('Invalid. Please re-enter path to image.')
            continue
        if  '.' not in imagePath:
            print('Ensure that image path includes extension. Please re-enter image path.')
            continue
        if cv2.imread(imagePath,1) == None:
            print('Image does not exist at specified filepath. Please re-enter image path.')
            continue
        else:
            break
        
    return imagePath
        
def image_preparation(imagePath,color=0):
    """Prepares the image and image window for processing and capture of 
    mouse click coordinates
    
    Parameters include image path and whether the image should be loaded in 
    color(1) or grayscale(0); default is grayscale
    
    Returns the image in a processed and prepared window
    """
    img = cv2.imread(imagePath,color)
    
    #Get height & width of screen
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)
    
    #Scale the window to fit in the screen
    scale_width = width / img.shape[1]
    scale_height = height / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    
    #Create named window and resize for screen
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    
    return img

def mouse_callback(event, x, y, flags, params):
    """Sets up the mouse click tracking"""
    
    #right-click event value is 2
    if event == 2:
        global rightClicks

        #store the coordinates of the right-click event
        rightClicks.append([x, y])
        
        #After user selects 3 points on potting diameter, switch to reference diameter
        if len(rightClicks)==3:
            print('\nSelect three points along the circumference of the reference diameter.\n')
        
        #When all points are selected, close windows
        elif len(rightClicks)==6: 
            cv2.destroyAllWindows()
            
def calculate_pix_circle(p1, p2, p3):
    """Calculates a diameter when given 3 points using the equation of a 
    circle and system of equations solver
    
    Uses the nsolve function from sympy
    
    Parameters: 3 (x,y) coordinates
    Returns: Center of circle at (xc,yc) and radius of circle
    """
    
    #Required to make a first guess for parameters being solved for
    #Center x,y guesses are just the average of the 3 points
    avgX = float(p1[0] + p2[0] + p3[0])/3
    avgY = float(p1[1] + p2[1] + p3[1])/3
    
    #The guess for radius is just the hypotenuse of points 1 and 2
    radiusGuess = math.hypot(p2[0]-p1[0], p2[1] - p1[1])
    
    firstGuess = (avgX, avgY, radiusGuess)
    
    #Set symbols for use in equations required for nsolve
    x = Symbol('x')
    y = Symbol('y')
    r = Symbol('r')
    
    #Equations of a circle using each of the selected 3 points
    f1 = x**2 + y**2 - 2*x*p1[0] - 2*y*p1[1] + p1[0]**2 + p1[1]**2 - r**2 #Equation 1
    f2 = x**2 + y**2 - 2*x*p2[0] - 2*y*p2[1] + p2[0]**2 + p2[1]**2 - r**2 #Equation 2
    f3 = x**2 + y**2 - 2*x*p3[0] - 2*y*p3[1] + p3[0]**2 + p3[1]**2 - r**2 #Equation 3
    
    #Solve the system of equations for (x,y,r) using nsolve and the first guess
    xc, yc, radius = nsolve((f1,f2,f3), (x,y,r), (firstGuess))
    xc = float(xc)
    yc = float(yc)
    radius = float(radius)
    print('Center at (px): (x,y) ({:0.3f},{:0.3f}); Radius (px): {:0.3f}'.format(xc, yc, radius))
    return xc, yc, radius

def triangle_area_to_circle_area_ratio(p1,p2,p3,radius):
    """Checks if user selected points that are sufficiently spaced apart to 
    increase measurement precision. Compares the area of a triangle created by
    the 3 points selected to the area of the circle made by the 3 points selected.
    
    If the ratio is too low, it notifies the user. 
    
    TODO: Force user to pick quality points
    
    Parameters: 3 (x,y) coordinates and a radius
    Returns: Triangle Area / Circle Area
    """
    
    #Find distance between all points
    a = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) #Dist between 1 & 2
    b = math.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2) #Dist between 1 & 3
    c = math.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2) #Dist between 2 & 3

    #Triangle Area is equal to sqrt(p*(p-a)*(p-b)*(p-c));
    #Where: 
        # p = half perimeter of the triangle
        # a, b, c are the distances between the points
        
    p = (a + b + c) / 2
    
    triangleArea = math.sqrt(p*(p-a)*(p-b)*(p-c))
    
    #Circle Area:
    circleArea = pi * radius**2
    
    #Ratio
    ratio = triangleArea / circleArea
    return ratio

def convert_from_pixels_to_inches(refPixelLength, actualRefLength, sampleMsmtPixels):
    """Converts a length from Pixels to units of the reference dimension.
    
    Parameters: dimension of reference object in pixels, actual reference object
    dimension, dimension of sample object in pixels
    
    Returns: dimension of sample object in reference units
    """
    # 2x multiplier to convert radii to diameters
    lengthOfSampleMeasurementInRefUnits = (actualRefLength / (2 * refPixelLength)) * (2 * sampleMsmtPixels)
    
    return lengthOfSampleMeasurementInRefUnits

def analysis_results(serialNumber, descriptor, referenceDimension, sampleMeasurement, refCirclePx, sampleCirclePx, img):
    """Exports results of sample measurements.
    Draws circles created by user point selection, prints measurement results, 
    overlays text on samlple image to identify sample.
    
    Parameters: Sample serial number, dimension of reference object, measurement of 
    sample in reference units, circle dimensions in pixels for reference and 
    sample features, image to overlay results on
    
    Returns: Image with overlaid results
    """
    
    
    print('\nSample diameter measurement: {:0.3f}'.format(sampleMeasurement))
    
    #Create strings to overlay on image
    refDiameterString = 'Ref Diameter: {:0.3f} '.format(referenceDimension)
    sampleDiameterString  = 'Measured Diameter: {:0.3f} '.format(sampleMeasurement)
    
    #Draw circles using selected points, add sample identifier text, measurement results
    img = cv2.circle(img, (int(refCirclePx[0]),int(refCirclePx[1])), int(refCirclePx[2]), (0,0,255), 4)
    img = cv2.circle(img, (int(sampleCirclePx[0]),int(sampleCirclePx[1])), int(sampleCirclePx[2]), (255,0,0), 4)
    img = cv2.putText(img, refDiameterString, (int(refCirclePx[0]),int(refCirclePx[1]+refCirclePx[2])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 8)
    img = cv2.putText(img, sampleDiameterString, (int(sampleCirclePx[0]),int(sampleCirclePx[1]+sampleCirclePx[2])), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 8)
    img = cv2.putText(img, serialNumber, (int(sampleCirclePx[0]),int(sampleCirclePx[1]-sampleCirclePx[2]/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 8)
    img = cv2.putText(img, descriptor, (int(sampleCirclePx[0]),int(sampleCirclePx[1]-sampleCirclePx[2]/1.8)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 8)

    return img
#------------------------------------------------------------------------------
#Start of Main Program
#------------------------------------------------------------------------------
def main():
    #Get sample info from user
    serialNumber, descriptor, referenceDimension = get_sample_info()
    
    #Get picture
    imgPath = get_picture_to_analyze()
    
    #Check to make sure that user selects points that are sufficiently far apart
    #to ensure measurement precision
    
    while True:
        
        #Have the user pick 3 points along the sample and reference diameter
        try:
            
            #Prepare image
            img = image_preparation(imgPath)

            #Start tracking mouse clicks (right mouse button only)
            cv2.setMouseCallback('image', mouse_callback)
            
            #Show image and instruct user how to proceed
            img = cv2.imshow('image', img)
            print('\nSelect three points along the circumference of the potting diameter.')
            cv2.waitKey(0)
   
            #Calculate the diameters of the reference circle and sample circle in pixels
            refCirclePx = calculate_pix_circle(rightClicks[3],rightClicks[4],rightClicks[5])
            sampleCirclePx = calculate_pix_circle(rightClicks[0],rightClicks[1],rightClicks[2])
            
            #Check to make sure the user selected adequate points for diameter calculation
            refRatio = triangle_area_to_circle_area_ratio(rightClicks[3],rightClicks[4],rightClicks[5], refCirclePx[2])
            sampleRatio = triangle_area_to_circle_area_ratio(rightClicks[0],rightClicks[1],rightClicks[2], sampleCirclePx[2])
        
        except ValueError:
            print('Value Error')
            continue
            
        #If user did not select adequate points, let them know and have them pick new points
        if refRatio < 0.1 or sampleRatio < 0.1:
            print('Selected points too close together. Pick points further away from each other.')
            del rightClicks[:]
            continue
        
        else:
            break
            
    
    #Convert from pixels to reference dimension units and calculate sample diameter
    pottingDiameter = convert_from_pixels_to_inches(refCirclePx[2], referenceDimension, sampleCirclePx[2])
    
    #Reload original image in color
    img = image_preparation(imgPath,1)
    
    #Export and display results of measurement
    img = analysis_results(serialNumber, descriptor, referenceDimension, pottingDiameter, refCirclePx, sampleCirclePx, img)
    img = cv2.imshow('image', img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
    
