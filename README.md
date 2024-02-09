## Laser calibration

Program to calibrate planar light stripes using a camera calibration checkerboard and comparing meshes.
This code is based in the code of the paper "**Complete calibration of a structural light stripe vision sensor through planar target of unknown orientations**" and the code of the same paper [Light-Strip Calibration](https://github.com/hjamal3/Light-Stripe-Calibration)

### Codes in this repository
* calibrate.py, this code calibrate the laser-stripe with the images in the folders. It uses the class Laser Calib which is located in laser_calibration.py.
* helper.py, this code contains a lot of usefull functions for the others codes
* mapping.py, this code used the calibrate reuslts and compute the coordinates of the target in a 3D plot. 
* comparing_meshes.py, code to compare point clouds
* laser_calibration.py, code with the class of the laser calibration

### TODO before run the code calibrate.py
* Use the correct path to the camera calibration parameters.
* Insert all of your images in the 'images' folder.  
* Insert all the masks in the mask folder.
* Change internal parameters of file to fit checkerboard size and dimensions in calibrate.py.

To run: python calibrate.py.  or comparing_meshes.py

* Images should contain the laser over a fully visible checkerboard for the calibrate.py.  

![Image](images/1661359028.png)

* Masks should contain the masks of the same images in the folder images (with the same name).



### TODO before run the code mapping.py
* Be carefull of the names of the calibration parameters of the laser-stripe calibration and camera calibration parameters.
* Change the path to the masks that you want to use to make the reconstruction.

### Libraries that you will need
* Numpy
* Open cv
* Glob
* Sys
* Sympy
* Matplotlib

### Flow of the code calibrate.py

1. Read the RGB images of the folder "images".
2. Begin a loop from image to image.
3. Read the matching mask of the image in the folder "mask".
4. We will obtain the white pixeles of the mask and we will fit them in a line. Then we will transform the mask into a one pixel per column mask. 
5. Create the mask with one white pixel per column with the function "image_creating".
6. Find the cheesboard in the RGB image and compute the 3D coordenates of the corners. If the corners are not found, we continue woth another image. 
7. Calculate the corners index of the corners that we want, in this case a vertical line (3, 21, 34). Then we can obtain the coordinates of this corners in pixels.
8. Obtain the 3D coodenates of the vertical corners with the camera parameters of the image.
9. Found the pixel where the vertical points cross the laser line (q point) in pixels.
10. Calculate the cross radio of the segments of the laser. 
11. Obtain the x, y coordenates of the q point. 
12. Compute the x, y coordenate of the q point into the camera frame.
13. Continue with the loop and save the q point in the camera frame.
14. When the loop is over we can obtain the 3D plane of the laser based in the 1 points that we have, we need at least 3. 
15. Use the Optimization function of minimum squares and obtain the plane parameters. 



### Flow of the code mapping.py

We are using the calibration parameters of the housing, after the calibration we are using the parameters that we found to make the reconstruction.

1. Read the masks images of the path.
2. Begin a loop from mask to mask.
3. Undistord the masks with the camera calibration parameters. 
4. With the undistord masks, create a mask with one white pixel per column. 
5. Compute the values of the pixels in meters. 
6. Compute the Z value with the plane equation in meters in the camera frame. 
7. Save the vaules for all the images. 
8. Plot the 3D reconstruction. 

