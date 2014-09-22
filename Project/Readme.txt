####################################################################################
#										   #
#           File for any comments and remarks concerning the GPU project           #
#										   #
####################################################################################

		Technical University of Munich, Munich, Germany
			   Computer Vision Group
				Sept. 2014

Autors: 
		Mariusz Bujny 		(email: mariusz.bujny@gmail.com)
		Aleksandra Pachalieva	(email: apachalieva@gmail.com)

supervised by:  Thomas Moellenhoff      (email: moellenh@in.tum.de)


The following application is a combined project between Scientific Computing department 
and the Computer Vision depatment at Technical University Munich, Munich, Germany. 
The goal is to be developed an inpainting algorithm based on Navier-Stokes equations. 
The used references are given at the end of this file. 

1. Organization of the code:
The project contains of CFD and Computer Vision part. The CFD part is organised in an 
additional folder and the whole execution of the implemented algorithm for solving the 
Navier-Stockes equation is completely independent of the rest of the code. 
The computer vision code is written mainly using CUDA functions, which are defined in 
a file called "functionalities.cu" and "functionalities.h". In the main function only 
the basic algorithm is executed. 

2. Solution of the Navier-Stokes Problem:

3. Preparation of the images:
RGB -> Chrome Transformation ()  
Additional picture created to indicate the inpainitting region. 







####################################################################################
References:
1. M. Bertalmio, A. L. Bertozzi, G. Sapiro - "Navier-Stokes, Fluid Dynamics, and Image 
and Video Inpainting".
2. CFD Lab, Worksheet 3 - some parts of the code are erased, because they are not used.
3. 
