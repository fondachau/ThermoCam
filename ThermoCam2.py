import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
#import plotly.plotly as py
import time
from scipy import fftpack
import numpy.fft as fft
from numpy import *


ircap=cv2.VideoCapture(2)
vidcap=cv2.VideoCapture(0)
vidcap.set(3,960);
vidcap.set(4,720);
ret, irold_frame = ircap.read()
ret, vidold_frame = vidcap.read()
vidold_frame_crop = vidold_frame[:, 160:1120]
vidheight = np.size(vidold_frame_crop, 0)
vidwidth = np.size(vidold_frame_crop, 1)

print(vidheight)
print(vidwidth)

alpha=0.5
beta = ( 1.0 - alpha )


while(1):
	ret, ircurr = ircap.read()
	ret,vidcurr=vidcap.read()
	
	vidcurr_crop = vidcurr[:, 160:1120]
	irlargeframe=cv2.resize(ircurr,(int(vidwidth),int(vidheight)))
	
	hor2image = np.hstack((irlargeframe,vidcurr_crop))
	dst = cv2.addWeighted( irlargeframe, alpha, vidcurr_crop, beta, 0.0);
	
	cv2.imshow('frame',vidcurr)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

		
cv2.destroyAllWindows()	
vidcap.release()		

