import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
#import plotly.plotly as py
import time
import numpy as np
from scipy import fftpack
import numpy.fft as fft
from numpy import *


Znet=0
#cap = cv2.VideoCapture("outcpp2.avi")
cap=cv2.VideoCapture(1)
ret, old_frame = cap.read()
## get first frame to compair
height = np.size(old_frame, 0)
width = np.size(old_frame, 1)

while(1):

	#print(time.process_time())
	ret, curr = cap.read()
	
	
	
	#print(motion)
	cv2.imshow('frame',curr)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

		
cv2.destroyAllWindows()	
cap.release()		
