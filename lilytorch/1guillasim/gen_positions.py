'''
for (int i=0; i<NumMotors; i++) {
    double x = (float)(i + 1)/8; // 8 because of the 8 motors
    theta[i] = amplitDEGtoRAD*(1 + 0.323*(x - 1) + 0.31*(pow(x, 2) - 1)) * sin((2*M_PI*i)/lambda*14 - 2*M_PI*freq*time) // anguilliform , 14 because of body length = 14 times the joint length
}
'''

'''
ms mpt   A(deg) lambda(BL) f(Hz)   U(m/s)  Atail(m) tailRigidity
62   2 0.571429         15     1 0.215971 0.0575421            5
63   2 0.571429         20     1 0.273857 0.0735966            5
64   2 0.571429         25     1 0.330697 0.0909043            5
65   2 0.571429         30     1 0.371438  0.105378            5
69   2 0.714286         25     1  0.44496   0.12845            5
74   2 0.857143         25     1 0.408155  0.133501            5
79   2        1         25     1 0.418715  0.139161            5
'''


import numpy as np
import sys
import os
import matplotlib.pyplot as plt

tstop         = 10 # im seconds
sampling_rate = 1000 # 1ms
wlength       = 1 # wavelength in body lengths (BL)
amp           = 10.0*(np.pi/180.0) # amplitude in radians (?)
freq          = 1 # frequency in Hz
nmotors       = 8  # number of motors

times = np.expand_dims(np.arange(0,tstop,1/sampling_rate),axis=1)
times_expanded=np.repeat(
    times,
    nmotors,
    axis=1
)

idxs=np.arange(0,8)  # motor indices
x=(idxs+1)/nmotors  # convert motor index to x value

factor=(1+0.323*(x-1)+0.31*(x**2-1))

thetas = amp*factor*np.sin(
    2*np.pi*(
        wlength*idxs/14.0-freq*times_expanded  # 14 because of body length = 14 times the joint length
    )
)

data=np.column_stack([times,thetas])

x=data[:,0]
y=data[:,1:]
colors =  plt.cm.jet( np.linspace( 0, 1, y.shape[1]) )
for i in range(y.shape[1]):
    plt.plot(x,y[:,i],color=colors[i],label=f'Motor {i+1}')
plt.legend()
plt.show()


path = os.path.join(os.path.dirname(__file__), "positions.csv") if len(sys.argv) == 1 else sys.argv[1]
np.savetxt(path, data,delimiter=',')

