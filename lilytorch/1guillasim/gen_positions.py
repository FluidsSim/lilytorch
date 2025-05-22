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

import os
import math

def gen_positions(path):
    nmotors = 8
    wlength = 15
    freq = 1
    if os.path.exists(path): os.remove(path)
    with open(path, "w") as f:
        for t in range(1000):
            data = [str(t/1000)]
            for i in range(nmotors):
                x = (i + 1) / 8
                theta = math.pi/180 * (1 + 0.323*(x - 1)) + 0.31*(x**2 - 1) * math.sin(2*math.pi*(14*i/wlength - freq*t/1000))
                data.append(str(theta))
            f.write(",".join(data) + "\n")

if __name__ == "__main__":
    import sys
    path = os.path.join(os.path.dirname(__file__), "positions.csv") if len(sys.argv) == 1 else sys.argv[1]
    gen_positions(path)