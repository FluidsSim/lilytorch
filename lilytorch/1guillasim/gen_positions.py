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

def generate_positions(
    tstop=10,
    sampling_rate=1000,
    wlength=1,
    amp_deg=20.0,
    freq=0.5,
    nmotors=8,
    TWL=14,
    save_path=None,
    plot=True
):
    amp = amp_deg * (np.pi / 180.0)
    times = np.expand_dims(np.arange(0, tstop, 1 / sampling_rate), axis=1)
    times_expanded = np.repeat(times, nmotors, axis=1)

    idxs = np.arange(nmotors)
    x = (idxs + 1) / nmotors
    factor = (1 + 0.323 * (x - 1) + 0.31 * (x ** 2 - 1))
    # factor[:-1] *= 0
    # factor[-1] = 4

    thetas = amp * factor * np.sin(
        2 * np.pi * (
            wlength * idxs / TWL - freq * times_expanded
        )
    )

    data = np.column_stack([times, thetas])

    if plot:
        x_plot = data[:, 0]
        y_plot = data[:, 1:]
        colors = plt.cm.jet(np.linspace(0, 1, y_plot.shape[1]))
        for i in range(y_plot.shape[1]):
            plt.plot(x_plot, y_plot[:, i], color=colors[i], label=f'Motor {i+1}')
        plt.legend()
        plt.show()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "positions.csv")
    np.savetxt(save_path, data, delimiter=',')

    return data

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_positions(save_path=path)
