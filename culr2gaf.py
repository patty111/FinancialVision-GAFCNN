import matplotlib.pyplot as plt
import numpy as np
from utils.util_process import *
from PIL import Image
from matplotlib import cm

np.set_printoptions(suppress=True)

# ETH OHLC from 2023 12 11 ~ 2023 12 18
data = np.array([
    [2352.31, 2354.9, 2133.47, 2224.1],
    [2224.1, 2242.96, 2165.58, 2202.33],
    [2202.33, 2284.2, 2145.57, 2260.72],
    [2260.72, 2332.24, 2231.57, 2316.03],
    [2316.03, 2318.06, 2201.14, 2220.32],
    [2220.32, 2261.92, 2210.47, 2227.14],
    [2227.14, 2245.8, 2190.85, 2194.84],
    # [2194.84, 2222.91, 2115.85, 2203.75],
])

# Reshape the data to match the expected input shape for ohlc2culr
ohlc = data.reshape(1, *data.shape) # (1, 5, 4) -> (N, ts_n, 4) -> num of instances, length of time series, num of features 
# print(ohlc[0])

# Convert the OHLC data to CULR data
culr = ohlc2culr(ohlc)
# print(culr[0])

# Convert the CULR data to GAF
close = culr[0, :, 0].reshape(1, -1, 1)
upper = culr[0, :, 1].reshape(1, -1, 1)
lower = culr[0, :, 2].reshape(1, -1, 1)
realbody = culr[0, :, 3].reshape(1, -1, 1)

CULR = {
    'close': close,
    'upper': upper,
    'lower': lower,
    'realbody': realbody
}

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('CULR GAF')

# 4 in one
for i, (name, data) in enumerate(CULR.items()):
    gasf = get_gasf(data.reshape(1, -1, 1))
    gasf_mean = gasf[0].mean(axis=-1)


    # print(np.array2string(gasf_mean, precision=4, separator=' ', suppress_small=True, max_line_width=1000))

    # Display the GASF image in the subplot
    ax = axs[i // 2, i % 2]
    # im = ax.imshow(gasf_mean, cmap='rainbow', origin='lower')
    im = ax.imshow(gasf_mean, cmap='grey', origin='lower')
    ax.set_title(f'GAF {name}')

# Add a colorbar to the figure
fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.0457, pad=0.04)
fig.savefig('./results/culr/culr_gaf.png')
# plt.show()


# Seperate
for i, (name, data) in enumerate(CULR.items()):
    gasf = get_gasf(data.reshape(1, -1, 1))
    gasf_mean = gasf[0].mean(axis=-1)

    # print(gasf_mean)

    my_array = np.array(gasf_mean)

    dpi = 100.0
    w, h = my_array.shape[1]/dpi, my_array.shape[0]/dpi
    fig = plt.figure(figsize=(w,h), dpi=dpi)
    fig.figimage(my_array, cmap='gray')
    plt.savefig(f'results/culr/gaf_{name}.png')

