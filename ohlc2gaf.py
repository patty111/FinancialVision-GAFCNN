import matplotlib.pyplot as plt
import numpy as np
from utils.util_process import *

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
print(ohlc[0])

# Convert the OHLC data to GAF
open = ohlc[0, :, 0].reshape(1, -1, 1)
high = ohlc[0, :, 1].reshape(1, -1, 1)
low = ohlc[0, :, 2].reshape(1, -1, 1)
close = ohlc[0, :, 3].reshape(1, -1, 1)

OHLC = {
    'open': open,
    'high': high,
    'low': low,
    'close': close
}

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('OHLC GAF')

# 4 in one
for i, (name, data) in enumerate(OHLC.items()):
    gasf = get_gasf(data.reshape(1, -1, 1))
    gasf_mean = gasf[0].mean(axis=-1)

    # Display the GASF image in the subplot
    ax = axs[i // 2, i % 2]
    # im = ax.imshow(gasf_mean, cmap='rainbow', origin='lower')
    im = ax.imshow(gasf_mean, cmap='grey', origin='lower')
    ax.set_title(f'GAF {name}')

# Add a colorbar to the figure
fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.0457, pad=0.04)
fig.savefig('./results/ohlc/ohlc_gaf.png')
plt.show()


# Seperate
for i, (name, data) in enumerate(OHLC.items()):
    gasf = get_gasf(data.reshape(1, -1, 1))
    gasf_mean = gasf[0].mean(axis=-1)

    # Create a new figure for each image
    fig, ax = plt.subplots()

    # Display the GASF image in the subplot
    im = ax.imshow(gasf_mean, cmap='grey', origin='lower')
    # ax.set_title(f'GAF {name}')

    # Add a colorbar to the figure
    # fig.colorbar(im, fraction=0.0457, pad=0.04)

    ax.axis('off')
    
    # Save the current figure (including the subplot) in a separate file
    plt.savefig(f'results/ohlc/gaf_{name}.png', bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)