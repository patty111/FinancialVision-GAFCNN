import matplotlib.pyplot as plt
import numpy as np
from utils.util_process import *

np.set_printoptions(suppress=True)

# ETH OHLC from 2023 12 11 ~ 2023 12 15
data = np.array([
    [2315.2, 2317.35, 2235.7, 2240.6],
    [2261.96, 2331.61, 2237.04, 2315.31],
    [2204.09, 2283.90, 2148.74, 2260.15],
    [2224.46, 2243.31, 2168.17, 2203.47],
    [2351.73, 2354.99, 2164.26, 2225.31]
])

# Reshape the data to match the expected input shape for ohlc2culr
ohlc = data.reshape(1, *data.shape) # (1, 5, 4) -> (N, ts_n, 4) -> num of instances, length of time series, num of features 
print(ohlc[0])

# Convert the OHLC data to CULR data
culr = ohlc2culr(ohlc)
print(culr[0])

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

# 4 in one
for i, (name, data) in enumerate(CULR.items()):
    gasf = get_gasf(data.reshape(1, -1, 1))
    gasf_mean = gasf[0].mean(axis=-1)

    # Display the GASF image in the subplot
    ax = axs[i // 2, i % 2]
    # im = ax.imshow(gasf_mean, cmap='rainbow', origin='lower')
    im = ax.imshow(gasf_mean, cmap='rainbow', origin='lower')
    ax.set_title(f'GAF {name}')

# Add a colorbar to the figure
fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.0457, pad=0.04)
fig.savefig('./results/culr_gaf.png')
plt.show()


# Seperate
for i, (name, data) in enumerate(CULR.items()):
    gasf = get_gasf(data.reshape(1, -1, 1))
    gasf_mean = gasf[0].mean(axis=-1)

    # Create a new figure for each image
    fig, ax = plt.subplots()

    # Display the GASF image in the subplot
    im = ax.imshow(gasf_mean, cmap='grey', origin='lower')
    ax.set_title(f'GAF {name}')

    # Add a colorbar to the figure
    fig.colorbar(im, fraction=0.0457, pad=0.04)

    # Save the current figure (including the subplot) in a separate file
    plt.savefig(f'results/gasf_{name}.png', dpi=500)

    # Close the figure to free up memory
    plt.close(fig)