# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
from venusdata import heights78

# %%    
def profiles(xps, hlabels):

    heights = np.arange(0,35)
    for xp in xps:
        y_vals = 130*np.exp(-xp*(heights-21))
        y_vals[0:22] = 130
        plt.plot(y_vals, heights, label=f"Exp={xp:.4f}, top={y_vals[-1]:.4f} ppm")
    plt.xlabel("ppm")
    plt.ylabel("Height (km)")
    plt.yticks(heights[::5], np.round(hlabels[:35],0)[::5])
    plt.legend()
    plt.title("SO2 profiles")
    plt.show()

    return


# %%
y_max_vals = np.array([0.01, 0.2575, 0.505, 0.7525, 1.0])
exponents = -(1/13)*np.log(y_max_vals/130)
profiles(exponents, heights78)
# %%
