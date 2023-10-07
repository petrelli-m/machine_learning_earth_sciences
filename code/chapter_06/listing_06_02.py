import numpy as np
from skimage import exposure, io
from skimage.transform import resize
import matplotlib.pyplot as plt

r_g_b = np.dstack([bands_dict['B4'], 
                   bands_dict['B3'], 
                   bands_dict['B2']])

# contrast stretching and rescaling between [0,1]
p2, p98 = np.percentile(r_g_b, (2,98))
r_g_b = exposure.rescale_intensity(r_g_b, in_range=(p2, p98))
r_g_b = r_g_b / r_g_b.max()

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(r_g_b)
ax.axis('off')

