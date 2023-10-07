import numpy as np
from skimage import exposure, io
from skimage.transform import resize
import matplotlib.pyplot as plt

nir_r_g = np.dstack([bands_dict['B8'], 
                   bands_dict['B4'], 
                   bands_dict['B3']])

# contrast stretching and rescaling between [0,1]
p2, p98 = np.percentile(nir_r_g, (2,98))
nir_r_g = exposure.rescale_intensity(nir_r_g, in_range=(p2, p98))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(nir_r_g)
ax.axis('off')


