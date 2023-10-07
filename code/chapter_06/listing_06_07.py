from sklearn.preprocessing import  RobustScaler
from sklearn import  mixture
import matplotlib.colors as mc
import matplotlib.pyplot as plt

X = RobustScaler().fit_transform(my_array_1d)
my_ml_model = mixture.GaussianMixture(n_components=5, covariance_type="full")
labels_1d = my_ml_model.predict(X)

labels_2d = labels_1d.reshape(my_array_2d[:,:,0].shape) 

cmap = mc.LinearSegmentedColormap.from_list("", ["black","red","yellow", "green","blue"])
fig, ax = plt.subplots(figsize=[18,18])
ax.imshow(labels_2d, cmap=cmap)
ax.axis('off')
