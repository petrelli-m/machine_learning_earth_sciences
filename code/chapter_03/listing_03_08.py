import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

X = my_data[['d(30Si/28Si)','d(29Si/28Si)']].to_numpy()

scalers = [("Unscaled", X),
        ("Standard Scaler", StandardScaler().fit_transform(X)),
        ("Min. Max. Scaler", MinMaxScaler().fit_transform(X)),
        ("Robust Scaler", RobustScaler().fit_transform(X))]

fig = plt.figure(figsize=(10,7)) 

for ix, my_scaler in enumerate(scalers):
    ax = fig.add_subplot(2,2,ix+1)
    scaled_X = my_scaler[1]
    ax.set_title(my_scaler[0])
    ax.scatter(scaled_X[:,0], scaled_X[:,1], 
               marker='o', edgecolor='k', color='#db0f00', 
               alpha=0.6, s=40)
    ax.set_xlabel(r'${\delta}^{30}Si_{28} [\perthousand]$')  
    ax.set_ylabel(r'${\delta}^{29}Si_{28} [\perthousand]$')

fig.set_tight_layout(True)