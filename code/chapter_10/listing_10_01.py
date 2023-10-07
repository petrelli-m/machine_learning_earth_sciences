import matplotlib.pyplot as plt

my_pressure_mean = np.mean(my_pressure_dist)
my_pressure_std = np.std(my_pressure_dist)

fig, ax = plt.subplots()
ax.hist(my_pressure_dist, density=True, bins='auto', 
        color='#0F7F8B', label='Pressure estimates')
ax.axvline(my_pressure_mean, color='#C82127', label='mean value')
ax.axvspan(my_pressure_mean - my_pressure_std,
           my_pressure_mean + my_pressure_std, 
           color='#F15C61', alpha=0.4,
           label=r'1$\sigma$ estimate')
ax.set_xlabel('Pressure [MPa]')
ax.set_ylabel('Probability Density')
ax.legend()
plt.show()
