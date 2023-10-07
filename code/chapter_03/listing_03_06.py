import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(1,2,1)
ax1.plot(my_data['12C/13C'], my_data['14N/15N'], 
            marker='o', markeredgecolor='k', 
            markerfacecolor='#BFD7EA', linestyle='', 
            color='#7d7d7d',  
            markersize=6)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel(r'$^{12}C/^{13}C$')  
ax1.set_ylabel(r'$^{14}N/^{15}N$')

ax2 = fig.add_subplot(1,2,2)
ax2.hist(my_data['12C/13C'], density=True, bins='auto', 
         histtype='stepfilled',  color='#BFD7EA', edgecolor='black',)
ax2.set_xlim(-1,250)  
ax2.set_xlabel(r'$^{12}C/^{13}C$')  
ax2.set_ylabel('Probability Density')

fig.set_tight_layout(True)
