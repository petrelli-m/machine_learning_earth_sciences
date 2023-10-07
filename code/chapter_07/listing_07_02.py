import numpy as np
import matplotlib.pyplot as plt
line_colors = ['#F15C61','#0F7F8B','#0A3A54','#C82127']

# linear data set with noise
n = 100
theta_1, theta_2 = 3, 1 # target value for theta_1 & theta_2
x = np.linspace(-10, 10, n) 
np.random.seed(40)
noise = np.random.normal(loc=0.0, scale=1.0, size=n)
y = theta_1 + theta_2 * x + noise
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
ax1.scatter(x, y, c='#BFD7EA',  edgecolor='k')

my_theta_1, my_theta_2  = 0, 0 # arbitrary initial values
gamma = 0.0005 # learning rate
t_final = 10001 # umber of itrations  
n = len(x)
to_plot,  cost_function = [1, 25, 500, 10000], [] 
# Gradient Descent 
for i in range(t_final):   
    #Eq. 4.30
    D_theta_1 = (-2/n)*np.sum(y-(my_theta_1 + my_theta_2*x))
    #Eq. 4.31
    D_theta_2 = (-2/n)*np.sum(x*(y-(my_theta_1+my_theta_2*x)))  
    
    my_theta_1 = my_theta_1 - gamma * D_theta_1 #Eq. 4.32 
    my_theta_2 = my_theta_2 - gamma * D_theta_2 #Eq. 4.33 
    cost_function.append((1/n) * np.sum(y - (my_theta_1 + my_theta_2 * x))**2)
    
    if i in to_plot:
        color_index = to_plot.index(i)
        my_y = my_theta_1 + my_theta_2 * x  
        ax1.plot(x,my_y, color=line_colors[color_index],
                label='iter:  {:.0f}'.format(i) + ' - ' +
                r'$\theta_1 = $' + '{:.2f}'.format(my_theta_1) +
                ' - ' +
                r'$\theta_2 = $' + '{:.2f}'.format(my_theta_2))
ax1.set_xlabel('x')   
ax1.set_ylabel('y')    
ax1.legend()
cost_function = np.array(cost_function)
iterations = range(t_final)
ax2.plot(iterations,cost_function, color='#C82127',
         label='mean squared-error cost function Eq.4.29') 
ax2.set_xlabel('Iteration')   
ax2.set_ylabel('Cost Function Value')    
ax2.legend()
fig.tight_layout()

