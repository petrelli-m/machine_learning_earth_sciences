th = 16.5
fig, ax = plt.subplots(figsize = (10,6))
ax.set_title("Hierarchical clustering dendrogram")
set_link_color_palette(['#000000','#C82127', '#0A3A54',
            '#0F7F8B', '#BFD7EA', '#F15C61', '#E8BFE7'])

plot_dendrogram(model, truncate_mode='level', p=5, 
                color_threshold=th, 
                above_threshold_color='grey')

plt.axhline(y = th, color = "k", linestyle = "--", lw=1)
ax.set_xlabel("Number of points in node")

fig, ax = plt.subplots(figsize = (10,6))
ax.set_title("Hierarchical clustering dendrogram")
ax.set_ylabel('Height')

plot_dendrogram(model, truncate_mode='lastp', p=6, 
                color_threshold=0, 
                above_threshold_color='k')

ax.set_xlabel("Number of points in node")



