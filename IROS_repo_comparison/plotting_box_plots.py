import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import seaborn as sns
from pylab import setp

sns.set_theme(style = "whitegrid", palette = 'tab10')
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue',linewidth=3)
    setp(bp['medians'][0], color='orange',linewidth=4)

    setp(bp['boxes'][1], color='red',linewidth=3)
    setp(bp['medians'][1], color='orange',linewidth=4)

def setViolinColors(bp):
    setp(bp['bodies'][0], color='blue',alpha=1,linewidth=3)
    setp(bp['bodies'][1], color='red',alpha=1,linewidth=3)
    vp = bp["cbars"]
    vp.set_edgecolor("k")
    vp.set_linewidth(3)

folder = "data_comparison"

filename = "data_collision_per_scene.npz"
coll_free_mmd = []
coll_free_baseline = []

temp_file = np.load(folder + "/{}".format(filename))
coll_free_mmd = temp_file["coll_free_mmd"]
coll_free_baseline = temp_file["coll_free_baseline"]

data = [coll_free_mmd,coll_free_baseline]

fig, axs = plt.subplots(1,2, figsize=(12, 6))

labels_synthetic = ["Ours","[6]"]
x_synthetic = np.array([0.5,1.5]) # the label locations
widths = 0.8
width= 1.0
whiskerprops = dict(linestyle='-',linewidth=3.0, color='black')

bp = axs[0].boxplot(data,positions=[1,2],showfliers=False,widths=widths,whiskerprops=whiskerprops)
setBoxColors(bp)
axs[0].set_title("(a)", fontweight="bold",fontsize=15)
axs[0].set_xticks(x_synthetic + width / 2, labels_synthetic)

bp = axs[1].violinplot(data,positions=[1,2])
setViolinColors(bp)
axs[1].set_title("(b)", fontweight="bold",fontsize=15)
axs[1].set_xticks(x_synthetic + width / 2, labels_synthetic)
axs[1].set_ylabel('#Coll. free trajs', fontweight = "bold", fontsize = 15)

for ax in axs.flat:
    ax.label_outer()
    ax.tick_params(labelbottom=True)
    ax.tick_params(labelleft=True)

fig.tight_layout(pad=0.5)
plt.show()