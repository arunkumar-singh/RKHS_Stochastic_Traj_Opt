import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import matplotlib.collections
import scipy
import seaborn as sns
from celluloid import Camera

sns.set_theme(style = "whitegrid", palette = 'tab10')
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')

def normal_vectors(x, y, scalar):
    tck = scipy.interpolate.splrep(x, y)
    y_deriv = scipy.interpolate.splev(x, tck, der=1)
    normals_rad = np.arctan(y_deriv)+np.pi/2.
    return np.cos(normals_rad)*scalar, np.sin(normals_rad)*scalar

num_reduced = 10 # number of samples in reduced set
num_batch = 100 # total number of samples in training set
num_val = 1000 # total number of samples in validation set
num = 100 # number of time steps in mpc horizon

### Info about sample data ####

# "45_-3_5.npz" - obstacle initial position (45,-3) with 5% and 95% probability of ydes_1 = 0.0 and ydes_2 = -3.5, respectively.
# "45_-3_10.npz" - obstacle initial position (45,-3) with 10% and 90% probability of ydes_1 = 0.0 and ydes_2 = -3.5, respectively.
# "45_-3_20.npz" - obstacle initial position (45,-3) with 20% and 80% probability of ydes_1 = 0.0 and ydes_2 = -3.5, respectively.
# "demo_inlane_slow_down.npz" - obstacle initial position (45,-3.5) with 5% and 95% probability of ydes_1 = 0.0 and ydes_2 = -3.5, respectively.

folder = "sample_data/"
filename = "demo_inlane_slow_down.npz" 

data_full = np.load(folder+filename,allow_pickle=True)
data = data_full["data"]
data_obs = data_full["data_obs"]

num_p = 25000
x_path = np.linspace(0,1000,num_p)
y_path = np.zeros(num_p)

idx = 0
x_global_init = x_path[idx]
y_global_init = y_path[idx]
psi_global_init = np.arctan2((y_path[idx+1]-y_path[idx]), (x_path[idx+1]-x_path[idx]))
v_global_init = 7.0
tot_time = np.linspace(0, 15, num)

x_obs_r_single_baseline,y_obs_r_single_baseline,x_obs_up,y_obs_up,x_obs_r_up,y_obs_r_up,x_obs_mean_up,y_obs_mean_up,x_obs_up_val,y_obs_up_val = data_obs

mmd,baseline,initial_state_frenet,obs_data_frenet,obs_traj_frenet = data[0]
global_x_mmd, global_y_mmd,psi_global_mmd,x_best_mmd,y_best_mmd,res_mmd,res_2_mmd,res_3_mmd,res_4_mmd,res_5_mmd = mmd

vx_mmd = np.diff(global_x_mmd)/0.15
vx_mmd = np.hstack((vx_mmd,vx_mmd[-1]))
vy_mmd = np.diff(global_y_mmd)/0.15
vy_mmd = np.hstack((vy_mmd,vy_mmd[-1]))

global_x_baseline, global_y_baseline,psi_global_baseline,x_best_baseline,y_best_baseline,res_baseline,res_2_baseline,res_3_baseline,res_4_baseline,\
                            res_5_baseline = baseline

x_obs_r_single_baseline,y_obs_r_single_baseline,x_obs_up,y_obs_up,x_obs_r_up,y_obs_r_up,x_obs_mean_up,y_obs_mean_up,x_obs_up_val,y_obs_up_val = data_obs

x_obs_r_up = x_obs_r_up[0:num_reduced]
y_obs_r_up = y_obs_r_up[0:num_reduced]

x_path_normal_lb,y_path_normal_lb = normal_vectors(x_path,y_path,5.25)
x_path_lb = x_path + x_path_normal_lb
y_path_lb = y_path + y_path_normal_lb

x_path_normal_ub,y_path_normal_ub = normal_vectors(x_path,y_path,-5.25)
x_path_ub = x_path + x_path_normal_ub
y_path_ub = y_path + y_path_normal_ub

x_path_normal_d_lb,y_path_normal_d_lb = normal_vectors(x_path,y_path,-1.75)
x_path_d_lb = x_path + x_path_normal_d_lb
y_path_d_lb = y_path + y_path_normal_d_lb

x_path_normal_d_ub,y_path_normal_d_ub = normal_vectors(x_path,y_path,1.75)
x_path_d_ub = x_path + x_path_normal_d_ub
y_path_d_ub = y_path + y_path_normal_d_ub

patch_ego_global = pt.Ellipse((0.0,0.0),width=6.0,height=3.0,angle=np.rad2deg(psi_global_init),facecolor='blue',edgecolor='black')
patch_obs_global = pt.Ellipse((x_obs_up_val[0,0],y_obs_up_val[0,0]),width=6.0,height=3.0,angle=0.0,facecolor='g',edgecolor='black')

psi_obs_up = np.arctan2(np.diff(y_obs_up_val,axis=1),np.diff(x_obs_up_val,axis=1))
psi_obs_up = np.hstack((psi_obs_up,psi_obs_up[:,-1].reshape(num_val,1)))

len_path = 3000

fig, ax = plt.subplots(figsize=(12,6))
camera = Camera(fig)

ax.axis("equal")
ax.set_xlim([-10,125])
ax.set_ylim([-30,30])
fig.tight_layout(pad=1.0)

## To reproduce the results shown in the submission video please use the following indx values.

# "45_-3_5.npz" - indx 4
# "45_-3_10.npz" - indx 6
# "45_-3_20.npz" - indx 108
# "demo_inlane_slow_down.npz" - indx 124

indx = 124

linewidth = 0.5
alpha = 1
text_x = 80

patch_obs_global = pt.Ellipse((x_obs_up_val[indx,0],y_obs_up_val[indx,0]),width=6.0,height=3.0,angle=np.rad2deg(psi_global_init),facecolor='g',edgecolor='black')

for jj in range(0,num,3):
    if(jj<=10):
        ax.text(0,17.5,"Samples drawn from mixture of Discrete Binomial and Gaussian distributions " "\n"
                "\n"
                "Possible obstacle trajectories in black \n \n Our optimal reduced set in red")
    x_obs = x_obs_up_val[indx,jj]
    y_obs = y_obs_up_val[indx,jj]
    angles = np.rad2deg(psi_obs_up[indx,jj])

    patch_obs_global_current = pt.Ellipse((x_obs,y_obs),width=6.0,height=3.0,angle=np.rad2deg(psi_obs_up[indx,jj]),\
                                            facecolor='g',edgecolor='black',alpha=alpha)

    patch_ego_global_current = pt.Ellipse((global_x_mmd[jj],global_y_mmd[jj]),width=6.0,height=3.0,angle=np.rad2deg(psi_global_mmd[jj]),\
                                            facecolor='blue',edgecolor='black',alpha=alpha)

    ax.add_patch(patch_obs_global_current)
    ax.add_patch(patch_ego_global_current)

    ax.plot(global_x_mmd[jj:],global_y_mmd[jj:] ,color='b',linewidth=4.0,label="Ego trajectory")
    ax.plot(x_obs_up_val[indx,:],y_obs_up_val[indx,:],color='b',linewidth=4.0,label="obstacle trajectory")
    ax.plot(x_obs_up_val[0,0],y_obs_up_val[0,0],color="k",marker="X",markersize=10,label="Obstacle start point")
    ax.plot(x_path_lb[0:len_path],y_path_lb[0:len_path],color='tab:brown',linewidth=3*linewidth,linestyle="--")
    ax.plot(x_path_ub[0:len_path],y_path_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
    ax.plot(x_path_d_lb[0:len_path],y_path_d_lb[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
    ax.plot(x_path_d_ub[0:len_path],y_path_d_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")

    for ii in range(0,len(x_obs_r_up)):
        ax.plot(x_obs_r_up[ii,:],y_obs_r_up[ii,:],color="r",linewidth=6*linewidth,alpha=1)
    for ii in range(0,len(x_obs_up_val)):
        ax.plot(x_obs_up_val[ii,:],y_obs_up_val[ii,:],color="k",linewidth=0.1*linewidth,alpha=0.8)   

    num_reduced_high = y_obs_r_up[y_obs_r_up[:,-1]<=-3.4 ].shape[0]
    num_reduced_low = y_obs_r_up[y_obs_r_up[:,-1]>=-0.1].shape[0]
    
    ax.text(0,-20,"Our reduced set {}% in high prob. region and \n {}% in low prob. region".format((num_reduced_high/10)*100,(num_reduced_low/10)*100))
    ax.text(text_x,20,"Ego velocity: {}".format(np.sqrt(vy_mmd[jj]**2+vx_mmd[jj]**2)))
    ax.annotate('High probability obstacle trajectories', xy=(60,-3.0), xytext=(45,-15),
        arrowprops=dict(facecolor='black', shrink=0.05),fontsize=10)
    
    if(jj>=11):
        ax.text(0,20,"Note how our optimal solution does not result in slowing-down behaviour \n since lane change of obstacle is less probable")

    camera.snap() 

animation = camera.animate()
animation.save('videos/animation_ours.mp4')
    