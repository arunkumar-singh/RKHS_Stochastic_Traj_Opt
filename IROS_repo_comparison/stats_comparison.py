import numpy as np
import jax.numpy as jnp
from jax import lax
import os, os.path

num_reduced = 10
num_circles = 3
num_batch = 100
num_validation = 1000

res_plot_matrix_val = np.zeros(2)
res_plot_matrix = np.zeros(2)
success = 0
total = 0
success_val = 0
temp_coll_free = 0
temp_coll_free_baseline = 0
temp_coll_free_baseline_val = 0
temp_coll_free_val = 0
coll_free_mmd = []
coll_free_baseline = []

folder = "data_comparison/"

for filename in os.scandir(folder):
    if("npz" in filename.name and filename.name!="data_collision_per_scene.npz"):
        print("filename ", filename.name)
        data_full = np.load(filename,allow_pickle=True)
        data = data_full["data"]
        data_obs = data_full["data_obs"]
    else:
        continue

    num_p = 25000
#################### scene 0 timestep 11
    x_path = np.linspace(0,1000,num_p)
    y_path = np.zeros(num_p)

    idx = 0
    x_global_init = x_path[idx]
    y_global_init = y_path[idx]
    psi_global_init = jnp.arctan2((y_path[idx+1]-y_path[idx]), (x_path[idx+1]-x_path[idx]))

    x_baseline,y_baseline,x_obs_up,y_obs_up,x_obs_r_up,y_obs_r_up,x_obs_mean_up,y_obs_mean_up,x_obs_up_val,y_obs_up_val = data_obs

    mmd,baseline,initial_state_frenet,obs_data_frenet,obs_traj_frenet = data[0]
    global_x_mmd, global_y_mmd,psi_global_mmd,x_best_mmd,y_best_mmd,res_mmd,res_2_mmd,res_3_mmd,res_4_mmd,res_5_mmd = mmd
    
    global_x_baseline, global_y_baseline,psi_global_baseline,x_best_baseline,y_best_baseline,res_baseline,res_2_baseline,res_3_baseline,res_4_baseline,\
                                res_5_baseline = baseline

    collision = []
    collision_val = []
    collision_baseline = []
    collision_val_baseline = []

    j=-1
    count=0
    count_val = 0
    count_baseline=0
    count_val_baseline = 0

    res_4_mmd = res_4_mmd[-1,:,:]
    res_4_baseline = res_4_baseline[-1,:,:]
    res_5_mmd = res_5_mmd[-1,:,:]
    res_5_baseline = res_5_baseline[-1,:,:]
    
    k_train = np.arange(0,num_circles*num_batch,num_circles)
    k_val = np.arange(0,num_circles*num_validation,num_circles)

    intersection_init = jnp.zeros(k_train.shape[0])
    intersection_baseline_init = jnp.zeros(k_train.shape[0])
    intersection_val_init = jnp.zeros(k_val.shape[0])
    intersection_baseline_val_init = jnp.zeros(k_val.shape[0])
  
    def lax_intersection(carry,idx):
        intersection,intersection_baseline = carry
        intersection = intersection.at[(idx/3).astype(int)].set(jnp.count_nonzero(res_4_mmd[idx,:]) + jnp.count_nonzero(res_4_mmd[idx+1,:]) + \
                                             jnp.count_nonzero(res_4_mmd[idx+2,:]))
        intersection_baseline = intersection_baseline.at[(idx/3).astype(int)].set(jnp.count_nonzero(res_4_baseline[idx,:]) + \
                                                              jnp.count_nonzero(res_4_baseline[idx+1,:]) +\
                                                                  jnp.count_nonzero(res_4_baseline[idx+2,:]))
        return (intersection,intersection_baseline),0
    
    carry_init = (intersection_init,intersection_baseline_init)
    carry_final,result = lax.scan(lax_intersection,carry_init,k_train)

    intersection,intersection_baseline = carry_final

    def lax_intersection_val(carry,idx):
        intersection,intersection_baseline = carry
        intersection = intersection.at[(idx/3).astype(int)].set(jnp.count_nonzero(res_5_mmd[idx,:]) + jnp.count_nonzero(res_5_mmd[idx+1,:]) + \
                                             jnp.count_nonzero(res_5_mmd[idx+2,:]))
        intersection_baseline = intersection_baseline.at[(idx/3).astype(int)].set(jnp.count_nonzero(res_5_baseline[idx,:]) + \
                                                              jnp.count_nonzero(res_5_baseline[idx+1,:]) + \
                                                                jnp.count_nonzero(res_5_baseline[idx+2,:]))
        return (intersection,intersection_baseline),0
    
    carry_init = (intersection_val_init,intersection_baseline_val_init)
    carry_final,result = lax.scan(lax_intersection_val,carry_init,k_val)

    intersection_val,intersection_baseline_val = carry_final

    count = np.count_nonzero(intersection)
    count_baseline = np.count_nonzero(intersection_baseline)
    count_val = np.count_nonzero(intersection_val)
    count_val_baseline = np.count_nonzero(intersection_baseline_val)

    res_plot_matrix[0] = num_batch-count
    res_plot_matrix_val[0] = num_validation-count_val
    res_plot_matrix[1] = num_batch-count_baseline
    res_plot_matrix_val[1] = num_validation-count_val_baseline

    diff = res_plot_matrix[0] - res_plot_matrix[1]              
    diff_val = res_plot_matrix_val[0] - res_plot_matrix_val[1]            
    
    if(diff>=0):
        success = success + 1

    if(diff_val>=0):
        success_val = success_val + 1

    total = total + 1
    
    temp_coll_free = temp_coll_free + res_plot_matrix[0]
    temp_coll_free_baseline = temp_coll_free_baseline + res_plot_matrix[1]
    temp_coll_free_val = temp_coll_free_val + res_plot_matrix_val[0]
    temp_coll_free_baseline_val = temp_coll_free_baseline_val + res_plot_matrix_val[1]

    coll_free_mmd = np.append(coll_free_mmd,res_plot_matrix_val[0])
    coll_free_baseline = np.append(coll_free_baseline,res_plot_matrix_val[1])
    
    np.savez(folder + "data_collision_per_scene",coll_free_mmd = coll_free_mmd,coll_free_baseline = coll_free_baseline)

mean_mmd = np.mean(np.asarray(coll_free_mmd))
mean_baseline = np.mean(np.asarray(coll_free_baseline))
std_mmd = np.std(np.asarray(coll_free_mmd))
std_baseline = np.std(np.asarray(coll_free_baseline))
error=[std_mmd,std_baseline]
mean= [mean_mmd,mean_baseline]

coll_free = temp_coll_free/total
coll_free_baseline = temp_coll_free_baseline/total
coll_free_val = temp_coll_free_val/total
coll_free_val_baseline = temp_coll_free_baseline_val/total

print(mean,error)

with open(folder + '/stats_overall.txt', 'w') as f:
    f.write('avg. collision-free trajs training ours/baseline {}/{} ;'.format(coll_free,coll_free_baseline) + "\n" + \
        'avg. collision-free trajs validation ours/baseline {}/{} ;'.format(coll_free_val,coll_free_val_baseline) + "\n" + \
            "success_training {}% ;".format((success/total)*100) + "\n" + "success_validation {}%".format((success_val/total)*100) + "\n"  + \
              "mean(ours/baseline) {} ".format(mean) + " std(ours/baseline) {}".format(error))