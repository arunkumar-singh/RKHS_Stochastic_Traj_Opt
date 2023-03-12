import numpy as np
import jax.numpy as jnp
import jax
import kernel_computation
import optimizer
from jax import random
import compute_beta
import os, os.path

def main():

    prob = optimizer.mpc_path_following()
    prob2 = kernel_computation.kernel_matrix()
    prob3 = compute_beta.beta_cem()

    key = random.PRNGKey(0)

    num_p = 25000
#################### scene 0 timestep 11
    x_path = np.linspace(0,1000,num_p)
    y_path = np.zeros(num_p)

    cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec = prob.path_spline(x_path, y_path)

    idx = 0

    x_global_init = x_path[idx]
    y_global_init = y_path[idx]
    psi_global_init = jnp.arctan2((y_path[idx+1]-y_path[idx]), (x_path[idx+1]-x_path[idx]))

    v_global_init = 10.0
    vdot_global_init = 0.0
    psidot_global_init = 0*jnp.pi/180
    
    x_global_shifted = 0.0
    y_global_shifted = 0.0

  # ###################################################### Optimization paramters Lagrange multipliers
    
    lamda_x = jnp.zeros((prob.num_batch, prob.nvar))
    lamda_y = jnp.zeros((prob.num_batch, prob.nvar))
    d_a = prob.a_max*jnp.ones((prob.num_batch, prob.num))
    alpha_a = jnp.zeros((prob.num_batch, prob.num))		
    alpha_v = jnp.zeros((prob.num_batch, prob.num))		
    d_v = prob.v_max*jnp.ones((prob.num_batch, prob.num))

    s_lane = jnp.zeros((prob.num_batch, 2*(prob.num-1)))
    s_long = jnp.zeros((prob.num_batch, (prob2.num_up-1)))

    ##################################################################

    v_des = 10

    ################################################## generating random samples for v_des, y_des

    mean_vx_1 = 0
    mean_vx_2 = 0
    mean_vx_3 = 0
    mean_vx_4 = 0

    mean_y_des_1 = 0.0
    mean_y_des_2 = 0.0
    mean_y_des_3 = 0.0
    mean_y_des_4 = 0.0

    mean_param = jnp.hstack(( mean_vx_1, mean_vx_2, mean_vx_3, mean_vx_4, mean_y_des_1, mean_y_des_2, mean_y_des_3, mean_y_des_4))

    diag_param = np.hstack(( 14, 14, 14, 14, 23.0, 23.0, 23.0, 23.0  ))

    cov_param = jnp.asarray(np.diag(diag_param))
    
    ## Lane constraints
    y_lane_lb = -0.3
    y_lane_ub = 0.3

    rootdir = "data_synthetic/"

    for filename in os.scandir(rootdir):
        if("_val" in filename.name):
            continue
        else:
            data=np.load(rootdir + filename.name)
            filename_val = filename.name.split(".",1)[0] + "_val.npz"
            data_val = np.load(rootdir + filename_val)

            print("filename ",filename.name,filename_val)

            x_obs_up_val=data_val["x_obs"] 
            y_obs_up_val=data_val["y_obs"]
        
            key,subkey = random.split(key)
            indx_random = jax.random.choice(key,jnp.arange(prob2.num_validation),(prob2.num_batch,1),replace=False).reshape(prob2.num_batch)
            
            x_obs_up = data["x_obs"][indx_random]
            y_obs_up = data["y_obs"][indx_random]

            cx = data["cx"][indx_random]
            cy = data["cy"][indx_random]

            B = np.hstack((cx,cy))
            ker_red,ker_mixed,ker_total = prob2.kernel_comp(B,B)
            Q = ker_total
            q = -(1/prob2.num_batch)*jnp.sum(ker_total.T,axis=1)

            beta_best = prob3.compute_cem(Q,q)
            idx_beta = jnp.argsort(jnp.abs(beta_best))
            cx_r_sparse = cx[idx_beta[prob3.num_samples-prob3.num_samples_reduced_set:prob3.num_samples]]
            cy_r_sparse = cy[idx_beta[prob3.num_samples-prob3.num_samples_reduced_set:prob3.num_samples]]
            x_obs_r_up_sparse = x_obs_up[idx_beta[prob3.num_samples-prob3.num_samples_reduced_set:prob3.num_samples]]
            y_obs_r_up_sparse = y_obs_up[idx_beta[prob3.num_samples-prob3.num_samples_reduced_set:prob3.num_samples]]

            x_obs_mean_up = np.mean(x_obs_r_up_sparse,axis=0)
            y_obs_mean_up = np.mean(y_obs_r_up_sparse,axis=0)

            A = np.hstack((cx_r_sparse,cy_r_sparse))
            ker_red,ker_mixed,ker_total = prob2.kernel_comp(A,B)
            beta = prob2.compute_beta_reduced(ker_red,ker_mixed).reshape(prob2.num_reduced)
    
            data = []

            num_samples_baseline = prob2.num_reduced

            x_ego_initial = x_global_init + v_global_init*np.cos(psi_global_init)*prob.tot_time
            y_ego_initial = y_global_init + v_global_init*np.sin(psi_global_init)*prob.tot_time
            cost = prob2.compute_f_bar_baseline_vmap(x_ego_initial,y_ego_initial,x_obs_up,y_obs_up) # timesteps x samples
            cost = cost.T # samples x timesteps
            cost = np.amax(cost,axis=1)
            idx_cost = jnp.argsort(cost)

            x_obs_r_single_baseline = x_obs_up[idx_cost[-prob2.num_reduced:]]
            y_obs_r_single_baseline = y_obs_up[idx_cost[-prob2.num_reduced:]]

            x_obs_up = x_obs_up - x_global_init
            y_obs_up = y_obs_up - y_global_init
            x_obs_r_up_sparse = x_obs_r_up_sparse - x_global_init
            y_obs_r_up_sparse = y_obs_r_up_sparse - y_global_init
            x_obs_mean_up = x_obs_mean_up - x_global_init
            y_obs_mean_up = y_obs_mean_up - y_global_init
            x_obs_up_val = x_obs_up_val - x_global_init
            y_obs_up_val = y_obs_up_val - y_global_init
            
            x_waypoints, y_waypoints, phi_Waypoints = prob.waypoint_generator(x_global_init, y_global_init, x_path, y_path, 
                                                                            arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length)
            
            ########### shifting waypoints to the center of the ego vehicle
            x_waypoints_shifted = x_waypoints-x_global_init
            y_waypoints_shifted = y_waypoints-y_global_init

            x_global_shifted = 0.0 #x_global_init
            y_global_shifted = 0.0 #y_global_init

            global_initial_state = np.hstack(( x_global_shifted, y_global_shifted, v_global_init, vdot_global_init, psi_global_init, psidot_global_init))
            
            mmd, baseline,initial_state_frenet,obs_data_frenet,obs_traj_frenet\
            = prob.compute_mpc_command(x_obs_up_val,y_obs_up_val,num_samples_baseline,x_obs_r_single_baseline,y_obs_r_single_baseline,x_obs_up ,y_obs_up ,\
                x_obs_mean_up ,y_obs_mean_up ,beta,x_obs_r_up_sparse,y_obs_r_up_sparse,x_waypoints_shifted, y_waypoints_shifted,global_initial_state, lamda_x, lamda_y, 
                                        v_des, y_lane_lb, y_lane_ub, mean_param, cov_param, s_lane,s_long)
           
            data_set = (mmd,baseline,initial_state_frenet,obs_data_frenet,obs_traj_frenet)
            data.append(data_set)
            
            data_obs = x_obs_r_single_baseline,y_obs_r_single_baseline,x_obs_up,y_obs_up,x_obs_r_up_sparse,y_obs_r_up_sparse,x_obs_mean_up,y_obs_mean_up,x_obs_up_val,y_obs_up_val

            np.savez("data_comparison/" + filename.name, data=np.asanyarray(data,dtype=object),\
                        data_obs=np.asanyarray(data_obs,dtype=object))
            
if __name__ == '__main__':
    main()