import numpy as np
import matplotlib.pyplot as plt
import kernel_computation
import matplotlib.pyplot as plt
import matplotlib.patches as pt
from jax import random

def main():
    prob2 = kernel_computation.kernel_matrix()
    key = random.PRNGKey(0)

    ## Specify the desired y's
    ydes_1 = 0.0
    ydes_2 = 3.5

    ## Specify the range of x and y for obstacle initial position
    obs_x_arr = np.array([45,60,75])
    obs_y_arr = np.array([-3,0])

    ## Specify the range of probabilities corresponding to ydes_1
    prob_arr = np.linspace(0.05,0.3,6)
    
    ## Store the data in this folder
    path = "data_synthetic/"

    for obs_x in obs_x_arr:
        for obs_y in obs_y_arr:
            for prob in prob_arr:
                prob_des_1 = prob
                prob_des_2 = 1-prob_des_1

                x_init = obs_x
                y_init = obs_y
                vx_init = 5.0
                vy_init = 0.0
                ax_init = 0.0
                ay_init = 0.0

                b_eq_x,b_eq_y = prob2.compute_boundary_vec(x_init,vx_init,ax_init,y_init,vy_init,ay_init)

                mean_vx = 5
                mean_param = mean_vx
                diag_param = 2
                cov_param = diag_param

                y_des= np.asarray([ydes_1,ydes_2])
                probabilities = np.hstack((prob_des_1,prob_des_2 ))
                
                key,subkey = random.split(key)
                y_samples = np.random.choice(y_des, prob2.num_validation, p=list(probabilities))
                cx,cy,x_obs,y_obs = prob2.compute_obs_guess(b_eq_x,b_eq_y,mean_param,cov_param,y_samples)

                x_obs = np.array(x_obs)
                y_obs = np.array(y_obs)

                ## Store the training data with this filename
                filename = "{}_{}_{}".format(int(obs_x),int(obs_y),int(prob*100))
                np.savez(path + filename,cx=cx,cy=cy,x_obs=x_obs,y_obs=y_obs)

                key,subkey = random.split(key)
                y_samples = np.random.choice(y_des, prob2.num_validation, p=list(probabilities))
                cx,cy,x_obs,y_obs = prob2.compute_obs_guess_val(b_eq_x,b_eq_y,mean_param,cov_param,y_samples)
                
                x_obs = np.array(x_obs)
                y_obs = np.array(y_obs)

                ## Store the validation data with this filename
                filename_val = "{}_{}_{}_val".format(int(obs_x),int(obs_y),int(prob*100))

                np.savez(path + filename_val,cx=cx,cy=cy,x_obs=x_obs,y_obs=y_obs)

if __name__ == '__main__':
    main()
