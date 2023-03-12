import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import kernel_computation
import compute_beta
from celluloid import Camera
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ydes_1", type=float, default=0.0, help="desired y-coordinate")
    parser.add_argument("--ydes_2",  type=float, default=3.5, help="desired y-coordinate")
    parser.add_argument("--prob", type=float, default=0.1, help="desired probability corresponding to ydes_1")

    args = parser.parse_args()
    ydes_1 = args.ydes_1
    ydes_2 = args.ydes_2
    prob_des_1 = args.prob
    prob_des_2 = 1-prob_des_1

    kernel_obj = kernel_computation.kernel_matrix()
    beta_obj = compute_beta.beta_cem()
        
    ## Initial state of the obstacle
    x_init = 0.0
    y_init = 0.0
    vx_init = 5.0
    vy_init = 0.0
    ax_init = 0.0
    ay_init = 0.0

    ## Boundary conditions for the obstacle trajectories
    b_eq_x,b_eq_y = kernel_obj.compute_boundary_vec(x_init,vx_init,ax_init,y_init,vy_init,ay_init)

    ## Generating trajectory samples; y is sampled from a discrete distribution and x is sampled via sampling velocity in the x-direction from a gaussian
    mean_vx = 5
    mean_param = mean_vx
    diag_param = 2
    cov_param = diag_param

    ydes= np.asarray([ydes_1,ydes_2])
    probabilities = np.hstack((prob_des_1,prob_des_2))

    y_samples = np.random.choice(ydes, kernel_obj.num_batch, p=list(probabilities))
    cx,cy,x_samples,y_samples,v_des,rv_vel = kernel_obj.compute_obs_guess(b_eq_x,b_eq_y,mean_param,cov_param,y_samples)

    x_samples = np.array(x_samples)
    y_samples = np.array(y_samples)

    ## Top 10 most probable ground truth samples
    idx_samples = np.argsort(rv_vel.pdf(v_des))
    x_samples_top = x_samples[idx_samples[-kernel_obj.num_reduced:]]
    y_samples_top = y_samples[idx_samples[-kernel_obj.num_reduced:]]

    ## kernel matrix computation for reduced set selection
    B = np.hstack((cx,cy))
    ker_red,ker_mixed,ker_total = kernel_obj.kernel_comp(B,B)
    Q = ker_total
    q = -(1/kernel_obj.num_batch)*np.sum(ker_total.T,axis=1)

    cost_cem,beta = beta_obj.compute_cem(Q,q)
    
    ##### Matplotlib stuff
    fig = plt.figure(figsize=(12,6))
    camera = Camera(fig)

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.axis("equal")
    ax1.set_xlim([-5,35])
    ax1.set_ylim([-20,20])

    patch_global_obs = pt.Ellipse((x_init,y_init),width=6.0,height=3.0,angle=0.0,facecolor='green',edgecolor='black')

    fig.tight_layout(pad=1.0)

    linewidth = 0.5
    text_x = -4.6

    for i in range(0,len(cost_cem)+3):
        print(i)
        if i<=2:
            for ii in range(0,x_samples.shape[0]):
                ax1.plot(x_samples[ii,:],y_samples[ii,:],color = "r",linewidth= linewidth,alpha = 0.08)
            for ii in range(0,x_samples_top.shape[0]):
                ax1.plot(x_samples_top[ii,:],y_samples_top[ii,:],color="k",linewidth=linewidth, alpha = 0.8)

            ax1.text(text_x,15,"Samples drawn from mixture of Discrete Binomial and Gaussian distributions " "\n"
                    "\n"
                    "Top 10 most probable ground truth samples (black)")
        else:
            beta_best = beta[i-3]
            idx_beta = np.argsort(np.abs(beta_best))
           
            indx = idx_beta[beta_obj.num_samples-beta_obj.num_samples_reduced_set:beta_obj.num_samples]
            indx_diff = idx_beta[0:beta_obj.num_samples-beta_obj.num_samples_reduced_set]
            y_samples_diff = y_samples[indx_diff]
            x_samples_diff = x_samples[indx_diff]
            alpha_diff = np.abs(beta_best[indx_diff])/(np.amax(np.abs(beta_best)))
            alpha_diff = np.clip(alpha_diff,0,1)

            x_obs_r_up_sparse = x_samples[indx]
            y_obs_r_up_sparse = y_samples[indx]

            ax1.text(text_x,15,"As the CEM iters. progress, the top 10 samples(dark red) move closer to the " "\n"
                     "top 10 most probable ground truth samples(black)")
            
            ax1.text(text_x,12,"Mean of top 10 samples: {}".format(np.mean(np.abs(beta_best[indx]))))
            ax1.text(text_x,9,"Mean of bottom 90 samples: {}".format(np.mean(np.abs(beta_best[indx_diff]))))

            for ii in range(0,len(x_obs_r_up_sparse)):
                ax1.plot(x_obs_r_up_sparse[ii,:],y_obs_r_up_sparse[ii,:],color="r",linewidth=3*linewidth,alpha=1)
            for ii in range(0,x_samples_top.shape[0]):
                ax1.plot(x_samples_top[ii,:],y_samples_top[ii,:],color="k",linewidth=linewidth, alpha = 1)
            for jj in range(0,len(x_samples_diff)):
                ax1.plot(x_samples_diff[jj,:],y_samples_diff[jj],color="r", linewidth=linewidth,alpha= 0.08)
                   
            ax2.plot(cost_cem[0:i-3],color = "b",linewidth = 4 )
            ax2.text(40,2.5,"Iteration: {} CEM Cost: {}".format(i-3,cost_cem[i]))
        ax1.add_patch(patch_global_obs)
        
        camera.snap()

    animation = camera.animate()
    animation.save('videos/animation.mp4')
   
if __name__ == '__main__':
    main()

