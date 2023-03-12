import jax.numpy as jnp
from functools import partial
from jax import jit,vmap
import pol_matrix_comp
from scipy.stats import multivariate_normal

class kernel_matrix():
    def __init__(self):
        self.v_min = 0.1
        self.v_max = 30

        self.num_batch = 100
        self.num_reduced = 10
        self.num_validation = self.num_batch

        self.t_fin = 15
        self.num = 100
        self.t = self.t_fin/self.num
        self.ellite_num = 50
        self.ellite_num_projection = 150

        tot_time = jnp.linspace(0, self.t_fin, self.num)
        self.tot_time = tot_time
        tot_time_copy = tot_time.reshape(self.num, 1)

        self.P, self.Pdot, self.Pddot = pol_matrix_comp.pol_matrix_comp(tot_time_copy)

        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

        self.nvar = jnp.shape(self.P_jax)[1]
                
        self.num_up = self.num

        tot_time_up = jnp.linspace(0, self.t_fin, self.num_up)
        self.tot_time_up = tot_time_up.reshape(self.num_up, 1)
        self.t_up = self.t_fin/self.num_up

        self.P_up_reduced, self.Pdot_up_reduced, self.Pddot_up_reduced = pol_matrix_comp.pol_matrix_comp(self.tot_time_up)

        self.P_jax_up_reduced, self.Pdot_jax_up_reduced, self.Pddot_jax_up_reduced = jnp.asarray(self.P_up_reduced), jnp.asarray(self.Pdot_up_reduced), jnp.asarray(self.Pddot_up_reduced)
        
        self.nvar_red_up = jnp.shape(self.P_jax_up_reduced)[1]
        
        #### Kernel
        self.sigma = 30
        self.kernel_eval_red_vmap = jit(vmap(self.kernel_eval_red,in_axes=(0,None)))

        self.weight_smoothness_x = 1
        self.weight_smoothness_y = 1

        self.rho_v = 1 
        self.rho_offset = 1

        self.k_p_v = 2
        self.k_d_v = 2.0*jnp.sqrt(self.k_p_v)

        self.k_p = 2
        self.k_d = 2.0*jnp.sqrt(self.k_p)
        
        self.A_eq_x = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0]  ))
        self.A_eq_y = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.Pdot_jax[-1]  ))

#################################################

    @partial(jit, static_argnums=(0,))
    def kernel_eval_red(self,a,A):
        return jnp.exp(-(1/(2*self.sigma**2))*jnp.linalg.norm(a-A,axis=1)**2)
    
    @partial(jit, static_argnums=(0,))
    def kernel_comp(self,A,B): # A-reduced sample set matrix(m x 22) , B - total samples matrix (n x 22)
        
        kernel_matrix_reduced = self.kernel_eval_red_vmap(A,A)
        kernel_matrix_mixed = self.kernel_eval_red_vmap(B,A)
        kernel_matrix_total = self.kernel_eval_red_vmap(B,B)
        return kernel_matrix_reduced,kernel_matrix_mixed,kernel_matrix_total

    # @partial(jit, static_argnums=(0,))	
    def compute_obs_guess(self,b_eq_x,b_eq_y,mean_param,cov_param,y_samples):

        v_des ,rv_vel = self.sampling_param(mean_param,cov_param)
        y_des = y_samples

        #############################
        A_vd = self.Pddot_jax-self.k_p_v*self.Pdot_jax
        b_vd = -self.k_p_v*jnp.ones((self.num_batch, self.num))*(v_des)[:, jnp.newaxis]
        
        A_pd = self.Pddot_jax-self.k_p*self.P_jax#-self.k_d*self.Pdot_jax
        b_pd = -self.k_p*jnp.ones((self.num_batch, self.num ))*(y_des)[:, jnp.newaxis]

        cost_smoothness_x = self.weight_smoothness_x*jnp.identity(self.nvar)
        cost_smoothness_y = self.weight_smoothness_y*jnp.identity(self.nvar)
        
        cost_x = cost_smoothness_x+self.rho_v*jnp.dot(A_vd.T, A_vd)
        cost_y = cost_smoothness_y+self.rho_offset*jnp.dot(A_pd.T, A_pd)

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -self.rho_v*jnp.dot(A_vd.T, b_vd.T).T
        lincost_y = -self.rho_offset*jnp.dot(A_pd.T, b_pd.T).T
    
        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        #######################

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]
       
        c_x = primal_sol_x[0]
        c_y = primal_sol_y[0]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        return primal_sol_x, primal_sol_y ,x,y,v_des,rv_vel

    # @partial(jit, static_argnums=(0,))	
    def sampling_param(self,mean_param, cov_param):
        rv_vel = multivariate_normal(mean_param,cov_param,seed=0)
        param_samples = rv_vel.rvs(size=(self.num_batch,1))
        v_des = param_samples
                
        v_des = jnp.clip(v_des, self.v_min*jnp.ones(self.num_batch), self.v_max*jnp.ones(self.num_batch)   )
    
        neural_output_batch = v_des

        return neural_output_batch,rv_vel
    
    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init):

        x_init_vec = x_init*jnp.ones((self.num_validation, 1))
        y_init_vec = y_init*jnp.ones((self.num_validation, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_validation, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_validation, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_validation, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_validation, 1))

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((self.num_validation, 1  ))   ))

        return b_eq_x, b_eq_y