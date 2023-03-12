import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random,vmap,lax
import jax
from scipy.interpolate import CubicSpline
import jax.lax as lax
import pol_matrix_comp

class kernel_matrix():
    def __init__(self):
        self.v_min = 0.1
        self.v_max = 30
        self.a_obs = 4.5
        self.b_obs = 3.5
        self.num_obs = 1
        self.num_circles = 3
        
        self.num_batch = 100
        self.num_reduced = 10
        self.num_validation = 1000

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
        
        self.A_eq_beta = jnp.ones((1,self.num_reduced))
        self.b_eq_beta = jnp.ones((1,1))
        self.rho_beta = 1.0

        key = random.PRNGKey(0)

        self.key = key
        self.key_val = random.PRNGKey(42)

        #### Kernel
        self.sigma = 30
        self.kernel_eval_red_vmap = jit(vmap(self.kernel_eval_red,in_axes=(0,None)))
        self.ker_del = jnp.ones((self.num_reduced*self.num_circles,self.num_reduced*self.num_circles))

        self.compute_f_bar_vmap = jit(vmap(self.compute_f_bar,in_axes=(0,0,None,None)))
        self.compute_f_vmap = jit(vmap(self.compute_f,in_axes=(0,0,None,None)))  
        self.compute_mmd_vmap = jit(vmap(self.compute_mmd,in_axes=(None,0)))

        self.kernel_comp_mmd_vmap = jit(vmap(self.kernel_comp_mmd,in_axes=(1,1)))
        self.kernel_eval_mmd_vmap = jit(vmap(self.kernel_eval_mmd,in_axes=(0,None)))

        self.num_samples_saa = self.num_reduced*self.num_circles

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

        self.compute_f_bar_baseline_vmap = jit(vmap(self.compute_f_bar_baseline,in_axes=(0,0,1,1)))
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

    #---gives cost_bar at time instant k=0,1,2,.. for all obstacle samples;num_reduced x 100
    @partial(jit, static_argnums=(0,))
    def compute_f_bar(self,x,y,x_obs,y_obs): # x,y represents one ego trajectory 
                                             # x_obs,y_obs represents jth obstacle trajectory waypoint at time instant k for all samples; num_reduced x 100
        wc_alpha = (x[0:self.num_up]-x_obs)
        ws_alpha = (y[0:self.num_up]-y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) + jnp.ones((self.num_reduced*self.num_circles,self.num_up))
        cost_bar = jnp.maximum(jnp.zeros((self.num_reduced*self.num_circles,self.num_up)), cost)
        
        return cost_bar # num_reduced x num_up
    
    @partial(jit, static_argnums=(0,))
    def compute_f(self,x,y,x_obs,y_obs): # x,y represents one ego trajectory 
                                             # x_obs,y_obs represents jth obstacle trajectory waypoint at time instant k for all samples; num_reduced x 100
        wc_alpha = (x[0:self.num_up]-x_obs)
        ws_alpha = (y[0:self.num_up]-y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) + jnp.ones((self.num_samples_saa,self.num_up))
        
        return cost # num_reduced x num_up

    # mmd cost for all time instants for all obstacle samples;matrix dims are num x (samples x samples) i.e 100 matrices each corr. to a time instant
    @partial(jit, static_argnums=(0,))
    def compute_mmd(self,beta,cost):
        beta_del = (1/(self.num_reduced))*jnp.ones((self.num_reduced))
        beta_del = jnp.repeat(beta_del,self.num_circles).reshape((self.num_reduced*self.num_circles,1))
        mmd_cost_init = jnp.zeros((self.num_up,1))
        beta = jnp.repeat(beta,self.num_circles)
        beta = beta.reshape((self.num_reduced*self.num_circles,1))

        cost = cost.reshape((self.num_reduced*self.num_circles,self.num_up))
        ker_cost,ker_cost_del,temp2 = self.kernel_comp_mmd_vmap(cost,jnp.zeros((self.num_reduced*self.num_circles,self.num_up)))
        
        def lax_mmd(carry,idx):
            mmd_cost = carry
            ker_cost_idx = ker_cost[idx].reshape((self.num_reduced*self.num_circles,self.num_reduced*self.num_circles)) #+ 0.1*jnp.identity(self.num_reduced)
            ker_cost_del_idx = ker_cost_del[idx].reshape((self.num_reduced*self.num_circles,self.num_reduced*self.num_circles)) # + 0.0001*jnp.identity(self.num_reduced)
            
            temp = jnp.dot(beta.T,jnp.dot(ker_cost_idx,beta))\
                    -2*jnp.dot(beta.T,jnp.dot(ker_cost_del_idx,beta_del))\
                         + jnp.dot(beta_del.T,jnp.dot(self.ker_del,beta_del))
            temp = temp.reshape(1)
            mmd_cost = mmd_cost.at[idx].set(temp)

            return mmd_cost,beta_del

        carry_init = mmd_cost_init
        carry_final,result = lax.scan(lax_mmd,carry_init,jnp.arange(self.num_up))
        mmd_cost = carry_final # 100x1

        return jnp.sum(mmd_cost)

    @partial(jit, static_argnums=(0,))
    def kernel_eval_mmd(self,a,A):
        return jnp.exp(-(1/(2*self.sigma**2))*jnp.absolute(a-A)**2)    #*jnp.linalg.norm(a-A,axis=0)**2)
    
    @partial(jit, static_argnums=(0,))
    def kernel_comp_mmd(self,a,b):
        
        kernel_matrix_a = self.kernel_eval_mmd_vmap(a,a)
        kernel_matrix_mixed = self.kernel_eval_mmd_vmap(b,a)
        kernel_matrix_b = self.kernel_eval_mmd_vmap(b,b)

        return kernel_matrix_a,kernel_matrix_mixed.T,kernel_matrix_b

    @partial(jit, static_argnums=(0,))
    def compute_f_bar_temp(self,x,y,x_obs,y_obs): # x,y represents one ego trajectory 
                                             # x_obs,y_obs represents jth obstacle trajectory waypoint at time instant k for all samples; num_reduced x 100
        wc_alpha = (x[0:self.num_up]-x_obs)
        ws_alpha = (y[0:self.num_up]-y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) +  jnp.ones((self.num_batch*self.num_circles,self.num_up))
        cost_bar = jnp.maximum(jnp.zeros((self.num_batch*self.num_circles,self.num_up)), cost)
        
        return cost_bar # num_batch x num

    @partial(jit, static_argnums=(0,))
    def compute_f_bar_temp_val(self,x,y,x_obs,y_obs): # x,y represents one ego trajectory 
                                             # x_obs,y_obs represents jth obstacle trajectory waypoint at time instant k for all samples; num_reduced x 100
        wc_alpha = (x[0:self.num_up]-x_obs)
        ws_alpha = (y[0:self.num_up]-y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) +  jnp.ones((self.num_validation*self.num_circles,self.num_up))
        cost_bar = jnp.maximum(jnp.zeros((self.num_validation*self.num_circles,self.num_up)), cost)
        
        return cost_bar # num_batch x num
    
    @partial(jit, static_argnums=(0,))
    def compute_beta_reduced(self,ker_red,ker_mixed):
        cost = self.rho_beta*ker_red + 0.01*jnp.identity(self.num_reduced)

        # 11x11
        cost_mat = jnp.vstack((  jnp.hstack(( cost, self.A_eq_beta.T )), jnp.hstack(( self.A_eq_beta, jnp.zeros(( jnp.shape(self.A_eq_beta)[0], jnp.shape(self.A_eq_beta)[0] )) )) ))
        
        lincost = -self.rho_beta*(1/self.num_batch)*jnp.sum(ker_mixed.T,axis=1).reshape(self.num_reduced,1)

        sol_beta = jnp.linalg.solve(cost_mat,jnp.vstack(( -lincost, self.b_eq_beta )))

        primal_sol_beta = sol_beta[0:self.num_reduced]

        return primal_sol_beta

    @partial(jit, static_argnums=(0,))
    def compute_coeff(self,x,y):

        cost = jnp.dot(self.P_jax_up_reduced.T, self.P_jax_up_reduced) + 0.001*jnp.identity(self.nvar_red_up)
        
        lincost_x = -jnp.dot(self.P_jax_up_reduced.T, x.T ).T 
        lincost_y = -jnp.dot(self.P_jax_up_reduced.T, y.T ).T 
        
        cx = jnp.linalg.solve(-cost, lincost_x.T).T
        cy = jnp.linalg.solve(-cost, lincost_y.T).T

        return cx,cy

    def path_spline(self, x_path, y_path):

        time_vec = np.linspace(0, self.t_fin, np.shape(x_path)[0])

        cs_x_path = CubicSpline(time_vec, x_path)
        cs_y_path = CubicSpline(time_vec, y_path)

        return cs_x_path, cs_y_path
    
    @partial(jit, static_argnums=(0,))
    def compute_f_bar_current(self,x,y,x_obs,y_obs): 
        wc_alpha = (x - x_obs)
        ws_alpha = (y - y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) + jnp.ones(self.num_batch)
        # cost_bar = jnp.maximum(jnp.zeros((self.num_batch,self.num_up)), cost)
        return cost
    
    @partial(jit, static_argnums=(0,))
    def compute_f_bar_baseline(self,x,y,x_obs,y_obs): 
        wc_alpha = (x - x_obs)
        ws_alpha = (y - y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) + jnp.ones(self.num_batch)
        # cost_bar = jnp.maximum(jnp.zeros(self.num_batch), cost)
        # idx_cost = jnp.argsort(cost)
        return cost #x_obs[idx_cost[-self.num_reduced:]],y_obs[idx_cost[-self.num_reduced:]]
    
    @partial(jit, static_argnums=(0,))	
    def compute_obs_guess(self,b_eq_x,b_eq_y,mean_param,cov_param,y_samples):
        v_des = self.sampling_param(mean_param,cov_param)
        y_des = y_samples

        #############################
        A_vd = self.Pddot_jax-self.k_p_v*self.Pdot_jax
        b_vd = -self.k_p_v*jnp.ones((self.num_validation, self.num))*(v_des)[:, jnp.newaxis]
        
        A_pd = self.Pddot_jax-self.k_p*self.P_jax#-self.k_d*self.Pdot_jax
        b_pd = -self.k_p*jnp.ones((self.num_validation, self.num ))*(y_des)[:, jnp.newaxis]

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

        return primal_sol_x, primal_sol_y ,x,y

    @partial(jit, static_argnums=(0,))	
    def sampling_param(self,mean_param, cov_param):
        key, subkey = random.split(self.key)
        param_samples = jax.random.normal(key,(self.num_validation, ))
        param_samples = mean_param + jnp.sqrt(cov_param)*param_samples
        v_des = param_samples

        v_des = jnp.clip(v_des, self.v_min*jnp.ones(self.num_validation), self.v_max*jnp.ones(self.num_validation)   )
    
        neural_output_batch = v_des

        return neural_output_batch

    @partial(jit, static_argnums=(0,))	
    def compute_obs_guess_val(self,b_eq_x,b_eq_y,mean_param,cov_param,y_samples):

        v_des = self.sampling_param_val(mean_param,cov_param)
        y_des = y_samples

        #############################
        A_vd = self.Pddot_jax-self.k_p_v*self.Pdot_jax
        b_vd = -self.k_p_v*jnp.ones((self.num_validation, self.num))*(v_des)[:, jnp.newaxis]
        
        A_pd = self.Pddot_jax-self.k_p*self.P_jax#-self.k_d*self.Pdot_jax
        b_pd = -self.k_p*jnp.ones((self.num_validation, self.num ))*(y_des)[:, jnp.newaxis]

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

        return primal_sol_x, primal_sol_y ,x,y

    @partial(jit, static_argnums=(0,))	
    def sampling_param_val(self, mean_param, cov_param):

        key, subkey = random.split(self.key_val)
        param_samples = jax.random.normal(key,(self.num_validation, ))
        param_samples = mean_param + jnp.sqrt(cov_param)*param_samples
        v_des = param_samples
                
        v_des = jnp.clip(v_des, self.v_min*jnp.ones(self.num_validation), self.v_max*jnp.ones(self.num_validation)   )
    
        neural_output_batch = v_des

        return neural_output_batch
    
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