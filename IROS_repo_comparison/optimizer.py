import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random,vmap,lax
import jax
import pol_matrix_comp
from scipy.interpolate import CubicSpline
import jax.lax as lax
import kernel_computation

class mpc_path_following():

    def __init__(self):
        self.r1 = 0.0
        self.r2 = 2.5
        self.r3 = -2.5
        self.dist_centre = np.array([self.r1,self.r2,self.r3])
        self.num_circles = 3

        self.margin = 0.6
        
        self.prob = kernel_computation.kernel_matrix()

        self.v_max = 30.0 
        self.v_min = 0.1
        self.a_max = 18.0
        self.a_centr = 1.5
        self.num_obs = 1
        self.num_batch = 1000
        self.steer_max = 1.2
        self.kappa_max = 0.895
        self.wheel_base = 2.875
        self.a_obs = 4.5 ########### rectangle of car and ego: length = 4, width = 1.4. We over approximate a_obs and b_obs to account for orientation change of obstacle and ego.
        self.b_obs = 3.5
    
        self.t_fin = 15
        self.num = 100
        self.t = self.t_fin/self.num
        self.ellite_num = 50
        self.ellite_num_projection = 150

        tot_time = np.linspace(0, self.t_fin, self.num)
        self.tot_time = tot_time
        tot_time_copy = tot_time.reshape(self.num, 1)

        self.P, self.Pdot, self.Pddot = pol_matrix_comp.pol_matrix_comp(tot_time_copy)

        # self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

        
        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

        self.nvar = jnp.shape(self.P_jax)[1]

        ################################################################
        self.A_eq_x = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0]  ))
        self.A_eq_y = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.Pdot_jax[-1]  ))
                    
        self.A_vel = self.Pdot_jax 
        self.A_acc = self.Pddot_jax
        self.A_projection = jnp.identity(self.nvar)
        
        ###################################3 obstacle avoidance		
        self.A_y_centerline = self.P_jax
        self.A_obs = jnp.tile(self.P_jax[0:self.prob.num_up], ((self.num_obs)*self.num_circles, 1))
        
        self.factor= self.prob.num_reduced
        self.A_obs_baseline = jnp.tile(self.P_jax[0:self.prob.num_up], (self.factor*(self.num_obs)*self.num_circles, 1))

        self.A_lane = jnp.vstack(( self.P_jax, -self.P_jax    ))

        ###################################################

        self.rho_nonhol = 1.0
        self.rho_ineq = 1
        self.rho_obs = 1.0
        self.rho_projection = 1.0
        self.rho_goal = 1.0
        self.rho_lane = 1.0
        self.rho_long = 1.0

        self.rho_v = 1 
        self.rho_offset = 1

        self.weight_smoothness_x = 1
        self.weight_smoothness_y = 1

        #################################################
        self.maxiter = 10
        self.maxiter_baseline = 10
        self.maxiter_cem = 20
        self.maxiter_cem_baseline = 20

        self.k_p_v = 2
        self.k_d_v = 2.0*jnp.sqrt(self.k_p_v)

        self.k_p = 2
        self.k_d = 2.0*jnp.sqrt(self.k_p)

        self.P_jax_1 = self.P_jax[0:25, :]
        self.P_jax_2 = self.P_jax[25:50, :]
        self.P_jax_3 = self.P_jax[50:75, :]
        self.P_jax_4 = self.P_jax[75:100, :]

        self.Pdot_jax_1 = self.Pdot_jax[0:25, :]
        self.Pdot_jax_2 = self.Pdot_jax[25:50, :]
        self.Pdot_jax_3 = self.Pdot_jax[50:75, :]
        self.Pdot_jax_4 = self.Pdot_jax[75:100, :]
            
        self.Pddot_jax_1 = self.Pddot_jax[0:25, :]
        self.Pddot_jax_2 = self.Pddot_jax[25:50, :]
        self.Pddot_jax_3 = self.Pddot_jax[50:75, :]
        self.Pddot_jax_4 = self.Pddot_jax[75:100, :]

        self.num_partial = 25
        
        ###########################################3
        key = random.PRNGKey(0)

        self.key = key

        ################################################ Matrices for Custom path smoothing
        self.rho_smoothing = 1.0

        self.A_smoothing = jnp.identity(self.num)
        self.A_jerk_smoothing = jnp.diff(jnp.diff(jnp.diff(self.A_smoothing, axis = 0), axis = 0), axis = 0)
        self.A_acc_smoothing = jnp.diff(jnp.diff(self.A_smoothing, axis = 0), axis = 0)
        self.A_vel_smoothing = jnp.diff(self.A_smoothing, axis = 0)
        cost_jerk_smoothing = jnp.dot(self.A_jerk_smoothing.T, self.A_jerk_smoothing)
        cost_acc_smoothing = jnp.dot(self.A_acc_smoothing.T, self.A_acc_smoothing)
        cost_vel_smoothing = jnp.dot(self.A_vel_smoothing.T, self.A_vel_smoothing)

        self.A_eq_smoothing = self.A_smoothing[0].reshape(1, self.num)
# 
        cost_smoothing = (20)*(cost_jerk_smoothing+0.0*cost_acc_smoothing)+self.rho_smoothing*jnp.dot(self.A_smoothing.T, self.A_smoothing)
        
        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_smoothing, self.A_eq_smoothing.T )), jnp.hstack(( self.A_eq_smoothing, jnp.zeros(( jnp.shape(self.A_eq_smoothing)[0], jnp.shape(self.A_eq_smoothing)[0] )) )) ))
        self.cost_smoothing_inv = jnp.linalg.inv(cost_mat_x)

        # self.cost_smoothing_inv = jnp.linalg.inv(self.cost_smoothing)
        self.maxiter_smoothing = 10

        #####################################################################


        self.jax_interp = jit(jnp.interp) ############## jitted interp
        self.interp_vmap = jit(vmap(self.interp,in_axes=(0,None,None)))
        self.frenet_to_global_vmap = jit(vmap(self.frenet_to_global,in_axes=(0,0,0,0,0)))

        ##########################################################################

        self.alpha_mean = 0.6
        self.alpha_cov = 0.6

        self.lamda = 0.9
        self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

        self.closest_point_vmap = jit(vmap(self.closest_point,in_axes=(0,None)))
        self.global_to_frenet_obs_vmap = jit(vmap(self.global_to_frenet_obs,in_axes=(0,0,0,0,None,None,None,None,None,None)))
        self.global_to_frenet_obs_vmap_1 = jit(vmap(self.global_to_frenet_obs_vmap,in_axes=(0,0,0,0,None,None,None,None,None,None)))

        self.gamma = 1.0
        self.gamma_obs = 1.0
        self.gamma_obs_long = 0.9

        # upper lane bound
        self.P_ub_1 = self.P_jax[1:self.num,:]
        self.P_ub_0 = self.P_jax[0:self.num-1,:]
        self.A_ub = self.P_ub_1 + (self.gamma-1)*self.P_ub_0

        # lower lane bound
        self.P_lb_1 = -self.P_jax[1:self.num,:]
        self.P_lb_0 = self.P_jax[0:self.num-1,:]
        self.A_lb = self.P_lb_1 + (1-self.gamma)*self.P_lb_0

        # vstack A_ub and A_lb
        self.A_lane_bound = jnp.vstack((self.A_ub,self.A_lb))
        self.d_separate = 5.0

        self.d_obs_vmap = jit(vmap(self.comp_d_obs_prev,in_axes=0))
        self.d_obs_vmap_baseline = jit(vmap(self.comp_d_obs_prev_baseline,in_axes=0))
        ############# Longitudinal barrier 
        self.P_ub_11 = self.P_jax[1:self.prob.num_up,:]
        self.P_ub_01 = self.P_jax[0:self.prob.num_up-1,:]

        self.A_long = self.P_ub_11 + (self.gamma_obs_long-1)*self.P_ub_01
        # self.A_barrier_long = jnp.tile(self.A_long, (self.num_circles*self.num_obs,1))
        self.A_barrier_long = self.A_long

        self.jax_interp = jit(jnp.interp) ############## jitted interp

        self.num_validation = 1000

    def path_spline(self, x_path, y_path):

        x_diff = np.diff(x_path)
        y_diff = np.diff(y_path)

        phi = np.unwrap(np.arctan2(y_diff, x_diff))
        phi_init = phi[0]
        phi = np.hstack(( phi_init, phi  ))


        arc = np.cumsum( np.sqrt( x_diff**2+y_diff**2 )   )
        arc_length = arc[-1]

        arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])

        cs_x_path = CubicSpline(arc_vec, x_path)
        cs_y_path = CubicSpline(arc_vec, y_path)
        cs_phi_path = CubicSpline(arc_vec, phi)

        return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec
        

    def waypoint_generator(self, x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length):

        
        idx = np.argmin( np.sqrt((x_global_init-x_path_data)**2+(y_global_init-y_path_data)**2))
        arc_curr = arc_vec[idx]
        arc_pred = arc_curr + 1000
        arc_look = np.linspace(arc_curr, arc_pred, self.num)

        x_waypoints = cs_x_path(arc_look)
        y_waypoints =  cs_y_path(arc_look)
        phi_Waypoints = cs_phi_path(arc_look)

        return x_waypoints, y_waypoints, phi_Waypoints


    @partial(jit, static_argnums=(0,))
    def compute_x_smoothing(self, x_waypoints, y_waypoints, alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing):

        b_x_smoothing = x_waypoints+d_smoothing*jnp.cos(alpha_smoothing)
        b_y_smoothing = y_waypoints+d_smoothing*jnp.sin(alpha_smoothing)

        lincost_x = -lamda_x_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, b_x_smoothing)
        lincost_y = -lamda_y_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, b_y_smoothing)
        
        b_eq_smoothing_x = x_waypoints[0]
        b_eq_smoothing_y = y_waypoints[0]

        sol_x = jnp.dot(self.cost_smoothing_inv, jnp.hstack((-lincost_x, b_eq_smoothing_x)))
        sol_y = jnp.dot(self.cost_smoothing_inv, jnp.hstack((-lincost_y, b_eq_smoothing_y)))

        x_smooth = sol_x[0:self.num]
        y_smooth = sol_y[0:self.num]

        return x_smooth, y_smooth

    
    @partial(jit, static_argnums=(0,))
    def compute_alpha_smoothing(self, x_smooth, y_smooth, x_waypoints, y_waypoints, threshold, lamda_x_smoothing, lamda_y_smoothing):

        wc_alpha_smoothing = (x_smooth-x_waypoints)
        ws_alpha_smoothing = (y_smooth-y_waypoints)
        
        alpha_smoothing  = jnp.arctan2( ws_alpha_smoothing, wc_alpha_smoothing )

        c1_d_smoothing = (jnp.cos(alpha_smoothing)**2 + jnp.sin(alpha_smoothing)**2 )
        c2_d_smoothing = (wc_alpha_smoothing*jnp.cos(alpha_smoothing) + ws_alpha_smoothing*jnp.sin(alpha_smoothing)  )

        d_smoothing  = c2_d_smoothing/c1_d_smoothing
        d_smoothing  = jnp.minimum( d_smoothing, threshold*jnp.ones(self.num ) )

        res_x_smoothing = wc_alpha_smoothing-d_smoothing*jnp.cos(alpha_smoothing)
        res_y_smoothing = ws_alpha_smoothing-d_smoothing*jnp.sin(alpha_smoothing)

        lamda_x_smoothing = lamda_x_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, res_x_smoothing).T
        lamda_y_smoothing = lamda_y_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, res_y_smoothing).T


        return alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing


    @partial(jit, static_argnums=(0,))
    def compute_path_parameters(self, x_path, y_path):

        Fx_dot = jnp.diff(x_path)
        Fy_dot = jnp.diff(y_path)

        Fx_dot = jnp.hstack(( Fx_dot[0], Fx_dot  ))

        Fy_dot = jnp.hstack(( Fy_dot[0], Fy_dot  ))

            
        Fx_ddot = jnp.diff(Fx_dot)
        Fy_ddot = jnp.diff(Fy_dot)

        Fx_ddot = jnp.hstack(( Fx_ddot[0], Fx_ddot  ))

        Fy_ddot = jnp.hstack(( Fy_ddot[0], Fy_ddot  ))

        
        arc = jnp.cumsum( jnp.sqrt( Fx_dot**2+Fy_dot**2 )   )
        arc_vec = jnp.hstack((0, arc[0:-1] ))
        # arc_vec = arc 

        arc_length = arc_vec[-1]

        kappa = (Fy_ddot*Fx_dot-Fx_ddot*Fy_dot)/((Fx_dot**2+Fy_dot**2)**(1.5))

        return Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length



    @partial(jit, static_argnums=(0,))
    def global_to_frenet(self, x_path, y_path, initial_state, arc_vec, Fx_dot, Fy_dot, kappa ):

        x_global_init, y_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init = initial_state
        idx_closest_point = jnp.argmin( jnp.sqrt((x_path-x_global_init)**2+(y_path-y_global_init)**2))
        closest_point_x, closest_point_y = x_path[idx_closest_point], y_path[idx_closest_point]

        x_init = arc_vec[idx_closest_point]

        kappa_interp = self.jax_interp(x_init, arc_vec, kappa)
        kappa_pert = self.jax_interp(x_init+0.001, arc_vec, kappa)

        kappa_prime = (kappa_pert-kappa_interp)/0.001

        Fx_dot_interp = self.jax_interp(x_init, arc_vec, Fx_dot)
        Fy_dot_interp = self.jax_interp(x_init, arc_vec, Fy_dot)

        normal_x = -Fy_dot_interp
        normal_y = Fx_dot_interp

        normal = jnp.hstack((normal_x, normal_y   ))
        vec = jnp.asarray([x_global_init-closest_point_x,y_global_init-closest_point_y ])
        y_init = (1/(jnp.linalg.norm(normal)))*jnp.dot(normal,vec)
        
        psi_init = psi_global_init-jnp.arctan2(Fy_dot_interp, Fx_dot_interp)
        psi_init = jnp.arctan2(jnp.sin(psi_init), jnp.cos(psi_init))
        
        vx_init = v_global_init*jnp.cos(psi_init)/(1-y_init*kappa_interp)
        vy_init = v_global_init*jnp.sin(psi_init)

        psidot_init = psidot_global_init-kappa_interp*vx_init

        ay_init = vdot_global_init*jnp.sin(psi_init)+v_global_init*jnp.cos(psi_init)*psidot_init
        
        ax_init_part_1 = vdot_global_init*jnp.cos(psi_init)-v_global_init*jnp.sin(psi_init)*psidot_init
        ax_init_part_2 = -vy_init*kappa_interp-y_init*kappa_prime*vx_init

        ax_init = (ax_init_part_1*(1-y_init*kappa_interp)-(v_global_init*jnp.cos(psi_init))*(ax_init_part_2) )/((1-y_init*kappa_interp)**2)
            
        psi_fin = 0.0

        return x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init

   
    @partial(jit, static_argnums=(0,))
    def custom_path_smoothing(self,  x_waypoints, y_waypoints, threshold):

        alpha_smoothing_init = jnp.zeros(self.num)
        d_smoothing_init = threshold*jnp.ones(self.num)
        lamda_x_smoothing_init = jnp.zeros(self.num)
        lamda_y_smoothing_init = jnp.zeros(self.num)


        def lax_smoothing(carry,idx):
            alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing,x_smooth,y_smooth = carry
            x_smooth, y_smooth = self.compute_x_smoothing(x_waypoints, y_waypoints, alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing)
            alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing = self.compute_alpha_smoothing(x_smooth, y_smooth, x_waypoints, y_waypoints, threshold, lamda_x_smoothing, lamda_y_smoothing)
            
            return (alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing,x_smooth,y_smooth),x_smooth

        carry_init = (alpha_smoothing_init,d_smoothing_init,lamda_x_smoothing_init,lamda_y_smoothing_init,x_waypoints,y_waypoints)
        carry_final,result = lax.scan(lax_smoothing,carry_init,jnp.arange(self.maxiter_smoothing))

        alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing,x_smooth,y_smooth = carry_final

        return x_smooth, y_smooth

    @partial(jit, static_argnums=(0,))	
    def sampling_param(self, y_lane_lb, y_lane_ub, mean_param, cov_param):

        key, subkey = random.split(self.key)
        param_samples = jax.random.multivariate_normal(key, mean_param, cov_param, (self.num_batch, ))
        
        v_des_1 = param_samples[:, 0]
        v_des_2 = param_samples[:, 1]
        v_des_3 = param_samples[:, 2]
        v_des_4 = param_samples[:, 3]
            
        y_des_1 = param_samples[:, 4]
        y_des_2 = param_samples[:, 5]
        y_des_3 = param_samples[:, 6]
        y_des_4 = param_samples[:, 7]
        
        # y_des = jnp.clip(y_des, y_lane_lb*jnp.ones(self.num_batch), y_lane_ub*jnp.ones(self.num_batch)   )
        
        v_des_1 = jnp.clip(v_des_1, self.v_min*jnp.ones(self.num_batch), self.v_max*jnp.ones(self.num_batch)   )
        v_des_2 = jnp.clip(v_des_2, self.v_min*jnp.ones(self.num_batch), self.v_max*jnp.ones(self.num_batch)   )
        
        v_des_3 = jnp.clip(v_des_3, self.v_min*jnp.ones(self.num_batch), self.v_max*jnp.ones(self.num_batch)   )
        
        v_des_4 = jnp.clip(v_des_4, self.v_min*jnp.ones(self.num_batch), self.v_max*jnp.ones(self.num_batch)   )

        neural_output_batch = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4)).T
        neural_output_batch = neural_output_batch.at[-10:,:].set(jnp.zeros((10,8)))

        return neural_output_batch


    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init, y_des):

        x_init_vec = x_init*jnp.ones((self.num_batch, 1))
        y_init_vec = y_init*jnp.ones((self.num_batch, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((self.num_batch, 1  ))   ))

        return b_eq_x, b_eq_y

    @partial(jit, static_argnums=(0,))
    def initial_alpha_d_obs(self, x_obs,y_obs,x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, lamda_x, lamda_y):

        wc_alpha_temp = (x_guess[:,0:self.prob.num_up]-x_obs[:,jnp.newaxis])
        ws_alpha_temp = (y_guess[:,0:self.prob.num_up]-y_obs[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        factor = 1
        wc_alpha = wc_alpha.reshape(self.num_batch, factor*self.prob.num_up*((self.num_obs)*self.num_circles))
        ws_alpha = ws_alpha.reshape(self.num_batch, factor*self.prob.num_up*((self.num_obs)*self.num_circles))

        alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
        c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_temp = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_batch,  factor*self.prob.num_up*((self.num_obs)*self.num_circles)  )), d_temp   )

        ################# velocity terms

        wc_alpha_vx = xdot_guess
        ws_alpha_vy = ydot_guess
        alpha_v = jnp.unwrap(jnp.arctan2( ws_alpha_vy, wc_alpha_vx))
        
        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
        
        d_temp_v = c2_d_v/c1_d_v
        
        d_v = jnp.clip(d_temp_v, self.v_min, self.v_max )
        
        ################# acceleration terms

        wc_alpha_ax = xddot_guess
        ws_alpha_ay = yddot_guess
        alpha_a = jnp.unwrap(jnp.arctan2( ws_alpha_ay, wc_alpha_ax))
        
    
        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        d_temp_a = c2_d_a/c1_d_a
        d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max  )

        
        #########################################33
        res_ax_vec = xddot_guess-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot_guess-d_a*jnp.sin(alpha_a)
        
        res_vx_vec = xdot_guess-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot_guess-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

            
        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))

        
        lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T
        lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T
    
        return alpha_obs, d_obs,alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v
        
    
    @partial(jit, static_argnums=(0,))
    def initial_alpha_d_obs_baseline(self, x_obs,y_obs,x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, lamda_x, lamda_y):

        wc_alpha_temp = (x_guess[:,0:self.prob.num_up]-x_obs[:,jnp.newaxis])
        ws_alpha_temp = (y_guess[:,0:self.prob.num_up]-y_obs[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.factor*self.prob.num_up*((self.num_obs)*self.num_circles))
        ws_alpha = ws_alpha.reshape(self.num_batch, self.factor*self.prob.num_up*((self.num_obs)*self.num_circles))

        alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
        c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_temp = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_batch,  self.factor*self.prob.num_up*((self.num_obs)*self.num_circles)  )), d_temp   )

        ################# velocity terms

        wc_alpha_vx = xdot_guess
        ws_alpha_vy = ydot_guess
        alpha_v = jnp.unwrap(jnp.arctan2( ws_alpha_vy, wc_alpha_vx))
        
        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
        
        d_temp_v = c2_d_v/c1_d_v
        
        d_v = jnp.clip(d_temp_v, self.v_min, self.v_max )
        
        ################# acceleration terms

        wc_alpha_ax = xddot_guess
        ws_alpha_ay = yddot_guess
        alpha_a = jnp.unwrap(jnp.arctan2( ws_alpha_ay, wc_alpha_ax))
        
    
        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        d_temp_a = c2_d_a/c1_d_a
        d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max  )

        
        #########################################33
        res_ax_vec = xddot_guess-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot_guess-d_a*jnp.sin(alpha_a)
        
        res_vx_vec = xdot_guess-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot_guess-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

            
        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))

        
        lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T
        lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T
    
        return alpha_obs, d_obs,alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v
        
    @partial(jit, static_argnums=(0,))	
    def compute_x_guess(self, b_eq_x, b_eq_y, neural_output_batch):

        v_des_1 = neural_output_batch[:, 0]
        v_des_2 = neural_output_batch[:, 1]
        v_des_3 = neural_output_batch[:, 2]
        v_des_4 = neural_output_batch[:, 3]

        # y_des = neural_output_batch[:, 4]

        y_des_1 = neural_output_batch[:, 4]
        y_des_2 = neural_output_batch[:, 5]
        y_des_3 = neural_output_batch[:, 6]
        y_des_4 = neural_output_batch[:, 7]
        
        #############################
        A_vd_1 = self.Pddot_jax_1-self.k_p_v*self.Pdot_jax_1# - self.k_d_v*self.Pddot_jax_1
        b_vd_1 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_1)[:, jnp.newaxis]

        A_vd_2 = self.Pddot_jax_2-self.k_p_v*self.Pdot_jax_2 #- self.k_d_v*self.Pddot_jax_2
        b_vd_2 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_2)[:, jnp.newaxis]

        A_vd_3 = self.Pddot_jax_3-self.k_p_v*self.Pdot_jax_3 #- self.k_d_v*self.Pddot_jax_3
        b_vd_3 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_3)[:, jnp.newaxis]

        A_vd_4 = self.Pddot_jax_4-self.k_p_v*self.Pdot_jax_4 #- self.k_d_v*self.Pddot_jax_4
        b_vd_4 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_4)[:, jnp.newaxis]

        # A_pd = self.Pddot_jax-self.k_p*self.P_jax-self.k_d*self.Pdot_jax
        # b_pd = -self.k_p*jnp.ones((self.num_batch, self.num ))*(y_des)[:, jnp.newaxis]

        A_pd_1 = self.Pddot_jax_1-self.k_p*self.P_jax_1 #-self.k_d*self.Pdot_jax_1
        b_pd_1 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_1)[:, jnp.newaxis]
        
        A_pd_2 = self.Pddot_jax_2-self.k_p*self.P_jax_2 #-self.k_d*self.Pdot_jax_2
        b_pd_2 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_2)[:, jnp.newaxis]
            
        A_pd_3 = self.Pddot_jax_3-self.k_p*self.P_jax_3 #-self.k_d*self.Pdot_jax_3
        b_pd_3 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_3)[:, jnp.newaxis]
        
        A_pd_4 = self.Pddot_jax_4-self.k_p*self.P_jax_4 #-self.k_d*self.Pdot_jax_4
        b_pd_4 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_4)[:, jnp.newaxis]
        
        ##############################################

        cost_smoothness_x = self.weight_smoothness_x*jnp.identity(self.nvar)
        cost_smoothness_y = self.weight_smoothness_y*jnp.identity(self.nvar)

        # cost_smoothness_x = self.weight_smoothness_x*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        # cost_smoothness_y = self.weight_smoothness_y**jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        
        
        cost_x = cost_smoothness_x+self.rho_v*jnp.dot(A_vd_1.T, A_vd_1)+self.rho_v*jnp.dot(A_vd_2.T, A_vd_2)+self.rho_v*jnp.dot(A_vd_3.T, A_vd_3)+self.rho_v*jnp.dot(A_vd_4.T, A_vd_4)
        cost_y = cost_smoothness_y+self.rho_offset*jnp.dot(A_pd_1.T, A_pd_1)+self.rho_offset*jnp.dot(A_pd_2.T, A_pd_2)+self.rho_offset*jnp.dot(A_pd_3.T, A_pd_3)+self.rho_offset*jnp.dot(A_pd_4.T, A_pd_4)
        
        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -self.rho_v*jnp.dot(A_vd_1.T, b_vd_1.T).T-self.rho_v*jnp.dot(A_vd_2.T, b_vd_2.T).T-self.rho_v*jnp.dot(A_vd_3.T, b_vd_3.T).T-self.rho_v*jnp.dot(A_vd_4.T, b_vd_4.T).T
        lincost_y = -self.rho_offset*jnp.dot(A_pd_1.T, b_pd_1.T).T-self.rho_offset*jnp.dot(A_pd_2.T, b_pd_2.T).T-self.rho_offset*jnp.dot(A_pd_3.T, b_pd_3.T).T-self.rho_offset*jnp.dot(A_pd_4.T, b_pd_4.T).T 
    
        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        #######################3

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        c_x = primal_sol_x[0]
        c_y = primal_sol_y[0]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T


        return primal_sol_x, primal_sol_y

    @partial(jit, static_argnums=(0,))	
    def compute_x(self,  x_obs, y_obs, alpha_obs, d_obs,lamda_x, lamda_y, b_eq_x, b_eq_y, alpha_a, d_a, alpha_v, d_v,\
         c_x_bar, c_y_bar, s_lane, y_lane_lb, y_lane_ub,s_long,x_obs_long,y_obs_long):

        b_lane_lb = -self.gamma*y_lane_lb*jnp.ones(( self.num_batch, self.num-1  ))
        b_lane_ub = self.gamma*y_lane_ub*jnp.ones(( self.num_batch, self.num-1  ))
        b_lane_bound = jnp.hstack((b_lane_ub,b_lane_lb))
        b_lane_aug = b_lane_bound-s_lane
        
        # b_long = x_obs[:,1:] - (1-self.gamma_obs_long)*x_obs[:,0:self.num-1] - self.gamma_obs_long*self.d_separate*jnp.ones(( self.num_obs*self.num_circles, self.num-1  ))
        # b_long_bound = jnp.tile(b_long.flatten(), (self.num_batch,1)) # batch x ((num-1)*total_obs)
        # b_long_aug = b_long_bound - s_long

        b_long = x_obs_long[0,1:] - (1-self.gamma_obs_long)*x_obs_long[0,0:self.prob.num_up-1] \
                - self.gamma_obs_long*self.d_separate*jnp.ones((self.prob.num_up-1))
        
        
        b_long_bound = jnp.tile(b_long, (self.num_batch,1)) # batch x ((num-1))
        
        b_long_aug = b_long_bound - s_long

        b_ax_ineq = d_a*jnp.cos(alpha_a)
        b_ay_ineq = d_a*jnp.sin(alpha_a)

        b_vx_ineq = d_v*jnp.cos(alpha_v)
        b_vy_ineq = d_v*jnp.sin(alpha_v)

        temp_x_obs = d_obs*jnp.cos(alpha_obs)*self.a_obs
        b_obs_x = x_obs.reshape(self.prob.num_up*((self.num_obs)*self.num_circles))+temp_x_obs
            
        temp_y_obs = d_obs*jnp.sin(alpha_obs)*self.b_obs
        b_obs_y = y_obs.reshape(self.prob.num_up*((self.num_obs)*self.num_circles))+temp_y_obs

        cost_x = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc) \
        +self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel) \
            # + self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)\
            #   + self.rho_long*jnp.dot(self.A_barrier_long.T, self.A_barrier_long) 

        cost_y = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc) \
            +self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel)\
            +self.rho_lane*jnp.dot(self.A_lane_bound.T, self.A_lane_bound)\
                # +self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)\
        
        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -lamda_x-self.rho_projection*jnp.dot(self.A_projection.T, c_x_bar.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T \
            -self.rho_ineq*jnp.dot(self.A_vel.T, b_vx_ineq.T).T \
                # - self.rho_obs*jnp.dot(self.A_obs.T, b_obs_x.T).T\
                            #   - self.rho_long*jnp.dot( self.A_barrier_long.T, b_long_aug.T).T 

        lincost_y = -lamda_y-self.rho_projection*jnp.dot(self.A_projection.T, c_y_bar.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T\
            -self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T\
                -self.rho_lane*jnp.dot(self.A_lane_bound.T, b_lane_aug.T).T \
                    # -self.rho_obs*jnp.dot(self.A_obs.T, b_obs_y.T).T\

        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        s_lane = jnp.maximum( jnp.zeros(( self.num_batch, 2*(self.num-1) )),-jnp.dot(self.A_lane_bound, primal_sol_y.T).T+b_lane_bound )
        # s_long = jnp.maximum( jnp.zeros(( self.num_batch, self.num_circles*self.num_obs*(self.num-1) )),-jnp.dot(self.A_barrier_long, primal_sol_x.T).T+b_long_bound)
        s_long = jnp.maximum( jnp.zeros(( self.num_batch, (self.prob.num_up-1) )),-jnp.dot(self.A_barrier_long, primal_sol_x.T).T+b_long_bound)

        res_lane_vec = jnp.dot(self.A_lane_bound, primal_sol_y.T).T-b_lane_bound+s_lane
        res_long_vec = jnp.dot(self.A_barrier_long, primal_sol_x.T).T-b_long_bound+s_long

        return primal_sol_x, primal_sol_y, x, y, xdot, ydot, xddot, yddot, res_lane_vec, s_lane,res_long_vec,s_long
    

    @partial(jit, static_argnums=(0,))	
    def compute_x_baseline(self,  x_obs, y_obs, alpha_obs, d_obs,lamda_x, lamda_y, b_eq_x, b_eq_y, alpha_a, d_a, alpha_v, d_v,\
         c_x_bar, c_y_bar, s_lane, y_lane_lb, y_lane_ub,s_long,x_obs_long,y_obs_long):

        b_lane_lb = -self.gamma*y_lane_lb*jnp.ones(( self.num_batch, self.num-1  ))
        b_lane_ub = self.gamma*y_lane_ub*jnp.ones(( self.num_batch, self.num-1  ))
        b_lane_bound = jnp.hstack((b_lane_ub,b_lane_lb))
        b_lane_aug = b_lane_bound-s_lane
        
        # b_long = x_obs[:,1:] - (1-self.gamma_obs_long)*x_obs[:,0:self.num-1] - self.gamma_obs_long*self.d_separate*jnp.ones(( self.num_obs*self.num_circles, self.num-1  ))
        # b_long_bound = jnp.tile(b_long.flatten(), (self.num_batch,1)) # batch x ((num-1)*total_obs)
        # b_long_aug = b_long_bound - s_long

        b_long = x_obs_long[0,1:] - (1-self.gamma_obs_long)*x_obs_long[0,0:self.prob.num_up-1] \
                - self.gamma_obs_long*self.d_separate*jnp.ones((self.prob.num_up-1))
        
        
        b_long_bound = jnp.tile(b_long, (self.num_batch,1)) # batch x ((num-1))
        
        b_long_aug = b_long_bound - s_long

        b_ax_ineq = d_a*jnp.cos(alpha_a)
        b_ay_ineq = d_a*jnp.sin(alpha_a)

        b_vx_ineq = d_v*jnp.cos(alpha_v)
        b_vy_ineq = d_v*jnp.sin(alpha_v)

        temp_x_obs = d_obs*jnp.cos(alpha_obs)*self.a_obs
        b_obs_x = x_obs.reshape(self.factor*self.prob.num_up*((self.num_obs)*self.num_circles))+temp_x_obs
            
        temp_y_obs = d_obs*jnp.sin(alpha_obs)*self.b_obs
        b_obs_y = y_obs.reshape(self.factor*self.prob.num_up*((self.num_obs)*self.num_circles))+temp_y_obs

        cost_x = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc) \
        +self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel) \
            + self.rho_obs*jnp.dot(self.A_obs_baseline.T, self.A_obs_baseline)\
            #   + self.rho_long*jnp.dot(self.A_barrier_long.T, self.A_barrier_long) 

        cost_y = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc) \
            +self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel)\
            +self.rho_lane*jnp.dot(self.A_lane_bound.T, self.A_lane_bound)\
                +self.rho_obs*jnp.dot(self.A_obs_baseline.T, self.A_obs_baseline)\
        
        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -lamda_x-self.rho_projection*jnp.dot(self.A_projection.T, c_x_bar.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T \
            -self.rho_ineq*jnp.dot(self.A_vel.T, b_vx_ineq.T).T \
                - self.rho_obs*jnp.dot(self.A_obs_baseline.T, b_obs_x.T).T\
                            #   - self.rho_long*jnp.dot( self.A_barrier_long.T, b_long_aug.T).T 

        lincost_y = -lamda_y-self.rho_projection*jnp.dot(self.A_projection.T, c_y_bar.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T\
            -self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T\
                -self.rho_lane*jnp.dot(self.A_lane_bound.T, b_lane_aug.T).T \
                    -self.rho_obs*jnp.dot(self.A_obs_baseline.T, b_obs_y.T).T\

        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        s_lane = jnp.maximum( jnp.zeros(( self.num_batch, 2*(self.num-1) )),-jnp.dot(self.A_lane_bound, primal_sol_y.T).T+b_lane_bound )
        # s_long = jnp.maximum( jnp.zeros(( self.num_batch, self.num_circles*self.num_obs*(self.num-1) )),-jnp.dot(self.A_barrier_long, primal_sol_x.T).T+b_long_bound)
        s_long = jnp.maximum( jnp.zeros(( self.num_batch, (self.prob.num_up-1) )),-jnp.dot(self.A_barrier_long, primal_sol_x.T).T+b_long_bound)

        res_lane_vec = jnp.dot(self.A_lane_bound, primal_sol_y.T).T-b_lane_bound+s_lane
        res_long_vec = jnp.dot(self.A_barrier_long, primal_sol_x.T).T-b_long_bound+s_long

        return primal_sol_x, primal_sol_y, x, y, xdot, ydot, xddot, yddot, res_lane_vec, s_lane,res_long_vec,s_long
    
    @partial(jit, static_argnums=(0,))
    def comp_d_obs_prev(self,row):
        d_obs_single_batch = jnp.reshape(row,(((self.num_obs)*self.num_circles),self.prob.num_up))
        d_obs_single_batch_modified = jnp.hstack((jnp.ones((((self.num_obs)*self.num_circles),1)), d_obs_single_batch[:,0:self.prob.num_up-1]))
        return d_obs_single_batch_modified.flatten()

    @partial(jit, static_argnums=(0,))
    def comp_d_obs_prev_baseline(self,row):
        d_obs_single_batch = jnp.reshape(row,((self.factor*(self.num_obs)*self.num_circles),self.prob.num_up))
        d_obs_single_batch_modified = jnp.hstack((jnp.ones(((self.factor*(self.num_obs)*self.num_circles),1)), d_obs_single_batch[:,0:self.prob.num_up-1]))
        return d_obs_single_batch_modified.flatten()


    @partial(jit, static_argnums=(0,))	
    def compute_alph_d(self,x_obs, y_obs, x, y, xdot, ydot, xddot, yddot, lamda_x, lamda_y,
                     alpha_v_prev, alpha_a_prev, d_v_prev, d_a_prev, y_lane_lb, y_lane_ub,
                      res_lane_vec, v_max_aug,d_obs_prev,res_long_vec):
        
        wc_alpha_temp = (x[:,0:self.prob.num_up]-x_obs[:,jnp.newaxis])
        ws_alpha_temp = (y[:,0:self.prob.num_up]-y_obs[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        factor = 1
        wc_alpha = wc_alpha.reshape(self.num_batch, factor*self.prob.num_up*((self.num_obs)*self.num_circles))
        ws_alpha = ws_alpha.reshape(self.num_batch, factor*self.prob.num_up*((self.num_obs)*self.num_circles))

        alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
        c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_temp = c2_d/c1_d

        d_obs_prev = self.d_obs_vmap(d_obs_prev)
        d_obs = jnp.maximum(jnp.ones((self.num_batch,factor*self.prob.num_up*((self.num_obs)*self.num_circles)  )) + (1-self.gamma_obs)*(d_obs_prev-1),d_temp)

        ################# velocity terms

        wc_alpha_vx = xdot
        ws_alpha_vy = ydot
        alpha_v = jnp.unwrap(jnp.arctan2( ws_alpha_vy, wc_alpha_vx))
        
        kappa_bound_d_v = jnp.sqrt(d_a_prev*jnp.abs(jnp.sin(alpha_a_prev-alpha_v))/self.kappa_max)

        v_min_aug = jnp.maximum( kappa_bound_d_v, self.v_min*jnp.ones(( self.num_batch, self.num  ))  )

        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
        
        d_temp_v = c2_d_v/c1_d_v
        
        d_v = jnp.clip(d_temp_v, v_min_aug, v_max_aug )
        # d_v = jnp.clip(d_temp_v, self.v_min, self.v_max )
 
        ################# acceleration terms

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot
        alpha_a = jnp.unwrap(jnp.arctan2( ws_alpha_ay, wc_alpha_ax))
    
        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        kappa_bound_d_a = (self.kappa_max*d_v**2)/jnp.abs(jnp.sin(alpha_a-alpha_v))
        a_max_aug = jnp.minimum( self.a_max*jnp.ones((self.num_batch, self.num)), kappa_bound_d_a )

        d_temp_a = c2_d_a/c1_d_a
        d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), a_max_aug  )
        # d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max  )

        #######################################################

        #########################################33
        res_ax_vec = xddot-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot-d_a*jnp.sin(alpha_a)
        
        res_vx_vec = xdot-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

            
        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))

        res_norm_batch = jnp.linalg.norm(res_acc_vec, axis =1)+jnp.linalg.norm(res_vel_vec, axis =1)\
            +jnp.linalg.norm(res_lane_vec, axis = 1) \
                # + jnp.linalg.norm(res_obs_vec, axis =1) \
                # + jnp.linalg.norm(res_long_vec,axis=1) 

        # res_norm_batch =  jnp.linalg.norm(res_acc_vec, axis =1)+jnp.linalg.norm(res_vel_vec, axis =1)\
        #     +jnp.linalg.norm(res_lane_vec, axis = 1)

        lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T \
            #  -self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T\
                #  -self.rho_long*jnp.dot(self.A_barrier_long.T, res_long_vec.T).T 

        lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T \
            -self.rho_lane*jnp.dot(self.A_lane_bound.T, res_lane_vec.T).T \
                # -self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T
    
        return alpha_obs, d_obs,alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v
    
    @partial(jit, static_argnums=(0,))	
    def compute_alph_d_baseline(self,x_obs, y_obs, x, y, xdot, ydot, xddot, yddot, lamda_x, lamda_y,
                     alpha_v_prev, alpha_a_prev, d_v_prev, d_a_prev, y_lane_lb, y_lane_ub,
                      res_lane_vec, v_max_aug,d_obs_prev,res_long_vec):
        
        wc_alpha_temp = (x[:,0:self.prob.num_up]-x_obs[:,jnp.newaxis])
        ws_alpha_temp = (y[:,0:self.prob.num_up]-y_obs[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch,self.factor*self.prob.num_up*((self.num_obs)*self.num_circles))
        ws_alpha = ws_alpha.reshape(self.num_batch, self.factor*self.prob.num_up*((self.num_obs)*self.num_circles))

        alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
        c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_temp = c2_d/c1_d

        d_obs_prev = self.d_obs_vmap_baseline(d_obs_prev)
        d_obs = jnp.maximum(jnp.ones((self.num_batch,self.factor*self.prob.num_up*((self.num_obs)*self.num_circles)  )) + (1-self.gamma_obs)*(d_obs_prev-1),d_temp)

        ################# velocity terms

        wc_alpha_vx = xdot
        ws_alpha_vy = ydot
        alpha_v = jnp.unwrap(jnp.arctan2( ws_alpha_vy, wc_alpha_vx))
        
        kappa_bound_d_v = jnp.sqrt(d_a_prev*jnp.abs(jnp.sin(alpha_a_prev-alpha_v))/self.kappa_max)

        v_min_aug = jnp.maximum( kappa_bound_d_v, self.v_min*jnp.ones(( self.num_batch, self.num  ))  )

        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
        
        d_temp_v = c2_d_v/c1_d_v
        
        d_v = jnp.clip(d_temp_v, v_min_aug, v_max_aug )
        # d_v = jnp.clip(d_temp_v, self.v_min, self.v_max )
 
        ################# acceleration terms

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot
        alpha_a = jnp.unwrap(jnp.arctan2( ws_alpha_ay, wc_alpha_ax))
    
        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        kappa_bound_d_a = (self.kappa_max*d_v**2)/jnp.abs(jnp.sin(alpha_a-alpha_v))
        a_max_aug = jnp.minimum( self.a_max*jnp.ones((self.num_batch, self.num)), kappa_bound_d_a )

        d_temp_a = c2_d_a/c1_d_a
        d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), a_max_aug  )
        # d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max  )

        #######################################################

        #########################################33
        res_ax_vec = xddot-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot-d_a*jnp.sin(alpha_a)
        
        res_vx_vec = xdot-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))

        res_norm_batch = jnp.linalg.norm(res_acc_vec, axis =1)+ jnp.linalg.norm(res_vel_vec, axis =1)\
            + jnp.linalg.norm(res_lane_vec, axis = 1) \
                + jnp.linalg.norm(res_obs_vec, axis =1) \
                # + jnp.linalg.norm(res_long_vec,axis=1) 

        # res_norm_batch =  jnp.linalg.norm(res_acc_vec, axis =1)+jnp.linalg.norm(res_vel_vec, axis =1)\
        #     +jnp.linalg.norm(res_lane_vec, axis = 1)

        lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T \
             -self.rho_obs*jnp.dot(self.A_obs_baseline.T, res_x_obs_vec.T).T\
                #  -self.rho_long*jnp.dot(self.A_barrier_long.T, res_long_vec.T).T 

        lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T \
            -self.rho_lane*jnp.dot(self.A_lane_bound.T, res_lane_vec.T).T \
                -self.rho_obs*jnp.dot(self.A_obs_baseline.T, res_y_obs_vec.T).T
    
        return alpha_obs, d_obs,alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v

    @partial(jit, static_argnums=(0, ))	
    def compute_projection(self, x_obs,y_obs,b_eq_x, b_eq_y, initial_state, lamda_x_init, lamda_y_init, c_x_bar, c_y_bar, y_lane_lb,
                         y_lane_ub, s_lane, arc_vec, kappa, v_des,s_long,x_obs_long,y_obs_long):

        # x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        s_lane_init = s_lane
        s_long_init = s_long

        # b_eq_x, b_eq_y = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)

        # alpha_obs, d_obs = self.initial_alpha_d_obs(x_guess, y_guess, x_obs, y_obs)


        x_guess = jnp.dot(self.P_jax, c_x_bar.T).T 
        y_guess = jnp.dot(self.P_jax, c_y_bar.T).T 


        xdot_guess = jnp.dot(self.Pdot_jax, c_x_bar.T).T 
        ydot_guess = jnp.dot(self.Pdot_jax, c_y_bar.T).T 


        xddot_guess = jnp.dot(self.Pddot_jax, c_x_bar.T).T 
        yddot_guess = jnp.dot(self.Pddot_jax, c_y_bar.T).T 


        alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, lamda_x_init, lamda_y_init, alpha_v_init, d_v_init = self.initial_alpha_d_obs(x_obs,y_obs,
                                                                                            x_guess,y_guess, xdot_guess, ydot_guess, xddot_guess,\
                                                                                                 yddot_guess, lamda_x_init, lamda_y_init)
        def lax_projection(carry,idx):

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, \
                    res_norm_batch, alpha_v, d_v, curvature, steering, s_lane,s_long = carry
            
            alpha_a_prev = alpha_a
            alpha_v_prev = alpha_v 
            d_v_prev = d_v
            d_a_prev = d_a
            d_obs_prev = d_obs

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_lane_vec, s_lane,res_long_vec,s_long = self.compute_x(x_obs, y_obs, alpha_obs, d_obs,\
                                                                                            lamda_x, lamda_y, b_eq_x, b_eq_y, \
                                                                                            alpha_a, d_a, alpha_v, d_v, \
                                                                                            c_x_bar, c_y_bar, s_lane, y_lane_lb, y_lane_ub,\
                                                                                            s_long,x_obs_long,y_obs_long)
            
            kappa_interp = self.jax_interp(jnp.clip(x, jnp.zeros((self.num_batch, self.num)), arc_vec[-1]*jnp.ones((self.num_batch, self.num)) ), arc_vec, kappa)

            v_max_aug = jnp.minimum(jnp.sqrt(self.a_centr/jnp.abs(kappa_interp)), self.v_max*jnp.ones((self.num_batch, self.num  ))    )

            alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v = self.compute_alph_d(x_obs, y_obs, x, y, xdot, ydot, xddot, yddot, lamda_x, 
                                                                                                                lamda_y, alpha_v_prev, alpha_a_prev, d_v_prev, d_a_prev,
                                                                                                                y_lane_lb, y_lane_ub, res_lane_vec, v_max_aug,d_obs_prev,res_long_vec)

            curvature_frenet = d_a*jnp.sin(alpha_a-alpha_v)/(d_v**2)
            steering = jnp.arctan((curvature_frenet+kappa_interp*jnp.cos(alpha_v)/(1-y*kappa_interp) )*self.wheel_base)

            return (c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, 
                    res_norm_batch, alpha_v, d_v, curvature, steering, s_lane,s_long )\
                    ,kappa_interp

        carry_init = (jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.num)), 
        jnp.zeros((self.num_batch,self.num)),jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)),
            jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num))
        , alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, lamda_x_init, lamda_y_init, jnp.zeros((self.num_batch)), 
        alpha_v_init, d_v_init, jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)), s_lane_init,s_long_init)

        carry_final,result = lax.scan(lax_projection,carry_init,jnp.arange(self.maxiter))
        c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, \
        res_norm_batch, alpha_v, d_v, curvature, steering, s_lane,s_long = carry_final

        kappa_interp= result[-1]

        return 	c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch, steering, s_lane,kappa_interp,s_long

    @partial(jit, static_argnums=(0, ))	
    def compute_projection_baseline(self, x_obs,y_obs,b_eq_x, b_eq_y, initial_state, lamda_x_init, lamda_y_init, c_x_bar, c_y_bar, y_lane_lb,
                         y_lane_ub, s_lane, arc_vec, kappa, v_des,s_long,x_obs_long,y_obs_long):

        # x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        s_lane_init = s_lane
        s_long_init = s_long

        # b_eq_x, b_eq_y = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)

        # alpha_obs, d_obs = self.initial_alpha_d_obs(x_guess, y_guess, x_obs, y_obs)


        x_guess = jnp.dot(self.P_jax, c_x_bar.T).T 
        y_guess = jnp.dot(self.P_jax, c_y_bar.T).T 


        xdot_guess = jnp.dot(self.Pdot_jax, c_x_bar.T).T 
        ydot_guess = jnp.dot(self.Pdot_jax, c_y_bar.T).T 


        xddot_guess = jnp.dot(self.Pddot_jax, c_x_bar.T).T 
        yddot_guess = jnp.dot(self.Pddot_jax, c_y_bar.T).T 


        alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, lamda_x_init, lamda_y_init, alpha_v_init, d_v_init = self.initial_alpha_d_obs_baseline(x_obs,y_obs,
                                                                                            x_guess,y_guess, xdot_guess, ydot_guess, xddot_guess,\
                                                                                                 yddot_guess, lamda_x_init, lamda_y_init)
        def lax_projection(carry,idx):

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, \
                    res_norm_batch, alpha_v, d_v, curvature, steering, s_lane,s_long = carry
            
            alpha_a_prev = alpha_a
            alpha_v_prev = alpha_v 
            d_v_prev = d_v
            d_a_prev = d_a
            d_obs_prev = d_obs

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_lane_vec, s_lane,res_long_vec,s_long = self.compute_x_baseline(x_obs, y_obs, alpha_obs, d_obs,\
                                                                                            lamda_x, lamda_y, b_eq_x, b_eq_y, \
                                                                                            alpha_a, d_a, alpha_v, d_v, \
                                                                                            c_x_bar, c_y_bar, s_lane, y_lane_lb, y_lane_ub,\
                                                                                            s_long,x_obs_long,y_obs_long)
            
            kappa_interp = self.jax_interp(jnp.clip(x, jnp.zeros((self.num_batch, self.num)), arc_vec[-1]*jnp.ones((self.num_batch, self.num)) ), arc_vec, kappa)

            v_max_aug = jnp.minimum(jnp.sqrt(self.a_centr/jnp.abs(kappa_interp)), self.v_max*jnp.ones((self.num_batch, self.num  ))    )

            alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v = self.compute_alph_d_baseline(x_obs, y_obs, x, y, xdot, ydot, xddot, yddot, lamda_x, 
                                                                                                                lamda_y, alpha_v_prev, alpha_a_prev, d_v_prev, d_a_prev,
                                                                                                                y_lane_lb, y_lane_ub, res_lane_vec, v_max_aug,d_obs_prev,res_long_vec)

            curvature_frenet = d_a*jnp.sin(alpha_a-alpha_v)/(d_v**2)
            steering = jnp.arctan((curvature_frenet+kappa_interp*jnp.cos(alpha_v)/(1-y*kappa_interp) )*self.wheel_base)

            return (c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, 
                    res_norm_batch, alpha_v, d_v, curvature, steering, s_lane,s_long )\
                    ,kappa_interp

        carry_init = (jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.num)), 
        jnp.zeros((self.num_batch,self.num)),jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)),
            jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num))
        , alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, lamda_x_init, lamda_y_init, jnp.zeros((self.num_batch)), 
        alpha_v_init, d_v_init, jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)), s_lane_init,s_long_init)

        carry_final,result = lax.scan(lax_projection,carry_init,jnp.arange(self.maxiter_baseline))
        c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, \
        res_norm_batch, alpha_v, d_v, curvature, steering, s_lane,s_long = carry_final

        kappa_interp= result[-1]

        return 	c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch, steering, s_lane,kappa_interp,s_long

    @partial(jit, static_argnums=(0, ))
    def compute_cost(self,x_obs_total,y_obs_total,beta, x_obs,y_obs,x_ellite_projection,ref_x, ref_y,global_x_ego, global_y_ego,y_ellite_projection, res_ellite_projection, \
        xdot_ellite_projection, ydot_ellite_projection, xddot_ellite_projection, yddot_ellite_projection, \
            y_lane_lb, y_lane_ub, v_des, steering_ellite_projection,kappa_interp):
                
        cost_steering = jnp.linalg.norm(steering_ellite_projection, axis = 1)
        steering_vel = jnp.diff(steering_ellite_projection, axis = 1)
        cost_steering_vel = jnp.linalg.norm(steering_vel, axis = 1)

        heading_angle = jnp.arctan2(ydot_ellite_projection, xdot_ellite_projection)
        heading_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_projection, self.num  )), jnp.abs(heading_angle)-10*jnp.pi/180   ), axis = 1)
        centerline_cost = jnp.linalg.norm(y_ellite_projection, axis = 1)
        
        v = jnp.sqrt(xdot_ellite_projection**2+ydot_ellite_projection**2)
        
        centr_acc = jnp.abs(xdot_ellite_projection**2*(kappa_interp))
        centr_acc_cost = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_projection, self.num  )), centr_acc-self.a_centr   ), axis = 1)

        d_tracking = jnp.sqrt((global_x_ego-ref_x)**2 + (global_y_ego-ref_y)**2)
        cost_tracking = jnp.linalg.norm(jnp.maximum( jnp.zeros((self.ellite_num_projection,self.num)),\
                                                     d_tracking - self.margin*jnp.ones((self.ellite_num_projection,self.num)) ),axis=1)
        
        cost_mmd = self.prob.compute_f_bar_vmap(x_ellite_projection[:,0:self.prob.num_up],y_ellite_projection[:,0:self.prob.num_up],x_obs,y_obs) # num_ellite_projection x num_reduced x num_up
        mmd_total = self.prob.compute_mmd_vmap(beta,cost_mmd).reshape((self.ellite_num_projection)) # 150x1

        # cost_batch = 10*mmd_total/jnp.max(mmd_total) \
        #     + 0.001*jnp.linalg.norm(v-v_des, axis = 1)\
        #      + 0.1*res_ellite_projection\
        #         + 0.1*cost_steering + 0.1*cost_steering_vel

        cost_batch = 100*mmd_total \
             + 0.0001*jnp.linalg.norm(v-v_des, axis = 1)\
        + res_ellite_projection\
                # + 0.1*cost_steering + 0.1*cost_steering_vel

        return cost_batch,mmd_total
    
    @partial(jit, static_argnums=(0, ))
    def compute_cost_baseline(self,x_obs_total,y_obs_total,beta, x_obs,y_obs,x_ellite_projection,ref_x, ref_y,global_x_ego, global_y_ego,y_ellite_projection, res_ellite_projection, \
        xdot_ellite_projection, ydot_ellite_projection, xddot_ellite_projection, yddot_ellite_projection, \
            y_lane_lb, y_lane_ub, v_des, steering_ellite_projection,kappa_interp):
                
        cost_steering = jnp.linalg.norm(steering_ellite_projection, axis = 1)
        steering_vel = jnp.diff(steering_ellite_projection, axis = 1)
        cost_steering_vel = jnp.linalg.norm(steering_vel, axis = 1)

        heading_angle = jnp.arctan2(ydot_ellite_projection, xdot_ellite_projection)
        heading_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_projection, self.num  )), jnp.abs(heading_angle)-10*jnp.pi/180   ), axis = 1)
        centerline_cost = jnp.linalg.norm(y_ellite_projection, axis = 1)
        
        v = jnp.sqrt(xdot_ellite_projection**2+ydot_ellite_projection**2)
        
        centr_acc = jnp.abs(xdot_ellite_projection**2*(kappa_interp))
        centr_acc_cost = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_projection, self.num  )), centr_acc-self.a_centr   ), axis = 1)

        d_tracking = jnp.sqrt((global_x_ego-ref_x)**2 + (global_y_ego-ref_y)**2)
        cost_tracking = jnp.linalg.norm(jnp.maximum( jnp.zeros((self.ellite_num_projection,self.num)),\
                                                     d_tracking - self.margin*jnp.ones((self.ellite_num_projection,self.num)) ),axis=1)
        
        # f = self.prob.compute_f_vmap(x_ellite_projection[:,0:self.prob.num_up],y_ellite_projection[:,0:self.prob.num_up],x_obs,y_obs)
        # saa_total = self.prob.compute_saa_vmap(f).reshape((self.ellite_num_projection))

        # cost_batch = 10*saa_total/jnp.max(saa_total) \
        #       + 0.001*jnp.linalg.norm(v-v_des, axis = 1)\
        #      + 0.1*res_ellite_projection\
        #         + 0.1*cost_steering + 0.1*cost_steering_vel

        cost_batch = res_ellite_projection \
            + 0.0001*jnp.linalg.norm(v-v_des, axis = 1)\
        
                # + 0.1*cost_steering + 0.1*cost_steering_vel

        return cost_batch#,saa_total

    @partial(jit, static_argnums=(0, ))
    def compute_ellite_samples(self, cost_batch, neural_output_projection):

        idx_ellite = jnp.argsort(cost_batch)

        neural_output_ellite = neural_output_projection[idx_ellite[0:self.ellite_num]]

        return neural_output_ellite, idx_ellite
    
    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):
        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)
        prods = d * jnp.outer(term_1,term_2)
        # prods = d*jnp.outer(diffs,diffs)
        return prods	
    
    @partial(jit, static_argnums=(0, ))
    def interp(self,x,xp,fp):
        return self.jax_interp(x,xp,fp)     
    
    @partial(jit, static_argnums=(0, ))
    def compute_shifted_samples(self, key, neural_output_ellite, cost_batch, idx_ellite, mean_param_prev, cov_param_prev, y_lane_ub, y_lane_lb):
        
        # temp = jnp.maximum(1, (15.5/((i+1))  ))

        # mean_param = jnp.mean(neural_output_ellite, axis = 0)
        # cov_param = jnp.cov(neural_output_ellite.T)+0.01*jnp.identity(8)

        # c_mean = (1-self.alpha_mean)*c_mean_prev+self.alpha_mean*c_mean
        # c_cov = (1-self.alpha_cov)*c_cov_prev+self.alpha_cov*c_cov

        cost_batch_temp = cost_batch[idx_ellite[0:self.ellite_num]]

        ######################################

        w = cost_batch_temp
        w_min = jnp.min(cost_batch_temp)
        w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
        sum_w = jnp.sum(w, axis = 0)
        mean_param = (1-self.alpha_mean)*mean_param_prev + self.alpha_mean*(jnp.sum( (neural_output_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
        diffs = (neural_output_ellite - mean_param)
        prod_result = self.vec_product(diffs, w)
        cov_param = (1-self.alpha_cov)*cov_param_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.01*jnp.identity(8)

        param_samples = jax.random.multivariate_normal(key, mean_param, cov_param, (self.num_batch-self.ellite_num, ))

        v_des_1 = param_samples[:, 0]
        v_des_2 = param_samples[:, 1]
        v_des_3 = param_samples[:, 2]
        v_des_4 = param_samples[:, 3]
        
        # y_des = param_samples[:, 4]

        y_des_1 = param_samples[:, 4]
        y_des_2 = param_samples[:, 5]
        y_des_3 = param_samples[:, 6]
        y_des_4 = param_samples[:, 7]
        
        # y_des = jnp.clip(y_des, y_lane_lb*jnp.ones(self.num_batch-self.ellite_num), y_lane_ub*jnp.ones(self.num_batch-self.ellite_num)   )
        
        v_des_1 = jnp.clip(v_des_1, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        v_des_2 = jnp.clip(v_des_2, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        
        v_des_3 = jnp.clip(v_des_3, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        
        v_des_4 = jnp.clip(v_des_4, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        
        neural_output_shift = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4   )).T

        neural_output_batch = jnp.vstack(( neural_output_ellite, neural_output_shift  ))
        neural_output_batch = neural_output_batch.at[-10:,:].set(jnp.zeros((10,8)))

        return mean_param, cov_param, neural_output_batch, cost_batch_temp

    @partial(jit, static_argnums=(0, ))	
    def compute_cem(self,x_obs_val,y_obs_val,x_obs_total,y_obs_total,beta,x_obs_mean,y_obs_mean,x_obs,y_obs,initial_state, lamda_x_init, lamda_y_init,
                    y_lane_lb, y_lane_ub, v_des, 
                    s_lane_init, mean_param_init, cov_param_init, \
                    x_path,y_path,arc_vec,Fx_dot,Fy_dot,\
                    kappa,s_long_init,x_obs_long,y_obs_long):
        
        res_init = jnp.zeros(self.maxiter_cem)
        res_2_init = jnp.zeros(self.maxiter_cem)
        res_3_init = jnp.zeros(self.maxiter_cem)
        res_4_init = jnp.zeros((self.maxiter_cem,self.prob.num_batch*self.num_circles,self.prob.num_up))
        res_5_init = jnp.zeros((self.maxiter_cem,self.num_validation*self.num_circles,self.prob.num_up))

        neural_output_batch_init = self.sampling_param(y_lane_lb, y_lane_ub, mean_param_init, cov_param_init)

        y_des = neural_output_batch_init[:, 4]

        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        b_eq_x, b_eq_y = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init, y_des)

        def lax_cem(carry,idx):

            res,res_2,res_3,res_4,res_5,lamda_x, lamda_y, s_lane, neural_output_batch,neural_output_ellite,mean_param,cov_param,s_long = carry

            c_x_bar, c_y_bar = self.compute_x_guess(b_eq_x, b_eq_y, neural_output_batch)

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch, steering, s_lane,kappa_interp,s_long = self.compute_projection(x_obs_mean,y_obs_mean,\
                b_eq_x, b_eq_y, initial_state, lamda_x, lamda_y, c_x_bar, c_y_bar,\
                y_lane_lb, y_lane_ub, s_lane, arc_vec, kappa, v_des,s_long,x_obs_long,y_obs_long)

            
            idx_ellite_projection = jnp.argsort(res_norm_batch)

            x_ellite_projection = x[idx_ellite_projection[0:self.ellite_num_projection]]
            y_ellite_projection = y[idx_ellite_projection[0:self.ellite_num_projection]]

            xdot_ellite_projection = xdot[idx_ellite_projection[0:self.ellite_num_projection]]
            ydot_ellite_projection = ydot[idx_ellite_projection[0:self.ellite_num_projection]]
            
            xddot_ellite_projection = xddot[idx_ellite_projection[0:self.ellite_num_projection]]
            yddot_ellite_projection = yddot[idx_ellite_projection[0:self.ellite_num_projection]]

            steering_ellite_projection = steering[idx_ellite_projection[0:self.ellite_num_projection]]
            kappa_ellite_projection = kappa_interp[idx_ellite_projection[0:self.ellite_num_projection]]

            c_x_ellite_projection = c_x[idx_ellite_projection[0:self.ellite_num_projection]]
            c_y_ellite_projection = c_y[idx_ellite_projection[0:self.ellite_num_projection]]
            res_ellite_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            neural_output_projection = neural_output_batch[idx_ellite_projection[0:self.ellite_num_projection]]
            
            ref_x = self.interp_vmap(x_ellite_projection, arc_vec, x_path )
            ref_y = self.interp_vmap(x_ellite_projection, arc_vec, y_path )
            dx_by_ds = self.interp_vmap(x_ellite_projection, arc_vec, Fx_dot )
            dy_by_ds = self.interp_vmap(x_ellite_projection, arc_vec, Fy_dot )

            global_x_ego, global_y_ego, psi_global	= self.frenet_to_global_vmap(y_ellite_projection, ref_x, ref_y, dx_by_ds, dy_by_ds)

            cost_batch,mmd_total= self.compute_cost(x_obs_total,y_obs_total,beta,x_obs,y_obs,x_ellite_projection,ref_x, ref_y,global_x_ego, global_y_ego,y_ellite_projection, res_ellite_projection, 
                                            xdot_ellite_projection, ydot_ellite_projection, xddot_ellite_projection, yddot_ellite_projection, 
                                            y_lane_lb, y_lane_ub, v_des, steering_ellite_projection,kappa_ellite_projection)

            ####################################################################################

            neural_output_ellite, idx_ellite = self.compute_ellite_samples(cost_batch, neural_output_projection)
            
            key, subkey = random.split(self.key)

            mean_param, cov_param, neural_output_batch, cost_batch_temp = self.compute_shifted_samples(key, neural_output_ellite, cost_batch, idx_ellite, mean_param, cov_param, y_lane_ub, y_lane_lb)
        
            idx_min = jnp.argmin(cost_batch_temp)
            c_x_ellite_projection = c_x_ellite_projection[idx_min]
            c_y_ellite_projection = c_y_ellite_projection[idx_min]
            steering_ellite_projection = steering_ellite_projection[idx_min]

            res = res.at[idx].set(jnp.min(cost_batch_temp))
            res_2 = res_2.at[idx].set(res_ellite_projection[idx_min])
            res_3 = res_3.at[idx].set(mmd_total[idx_min])

            temp = self.prob.compute_f_bar_temp(x_ellite_projection[idx_min][0:self.prob.num_up],y_ellite_projection[idx_min][0:self.prob.num_up],
                                        x_obs_total,y_obs_total).reshape((self.prob.num_batch*self.num_circles,self.prob.num_up))

            temp1 = self.prob.compute_f_bar_temp_val(x_ellite_projection[idx_min][0:self.prob.num_up],y_ellite_projection[idx_min][0:self.prob.num_up],
                                        x_obs_val,y_obs_val).reshape((self.num_validation*self.num_circles,self.prob.num_up))

            res_4 = res_4.at[idx,:,:].set(temp)
            res_5 = res_5.at[idx,:,:].set(temp1)

            return (res,res_2,res_3,res_4,res_5,lamda_x, lamda_y,s_lane, neural_output_batch,neural_output_ellite,mean_param,cov_param,s_long),\
                (c_x_ellite_projection,c_y_ellite_projection,steering_ellite_projection,c_x_bar,c_y_bar)

          
        carry_init = (res_init,res_2_init,res_3_init,res_4_init,res_5_init,lamda_x_init, lamda_y_init,s_lane_init, \
            neural_output_batch_init,jnp.zeros((self.ellite_num,8)),mean_param_init, cov_param_init,s_long_init)

        carry_final,result = lax.scan(lax_cem,carry_init,jnp.arange(self.maxiter_cem))

        res,res_2,res_3,res_4,res_5,lamda_x, lamda_y, s_lane, neural_output_batch_final,neural_output_ellite,mean_param, cov_param ,s_long= carry_final

        c_x_best = result[0][-1]
        c_y_best = result[1][-1]
       
        xdot_best = jnp.dot(self.Pdot_jax, c_x_best)
        ydot_best = jnp.dot(self.Pdot_jax, c_y_best)

        v_best = jnp.sqrt(xdot_best**2+ydot_best**2)
        steering_best = result[2][-1]
        
        
        x_best = jnp.dot(self.P_jax, c_x_best)
        y_best = jnp.dot(self.P_jax, c_y_best)
            
        c_x_bar = result[3][-1]
        c_y_bar = result[4][-1]

        x_guess = jnp.dot(self.P_jax, c_x_bar.T).T 
        y_guess = jnp.dot(self.P_jax, c_y_bar.T).T 

        return 	c_x_best, c_y_best,res,res_2,res_3,res_4,res_5

    @partial(jit, static_argnums=(0, ))	
    def compute_cem_baseline(self,x_obs_val,y_obs_val,x_obs_total,y_obs_total,beta,x_obs_mean,y_obs_mean,x_obs,y_obs,initial_state, lamda_x_init, lamda_y_init,
                    y_lane_lb, y_lane_ub, v_des, 
                    s_lane_init, mean_param_init, cov_param_init, \
                    x_path,y_path,arc_vec,Fx_dot,Fy_dot,\
                    kappa,s_long_init,x_obs_long,y_obs_long):
        
        res_init = jnp.zeros(self.maxiter_cem_baseline)
        res_2_init = jnp.zeros(self.maxiter_cem_baseline)
        res_3_init = jnp.zeros(self.maxiter_cem_baseline)
        res_4_init = jnp.zeros((self.maxiter_cem_baseline,self.prob.num_batch*self.num_circles,self.prob.num_up))
        res_5_init = jnp.zeros((self.maxiter_cem,self.num_validation*self.num_circles,self.prob.num_up))

        neural_output_batch_init = self.sampling_param(y_lane_lb, y_lane_ub, mean_param_init, cov_param_init)
       
        y_des = neural_output_batch_init[:, 4]

        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

        b_eq_x, b_eq_y = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init, y_des)

        def lax_cem(carry,idx):

            res,res_2,res_3,res_4,res_5,lamda_x, lamda_y, s_lane, neural_output_batch,neural_output_ellite,mean_param,cov_param,s_long = carry

            c_x_bar, c_y_bar = self.compute_x_guess(b_eq_x, b_eq_y, neural_output_batch)
            
            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch, steering, s_lane,kappa_interp,s_long = self.compute_projection_baseline(x_obs,y_obs,\
                b_eq_x, b_eq_y, initial_state, lamda_x, lamda_y, c_x_bar, c_y_bar,\
                y_lane_lb, y_lane_ub, s_lane, arc_vec, kappa, v_des,s_long,x_obs_long,y_obs_long)

            
            idx_ellite_projection = jnp.argsort(res_norm_batch)

            x_ellite_projection = x[idx_ellite_projection[0:self.ellite_num_projection]]
            y_ellite_projection = y[idx_ellite_projection[0:self.ellite_num_projection]]

            xdot_ellite_projection = xdot[idx_ellite_projection[0:self.ellite_num_projection]]
            ydot_ellite_projection = ydot[idx_ellite_projection[0:self.ellite_num_projection]]
            
            xddot_ellite_projection = xddot[idx_ellite_projection[0:self.ellite_num_projection]]
            yddot_ellite_projection = yddot[idx_ellite_projection[0:self.ellite_num_projection]]

            steering_ellite_projection = steering[idx_ellite_projection[0:self.ellite_num_projection]]
            kappa_ellite_projection = kappa_interp[idx_ellite_projection[0:self.ellite_num_projection]]

            c_x_ellite_projection = c_x[idx_ellite_projection[0:self.ellite_num_projection]]
            c_y_ellite_projection = c_y[idx_ellite_projection[0:self.ellite_num_projection]]
            res_ellite_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            neural_output_projection = neural_output_batch[idx_ellite_projection[0:self.ellite_num_projection]]
            
            ref_x = self.interp_vmap(x_ellite_projection, arc_vec, x_path )
            ref_y = self.interp_vmap(x_ellite_projection, arc_vec, y_path )
            dx_by_ds = self.interp_vmap(x_ellite_projection, arc_vec, Fx_dot )
            dy_by_ds = self.interp_vmap(x_ellite_projection, arc_vec, Fy_dot )

            global_x_ego, global_y_ego, psi_global	= self.frenet_to_global_vmap(y_ellite_projection, ref_x, ref_y, dx_by_ds, dy_by_ds)

            cost_batch= self.compute_cost_baseline(x_obs_total,y_obs_total,beta,x_obs,y_obs,x_ellite_projection,ref_x, ref_y,global_x_ego, global_y_ego,y_ellite_projection, res_ellite_projection, 
                                            xdot_ellite_projection, ydot_ellite_projection, xddot_ellite_projection, yddot_ellite_projection, 
                                            y_lane_lb, y_lane_ub, v_des, steering_ellite_projection,kappa_ellite_projection)

            ####################################################################################

            neural_output_ellite, idx_ellite = self.compute_ellite_samples(cost_batch, neural_output_projection)
            
            key, subkey = random.split(self.key)

            mean_param, cov_param, neural_output_batch, cost_batch_temp = self.compute_shifted_samples(key, neural_output_ellite, cost_batch, idx_ellite, mean_param, cov_param, y_lane_ub, y_lane_lb)
        
            idx_min = jnp.argmin(cost_batch_temp)
            c_x_ellite_projection = c_x_ellite_projection[idx_min]
            c_y_ellite_projection = c_y_ellite_projection[idx_min]
            steering_ellite_projection = steering_ellite_projection[idx_min]

            res = res.at[idx].set(jnp.min(cost_batch_temp))
            res_2 = res_2.at[idx].set(res_ellite_projection[idx_min])
            
            temp = self.prob.compute_f_bar_temp(x_ellite_projection[idx_min][0:self.prob.num_up],y_ellite_projection[idx_min][0:self.prob.num_up],
                                        x_obs_total,y_obs_total).reshape((self.prob.num_batch*self.num_circles,self.prob.num_up))
            temp1 = self.prob.compute_f_bar_temp_val(x_ellite_projection[idx_min][0:self.prob.num_up],y_ellite_projection[idx_min][0:self.prob.num_up],
                                        x_obs_val,y_obs_val).reshape((self.num_validation*self.num_circles,self.prob.num_up))

            res_4 = res_4.at[idx,:,:].set(temp)
            res_5 = res_5.at[idx,:,:].set(temp1)

            return (res,res_2,res_3,res_4,res_5,lamda_x, lamda_y,s_lane, neural_output_batch,neural_output_ellite,mean_param,cov_param,s_long),\
                (c_x_ellite_projection,c_y_ellite_projection,steering_ellite_projection,c_x_bar,c_y_bar)

          
        carry_init = (res_init,res_2_init,res_3_init,res_4_init,res_5_init,lamda_x_init, lamda_y_init,s_lane_init, \
            neural_output_batch_init,jnp.zeros((self.ellite_num,8)),mean_param_init, cov_param_init,s_long_init)

        carry_final,result = lax.scan(lax_cem,carry_init,jnp.arange(self.maxiter_cem_baseline))

        res,res_2,res_3,res_4,res_5,lamda_x, lamda_y, s_lane, neural_output_batch_final,neural_output_ellite,mean_param, cov_param ,s_long= carry_final

        c_x_best = result[0][-1]
        c_y_best = result[1][-1]
       
        xdot_best = jnp.dot(self.Pdot_jax, c_x_best)
        ydot_best = jnp.dot(self.Pdot_jax, c_y_best)

        v_best = jnp.sqrt(xdot_best**2+ydot_best**2)
        steering_best = result[2][-1]
        
        x_best = jnp.dot(self.P_jax, c_x_best)
        y_best = jnp.dot(self.P_jax, c_y_best)
            
        c_x_bar = result[3][-1]
        c_y_bar = result[4][-1]

        x_guess = jnp.dot(self.P_jax, c_x_bar.T).T 
        y_guess = jnp.dot(self.P_jax, c_y_bar.T).T 
        
        return 	c_x_best, c_y_best, res,res_2,res_3,res_4,res_5

    @partial(jit, static_argnums=(0, ))	
    def frenet_to_global(self, y_frenet, ref_x, ref_y, dx_by_ds, dy_by_ds):

        # global_x = np.zeros(len(ref_x))
        # global_y = np.zeros(len(ref_x))

        # dx_by_ds = np.gradient(ref_x)
        # dy_by_ds = np.gradient(ref_y)
      
        normal_x = -1*dy_by_ds
        normal_y = dx_by_ds

        norm_vec = jnp.sqrt(normal_x**2 + normal_y**2)
        normal_unit_x = (1/norm_vec)*normal_x
        normal_unit_y = (1/norm_vec)*normal_y

        global_x = ref_x + y_frenet*normal_unit_x
        global_y = ref_y + y_frenet*normal_unit_y

        psi_global = jnp.unwrap(jnp.arctan2(jnp.diff(global_y),jnp.diff(global_x)))

        return global_x, global_y, psi_global	    

    def compute_mpc_command(self,x_obs_val,y_obs_val,num_samples_baseline,x_obs_baseline,y_obs_baseline,x_obs_total,y_obs_total,x_obs_mean,y_obs_mean,beta,x_obs,y_obs,x_waypoints_shifted, y_waypoints_shifted,global_initial_state, lamda_x, lamda_y, 
                            v_des, y_lane_lb, y_lane_ub, mean_param, cov_param, s_lane,s_long):

        threshold = 0.1

        x_path, y_path = self.custom_path_smoothing(x_waypoints_shifted, y_waypoints_shifted, threshold)
        Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length = self.compute_path_parameters(x_path, y_path)
        
        num_p = 50000
        arc_vec_temp = jnp.linspace(0,arc_vec[-1],num_p)
       
        x_path_up = self.interp(arc_vec_temp,arc_vec,x_path)
        y_path_up = self.interp(arc_vec_temp,arc_vec,y_path)
        Fx_dot_up = self.interp(arc_vec_temp,arc_vec,Fx_dot)
        Fy_dot_up = self.interp(arc_vec_temp,arc_vec,Fy_dot)
        kappa_up = self.interp(arc_vec_temp,arc_vec,kappa)

        x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init = self.global_to_frenet(x_path_up, y_path_up,
                                                                                             global_initial_state, arc_vec_temp,
                                                                                              Fx_dot_up, Fy_dot_up, kappa_up )

        initial_state = jnp.asarray(np.hstack(( x_init, y_init, vx_init, vy_init, ax_init, ay_init )) )
        initial_state_frenet = jnp.asarray(np.hstack(( x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init )) )
        
        ###########################  obstacle trajectories global to frenet conversion 

        xdot_obs_mean = jnp.diff(x_obs_mean)/self.prob.t_up
        xdot_obs_mean = jnp.hstack((xdot_obs_mean,xdot_obs_mean[-1]))

        ydot_obs_mean = jnp.diff(y_obs_mean)/self.prob.t_up
        ydot_obs_mean = jnp.hstack((ydot_obs_mean,ydot_obs_mean[-1]))

        v_obs_mean = jnp.sqrt(xdot_obs_mean**2+ydot_obs_mean**2)
        psi_obs_mean = jnp.arctan2(jnp.diff(y_obs_mean),jnp.diff(x_obs_mean))
        psi_obs_mean = jnp.hstack((psi_obs_mean,psi_obs_mean[-1]))

        xdot_obs_total = jnp.diff(x_obs_total,axis=1)/self.prob.t_up
        xdot_obs_total = jnp.hstack((xdot_obs_total,xdot_obs_total[:,-1].reshape(self.prob.num_batch,1)))

        ydot_obs_total = jnp.diff(y_obs_total,axis=1)/self.prob.t_up
        ydot_obs_total = jnp.hstack((ydot_obs_total,ydot_obs_total[:,-1].reshape(self.prob.num_batch,1)))

        v_obs_total = jnp.sqrt(xdot_obs_total**2+ydot_obs_total**2)
        psi_obs_total = jnp.arctan2(jnp.diff(y_obs_total,axis=1),jnp.diff(x_obs_total,axis=1))
        psi_obs_total = jnp.hstack((psi_obs_total,psi_obs_total[:,-1].reshape(self.prob.num_batch,1)))

        xdot_obs = jnp.diff(x_obs,axis=1)/self.prob.t_up
        xdot_obs = jnp.hstack((xdot_obs,xdot_obs[:,-1].reshape(self.prob.num_reduced,1)))

        ydot_obs = jnp.diff(y_obs,axis=1)/self.prob.t_up
        ydot_obs = jnp.hstack((ydot_obs,ydot_obs[:,-1].reshape(self.prob.num_reduced,1)))

        v_obs = jnp.sqrt(xdot_obs**2+ydot_obs**2)
        psi_obs = jnp.arctan2(jnp.diff(y_obs,axis=1),jnp.diff(x_obs,axis=1))
        psi_obs = jnp.hstack((psi_obs,psi_obs[:,-1].reshape(self.prob.num_reduced,1)))

        xdot_obs_baseline = jnp.diff(x_obs_baseline,axis=1)/self.prob.t_up
        xdot_obs_baseline = jnp.hstack((xdot_obs_baseline,xdot_obs_baseline[:,-1].reshape(num_samples_baseline,1)))

        ydot_obs_baseline = jnp.diff(y_obs_baseline,axis=1)/self.prob.t_up
        ydot_obs_baseline = jnp.hstack((ydot_obs_baseline,ydot_obs_baseline[:,-1].reshape(num_samples_baseline,1)))

        v_obs_baseline = jnp.sqrt(xdot_obs_baseline**2+ydot_obs_baseline**2)
        psi_obs_baseline = jnp.arctan2(jnp.diff(y_obs_baseline,axis=1),jnp.diff(x_obs_baseline,axis=1))
        psi_obs_baseline = jnp.hstack((psi_obs_baseline,psi_obs_baseline[:,-1].reshape(num_samples_baseline,1)))

        xdot_obs_val = jnp.diff(x_obs_val,axis=1)/self.prob.t_up
        xdot_obs_val = jnp.hstack((xdot_obs_val,xdot_obs_val[:,-1].reshape(self.num_validation,1)))

        ydot_obs_val = jnp.diff(y_obs_val,axis=1)/self.prob.t_up
        ydot_obs_val = jnp.hstack((ydot_obs_val,ydot_obs_val[:,-1].reshape(self.num_validation,1)))

        v_obs_val = jnp.sqrt(xdot_obs_val**2+ydot_obs_val**2)
        psi_obs_val = jnp.arctan2(jnp.diff(y_obs_val,axis=1),jnp.diff(x_obs_val,axis=1))
        psi_obs_val = jnp.hstack((psi_obs_val,psi_obs_val[:,-1].reshape(self.num_validation,1)))

        x_obs_total,y_obs_total,psi_obs_total = \
                                                self.global_to_frenet_obs_vmap_1(x_obs_total,y_obs_total,v_obs_total,psi_obs_total,x_path_up, y_path_up,
                                                arc_vec_temp,
                                                    Fx_dot_up, Fy_dot_up, kappa )

        x_obs_mean,y_obs_mean,psi_obs_mean = \
                                                self.global_to_frenet_obs_vmap(x_obs_mean,y_obs_mean,v_obs_mean,psi_obs_mean,x_path_up, y_path_up,arc_vec_temp,
                                                    Fx_dot_up, Fy_dot_up, kappa )
        
        x_obs,y_obs,psi_obs = \
                                                self.global_to_frenet_obs_vmap_1(x_obs,y_obs,v_obs,psi_obs,x_path_up, y_path_up,arc_vec_temp,
                                                    Fx_dot_up, Fy_dot_up, kappa )

        x_obs_baseline,y_obs_baseline,psi_obs_baseline = \
                                                self.global_to_frenet_obs_vmap_1(x_obs_baseline,y_obs_baseline,v_obs_baseline,psi_obs_baseline,x_path_up, y_path_up,arc_vec_temp,
                                                    Fx_dot_up, Fy_dot_up, kappa )

        x_obs_val,y_obs_val,psi_obs_val = \
                                                self.global_to_frenet_obs_vmap_1(x_obs_val,y_obs_val,v_obs_val,psi_obs_val,x_path_up, y_path_up,arc_vec_temp,
                                                    Fx_dot_up, Fy_dot_up, kappa )


        x_obs_total,y_obs_total,psi_obs_total,x_obs_mean,y_obs_mean,psi_obs_mean,\
            x_obs,y_obs,psi_obs,x_obs_baseline,y_obs_baseline,psi_obs_baseline,\
                x_obs_val,y_obs_val,psi_obs_val\
                 = self.compute_splitting(x_obs_val,y_obs_val,psi_obs_val,v_obs_val,x_obs_total,y_obs_total,psi_obs_total,v_obs_total,\
            x_obs_mean,y_obs_mean,psi_obs_mean,v_obs_mean,\
               x_obs,y_obs,psi_obs,v_obs,\
                 x_obs_baseline,y_obs_baseline,psi_obs_baseline,v_obs_baseline)

        x_obs_mean = x_obs_mean.reshape((self.num_obs*self.num_circles,self.prob.num_up))
        y_obs_mean = y_obs_mean.reshape((self.num_obs*self.num_circles,self.prob.num_up))

        ###################################
        
        x_obs_long = x_obs_mean[0:self.num_obs*self.num_circles,0]
        y_obs_long = y_obs_mean[0:self.num_obs*self.num_circles,0]
        vx_obs_long = jnp.zeros(self.num_obs*self.num_circles)
        vy_obs_long = jnp.zeros(self.num_obs*self.num_circles)
      
        if(y_obs_long[0]< 1.80 and y_obs_long[0] > -1.80):
            x_obs_long = x_obs_long + vx_obs_long*self.prob.tot_time_up
            y_obs_long = y_obs_long + vy_obs_long*self.prob.tot_time_up
        else:
            x_obs_long = 10.0*jnp.ones(self.num_obs*self.num_circles) + 10.0*jnp.ones(self.num_obs*self.num_circles)*self.prob.tot_time_up
            y_obs_long = 10.0*jnp.ones(self.num_obs*self.num_circles) + jnp.zeros(self.num_obs*self.num_circles)*self.prob.tot_time_up

        x_obs_long = x_obs_long.T
        y_obs_long = y_obs_long.T
        
        c_x_best, c_y_best,res,res_2,res_3,res_4,res_5 = self.compute_cem(x_obs_val,y_obs_val,x_obs_total,y_obs_total,beta,x_obs_mean,y_obs_mean,x_obs,y_obs,initial_state,\
                                                                                lamda_x, lamda_y,
                                                                                y_lane_lb, y_lane_ub, v_des, 
                                                                                s_lane, mean_param, cov_param, \
                                                                                x_path,y_path,arc_vec,Fx_dot,Fy_dot,\
                                                                                kappa,s_long,x_obs_long,y_obs_long )
        x_best = jnp.dot(self.P_jax, c_x_best)
        y_best = jnp.dot(self.P_jax, c_y_best)

        ref_x = self.jax_interp(x_best, arc_vec, x_path )
        ref_y = self.jax_interp(x_best, arc_vec, y_path )
        dx_by_ds = self.jax_interp(x_best, arc_vec, Fx_dot )
        dy_by_ds = self.jax_interp(x_best, arc_vec, Fy_dot )

        global_x, global_y, psi_global	= self.frenet_to_global(y_best, ref_x, ref_y, dx_by_ds, dy_by_ds)

        c_x_best_baseline, c_y_best_baseline,res_baseline,res_2_baseline,res_3_baseline,res_4_baseline,res_5_baseline = self.compute_cem_baseline(x_obs_val,y_obs_val,x_obs_total,y_obs_total,\
                                                                                beta,x_obs_mean,y_obs_mean,x_obs_baseline,y_obs_baseline,initial_state,\
                                                                                 lamda_x, lamda_y,
                                                                                y_lane_lb, y_lane_ub, v_des, 
                                                                                s_lane, mean_param, cov_param, \
                                                                                x_path,y_path,arc_vec,Fx_dot,Fy_dot,\
                                                                                kappa,s_long,x_obs_long,y_obs_long )        
        
        x_best_baseline = jnp.dot(self.P_jax, c_x_best_baseline)
        y_best_baseline = jnp.dot(self.P_jax, c_y_best_baseline)

        ref_x_baseline = self.jax_interp(x_best_baseline, arc_vec, x_path )
        ref_y_baseline = self.jax_interp(x_best_baseline, arc_vec, y_path )
        dx_by_ds_baseline = self.jax_interp(x_best_baseline, arc_vec, Fx_dot )
        dy_by_ds_baseline = self.jax_interp(x_best_baseline, arc_vec, Fy_dot )

        global_x_baseline, global_y_baseline, psi_global_baseline	= self.frenet_to_global(y_best_baseline, ref_x_baseline, ref_y_baseline,
                                                                                         dx_by_ds_baseline, dy_by_ds_baseline)

        obs_data_frenet = (x_obs_total[0,0],y_obs_total[0,0],psi_obs_total[0,0])
        obs_traj_frenet = (x_obs_baseline,y_obs_baseline,x_obs,y_obs,x_obs_mean,y_obs_mean,x_obs_total,y_obs_total)

        mmd = (global_x, global_y,psi_global,x_best,y_best,res,res_2,res_3,res_4,res_5)
        alonso = (global_x_baseline, global_y_baseline,psi_global_baseline,x_best_baseline,y_best_baseline,
                res_baseline,res_2_baseline,res_3_baseline,res_4_baseline,res_5_baseline) 
        return mmd,alonso,initial_state_frenet,obs_data_frenet,obs_traj_frenet

   
    @partial(jit, static_argnums=(0,))
    def global_to_frenet_obs(self,x_obs,y_obs,v_obs,psi_obs, x_path, y_path, arc_vec, Fx_dot, Fy_dot, kappa ):

        idx_closest_point = jnp.argmin( jnp.sqrt((x_path-x_obs)**2+(y_path-y_obs)**2))
        closest_point_x, closest_point_y = x_path[idx_closest_point], y_path[idx_closest_point]

        x_init = arc_vec[idx_closest_point]

        Fx_dot_interp = self.jax_interp(x_init, arc_vec, Fx_dot)
        Fy_dot_interp = self.jax_interp(x_init, arc_vec, Fy_dot)

        normal_x = -Fy_dot_interp
        normal_y = Fx_dot_interp

        normal = jnp.hstack((normal_x, normal_y   ))
        vec = jnp.asarray([x_obs-closest_point_x,y_obs-closest_point_y ])
        y_init = (1/(jnp.linalg.norm(normal)))*jnp.dot(normal,vec)
        
        psi_init = psi_obs-jnp.arctan2(Fy_dot_interp, Fx_dot_interp)
        psi_init = jnp.arctan2(jnp.sin(psi_init), jnp.cos(psi_init))
        
        return x_init, y_init,psi_init

    @partial(jit, static_argnums=(0,))
    def closest_point(self,point, points):
        points = jnp.asarray(points)
        dist_2 = jnp.sum((points - point)**2, axis=1)
        return jnp.argmin(dist_2)

    @partial(jit, static_argnums=(0, ))
    def interp(self,x,xp,fp):
        return self.jax_interp(x,xp,fp)     

    @partial(jit, static_argnums=(0,))	
    def compute_splitting(self,x_obs_val,y_obs_val,psi_obs_val,v_obs_val,x_obs_total,y_obs_total,psi_obs_total,v_obs_total,\
            x_obs_mean,y_obs_mean,psi_obs_mean,v_obs_mean,\
               x_obs,y_obs,psi_obs,v_obs,\
                 x_obs_baseline,y_obs_baseline,psi_obs_baseline,v_obs_baseline):

        vx_obs_total = v_obs_total*jnp.cos(psi_obs_total)
        vy_obs_total = v_obs_total*jnp.sin(psi_obs_total)

        vx_obs = v_obs*jnp.cos(psi_obs)
        vy_obs = v_obs*jnp.sin(psi_obs)

        vx_obs_mean = v_obs_mean*jnp.cos(psi_obs_mean)
        vy_obs_mean = v_obs_mean*jnp.sin(psi_obs_mean)

        vx_obs_baseline = v_obs_baseline*jnp.cos(psi_obs_baseline)
        vy_obs_baseline = v_obs_baseline*jnp.sin(psi_obs_baseline)

        vx_obs_val = v_obs_val*jnp.cos(psi_obs_val)
        vy_obs_val = v_obs_val*jnp.sin(psi_obs_val)

        x_obs_frenet_circles_val = jnp.zeros((jnp.shape(x_obs_val)[0],self.num_circles,self.prob.num_up))
        y_obs_frenet_circles_val = jnp.zeros((jnp.shape(x_obs_val)[0],self.num_circles,self.prob.num_up))
        vx_obs_frenet_circles_val = jnp.zeros((jnp.shape(x_obs_val)[0],self.num_circles,self.prob.num_up))
        vy_obs_frenet_circles_val = jnp.zeros((jnp.shape(x_obs_val)[0],self.num_circles,self.prob.num_up))
        psi_obs_frenet_circles_val = jnp.zeros((jnp.shape(x_obs_val)[0],self.num_circles,self.prob.num_up))
       
        x_obs_frenet_circles_total = jnp.zeros((jnp.shape(x_obs_total)[0],self.num_circles,self.prob.num_up))
        y_obs_frenet_circles_total = jnp.zeros((jnp.shape(x_obs_total)[0],self.num_circles,self.prob.num_up))
        vx_obs_frenet_circles_total = jnp.zeros((jnp.shape(x_obs_total)[0],self.num_circles,self.prob.num_up))
        vy_obs_frenet_circles_total = jnp.zeros((jnp.shape(x_obs_total)[0],self.num_circles,self.prob.num_up))
        psi_obs_frenet_circles_total = jnp.zeros((jnp.shape(x_obs_total)[0],self.num_circles,self.prob.num_up))
        
        x_obs_frenet_circles_mean = jnp.zeros((self.num_circles,self.prob.num_up))
        y_obs_frenet_circles_mean = jnp.zeros((self.num_circles,self.prob.num_up))
        vx_obs_frenet_circles_mean = jnp.zeros((self.num_circles,self.prob.num_up))
        vy_obs_frenet_circles_mean = jnp.zeros((self.num_circles,self.prob.num_up))
        psi_obs_frenet_circles_mean = jnp.zeros((self.num_circles,self.prob.num_up))

        x_obs_frenet_circles = jnp.zeros((jnp.shape(x_obs)[0],self.num_circles,self.prob.num_up))
        y_obs_frenet_circles = jnp.zeros((jnp.shape(x_obs)[0],self.num_circles,self.prob.num_up))
        vx_obs_frenet_circles = jnp.zeros((jnp.shape(x_obs)[0],self.num_circles,self.prob.num_up))
        vy_obs_frenet_circles = jnp.zeros((jnp.shape(x_obs)[0],self.num_circles,self.prob.num_up))
        psi_obs_frenet_circles = jnp.zeros((jnp.shape(x_obs)[0],self.num_circles,self.prob.num_up))
       
        x_obs_frenet_circles_baseline = jnp.zeros((jnp.shape(x_obs_baseline)[0],self.num_circles,self.prob.num_up))
        y_obs_frenet_circles_baseline = jnp.zeros((jnp.shape(x_obs_baseline)[0],self.num_circles,self.prob.num_up))
        vx_obs_frenet_circles_baseline = jnp.zeros((jnp.shape(x_obs_baseline)[0],self.num_circles,self.prob.num_up))
        vy_obs_frenet_circles_baseline = jnp.zeros((jnp.shape(x_obs_baseline)[0],self.num_circles,self.prob.num_up))
        psi_obs_frenet_circles_baseline = jnp.zeros((jnp.shape(x_obs_baseline)[0],self.num_circles,self.prob.num_up))
       
        x_temp = jnp.zeros((self.num_circles,self.prob.num_up))
        y_temp = jnp.zeros((self.num_circles,self.prob.num_up))
        
        indices = [0,1,2]

        for l in range(0,jnp.shape(x_obs_total)[0]):
            x_temp = x_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.cos(psi_obs_total[l,:][jnp.newaxis,:]))
            y_temp = y_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.sin(psi_obs_total[l,:][jnp.newaxis,:]))
            
            x_obs_frenet_circles_total = x_obs_frenet_circles_total.at[l,indices,:].set(x_obs_total[l,:] + x_temp)
            y_obs_frenet_circles_total = y_obs_frenet_circles_total.at[l,indices,:].set(y_obs_total[l,:] + y_temp)
            vx_obs_frenet_circles_total = vx_obs_frenet_circles_total.at[l,indices,:].set(vx_obs_total[l,:])
            vy_obs_frenet_circles_total = vy_obs_frenet_circles_total.at[l,indices,:].set(vy_obs_total[l,:])
            psi_obs_frenet_circles_total = psi_obs_frenet_circles_total.at[l,indices,:].set(psi_obs_total[l,:])

        for l in range(0,jnp.shape(x_obs_baseline)[0]):
            x_temp = x_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.cos(psi_obs_baseline[l,:][jnp.newaxis,:]))
            y_temp = y_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.sin(psi_obs_baseline[l,:][jnp.newaxis,:]))
            
            x_obs_frenet_circles_baseline = x_obs_frenet_circles_baseline.at[l,indices,:].set(x_obs_baseline[l,:] + x_temp)
            y_obs_frenet_circles_baseline = y_obs_frenet_circles_baseline.at[l,indices,:].set(y_obs_baseline[l,:] + y_temp)
            vx_obs_frenet_circles_baseline = vx_obs_frenet_circles_baseline.at[l,indices,:].set(vx_obs_baseline[l,:])
            vy_obs_frenet_circles_baseline = vy_obs_frenet_circles_baseline.at[l,indices,:].set(vy_obs_baseline[l,:])
            psi_obs_frenet_circles_baseline = psi_obs_frenet_circles_baseline.at[l,indices,:].set(psi_obs_baseline[l,:])

        x_temp = x_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.cos(psi_obs_mean[jnp.newaxis,:]))
        y_temp = y_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.sin(psi_obs_mean[jnp.newaxis,:]))
        
        x_obs_frenet_circles_mean = x_obs_frenet_circles_mean.at[indices,:].set(x_obs_mean + x_temp)
        y_obs_frenet_circles_mean = y_obs_frenet_circles_mean.at[indices,:].set(y_obs_mean + y_temp)
        vx_obs_frenet_circles_mean = vx_obs_frenet_circles_mean.at[indices,:].set(vx_obs_mean)
        vy_obs_frenet_circles_mean = vy_obs_frenet_circles_mean.at[indices,:].set(vy_obs_mean)
        psi_obs_frenet_circles_mean = psi_obs_frenet_circles_mean.at[indices,:].set(psi_obs_mean)

        for l in range(0,jnp.shape(x_obs)[0]):
            x_temp = x_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.cos(psi_obs[l,:][jnp.newaxis,:]))
            y_temp = y_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.sin(psi_obs[l,:][jnp.newaxis,:]))
            
            x_obs_frenet_circles = x_obs_frenet_circles.at[l,indices,:].set(x_obs[l,:] + x_temp)
            y_obs_frenet_circles = y_obs_frenet_circles.at[l,indices,:].set(y_obs[l,:] + y_temp)
            vx_obs_frenet_circles = vx_obs_frenet_circles.at[l,indices,:].set(vx_obs[l,:])
            vy_obs_frenet_circles = vy_obs_frenet_circles.at[l,indices,:].set(vy_obs[l,:])
            psi_obs_frenet_circles = psi_obs_frenet_circles.at[l,indices,:].set(psi_obs[l,:])

        for l in range(0,jnp.shape(x_obs_val)[0]):
            x_temp = x_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.cos(psi_obs_val[l,:][jnp.newaxis,:]))
            y_temp = y_temp.at[indices,:].set(self.dist_centre[:,jnp.newaxis]*jnp.sin(psi_obs_val[l,:][jnp.newaxis,:]))
            
            x_obs_frenet_circles_val = x_obs_frenet_circles_val.at[l,indices,:].set(x_obs_val[l,:] + x_temp)
            y_obs_frenet_circles_val = y_obs_frenet_circles_val.at[l,indices,:].set(y_obs_val[l,:] + y_temp)
            vx_obs_frenet_circles_val = vx_obs_frenet_circles_val.at[l,indices,:].set(vx_obs_val[l,:])
            vy_obs_frenet_circles_val = vy_obs_frenet_circles_val.at[l,indices,:].set(vy_obs_val[l,:])
            psi_obs_frenet_circles_val = psi_obs_frenet_circles_val.at[l,indices,:].set(psi_obs_val[l,:])

        x_obs_frenet_circles_val = x_obs_frenet_circles_val.reshape(-1,x_obs_frenet_circles_val.shape[-1])
        y_obs_frenet_circles_val = y_obs_frenet_circles_val.reshape(-1,y_obs_frenet_circles_val.shape[-1])
        psi_obs_frenet_circles_val = psi_obs_frenet_circles_val.reshape(-1,psi_obs_frenet_circles_val.shape[-1])

        x_obs_frenet_circles_total = x_obs_frenet_circles_total.reshape(-1,x_obs_frenet_circles_total.shape[-1])
        y_obs_frenet_circles_total = y_obs_frenet_circles_total.reshape(-1,y_obs_frenet_circles_total.shape[-1])
        psi_obs_frenet_circles_total = psi_obs_frenet_circles_total.reshape(-1,psi_obs_frenet_circles_total.shape[-1])
        
        x_obs_frenet_circles_baseline = x_obs_frenet_circles_baseline.reshape(-1,x_obs_frenet_circles_baseline.shape[-1])
        y_obs_frenet_circles_baseline = y_obs_frenet_circles_baseline.reshape(-1,y_obs_frenet_circles_baseline.shape[-1])
        psi_obs_frenet_circles_baseline = psi_obs_frenet_circles_baseline.reshape(-1,psi_obs_frenet_circles_baseline.shape[-1])

        x_obs_frenet_circles_mean = x_obs_frenet_circles_mean.reshape(-1,x_obs_frenet_circles_mean.shape[-1])
        y_obs_frenet_circles_mean = y_obs_frenet_circles_mean.reshape(-1,y_obs_frenet_circles_mean.shape[-1])
        psi_obs_frenet_circles_mean = psi_obs_frenet_circles_mean.reshape(-1,psi_obs_frenet_circles_mean.shape[-1])

        x_obs_frenet_circles = x_obs_frenet_circles.reshape(-1,x_obs_frenet_circles.shape[-1])
        y_obs_frenet_circles = y_obs_frenet_circles.reshape(-1,y_obs_frenet_circles.shape[-1])
        psi_obs_frenet_circles = psi_obs_frenet_circles.reshape(-1,psi_obs_frenet_circles.shape[-1])

        return x_obs_frenet_circles_total,y_obs_frenet_circles_total,psi_obs_frenet_circles_total,\
            x_obs_frenet_circles_mean,y_obs_frenet_circles_mean,psi_obs_frenet_circles_mean,\
            x_obs_frenet_circles,y_obs_frenet_circles,psi_obs_frenet_circles,\
            x_obs_frenet_circles_baseline,y_obs_frenet_circles_baseline,psi_obs_frenet_circles_baseline,\
            x_obs_frenet_circles_val,y_obs_frenet_circles_val,psi_obs_frenet_circles_val


    