import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random

class beta_cem():

    def __init__(self):

        self.maxiter_beta_cem = 100
        self.num_samples = 100
        self.num_samples_reduced_set = 10

        self.mean_beta = jnp.zeros(self.num_samples)

        cov_diag = np.ones(self.num_samples)

        self.cov_beta = 5*jnp.asarray(np.diag(cov_diag))
        key = random.PRNGKey(0)
        self.num_ellite_beta = 20

        self.key = key

        self.cost_batch_func = jit(jax.vmap(self.compute_cost, in_axes=(0, None, None)  ))


    @partial(jit, static_argnums=(0,))	
    def compute_beta_samples_initial(self, mean_beta, cov_beta):

        key, subkey = random.split(self.key)
        
        beta_samples = jax.random.multivariate_normal(key, mean_beta, cov_beta, (self.num_samples, ))			

        
        return beta_samples

    @partial(jit, static_argnums=(0,))	
    def compute_cost(self, beta, Q, q):

        idx_beta = jnp.argsort(jnp.abs(beta))

        beta_bottom = beta[idx_beta[0:self.num_samples-self.num_samples_reduced_set]]
        beta_top = beta[idx_beta[self.num_samples-self.num_samples_reduced_set:self.num_samples]]

        cost_mmd = jnp.dot( beta.T, jnp.dot(Q, beta) )+jnp.dot(q.T, beta)
        cost_selection = jnp.sum(jnp.abs(beta_top))/jnp.sum(jnp.abs(beta_bottom))

        cost = cost_mmd - cost_selection #+ 0.01*jnp.sum(jnp.abs(beta))

        return cost


    @partial(jit, static_argnums=(0,))	
    def compute_mean_cov_beta(self, cost_batch, beta):

        key, subkey = random.split(self.key)
        
        idx_ellite = jnp.argsort(cost_batch)

        beta_ellite = beta[idx_ellite[0:self.num_ellite_beta]]

        mean_beta = jnp.mean(beta_ellite, axis = 0)
        cov_beta = jnp.cov(beta_ellite.T)+0.01*jnp.identity(self.num_samples)

        beta_samples = jax.random.multivariate_normal(key, mean_beta, cov_beta, (self.num_samples-self.num_ellite_beta, ))

        beta_samples = jnp.vstack(( beta_ellite, beta_samples  ))

        return beta_samples, mean_beta, cov_beta, beta_ellite, idx_ellite

    
    @partial(jit, static_argnums=(0, ))	
    def compute_cem(self, Q, q):

        res = jnp.zeros(self.maxiter_beta_cem)
        res_2 = jnp.zeros((self.maxiter_beta_cem,self.num_samples))

        mean_beta = self.mean_beta
        cov_beta = self.cov_beta

        beta_samples = self.compute_beta_samples_initial(mean_beta, cov_beta)

        for i in range(0, self.maxiter_beta_cem):

            cost_batch = self.cost_batch_func(beta_samples, Q, q)
            beta_samples, mean_beta, cov_beta, beta_ellite, idx_ellite = self.compute_mean_cov_beta(cost_batch, beta_samples)
            beta_best = beta_ellite[0]

            res = res.at[i].set((jnp.min(cost_batch)))
            res_2 = res_2.at[i].set(beta_best)

        return beta_best



    
