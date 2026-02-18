
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from pathlib import Path
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt; plt.ion()

##############################################################
# Model definition

model_code = """
functions {
    real hmm_marginal_banded(matrix log_omega,
                         vector Gamma_diag,
                         vector rho) {
        int K = dims(log_omega)[1];
        int N = dims(log_omega)[2];

        vector[K] log_alpha;
        vector[K] inner_sum;
        vector[2] inner_vec;
        vector[K] log_Gamma_diag = log(Gamma_diag);
        vector[K] log_1mGamma_diag = log1m(Gamma_diag);

        int min_i; int max_i;

        log_alpha = log_omega[, 1] + log(rho);

        if (N > 1) {
            for (n in 2:N) {
                for (i in 1:K) {
                    if (i == 1) {
                        inner_sum[i] = log_alpha[i] + log_Gamma_diag[i];
                    } else {
                        inner_vec[1] = log_alpha[i - 1] + log_1mGamma_diag[i - 1];
                        inner_vec[2] = log_alpha[i] + log_Gamma_diag[i];
                        inner_sum[i] = log_sum_exp(inner_vec);
                    }
                }
                log_alpha = log_omega[, n] + inner_sum;
            }
        }

        return log_sum_exp(log_alpha);
    }

    matrix diag_trans_to_full(vector trans_mat_diag) {
        int n = size(trans_mat_diag);
        matrix[n, n] full_mat = diag_matrix(trans_mat_diag);

        for (i in 1:(n - 1)) {
            full_mat[i, i + 1] = 1 - trans_mat_diag[i];
        }
        full_mat[n, n] = 1.0 - 1e-6;
        full_mat[n, 1] = 1e-6;

        return full_mat;
    }

    vector tile(vector x, int r) {
        int n = size(x);
        vector[n * r] result;
        for (i in 1:r) {
            result[((i - 1)*n + 1):(i*n)] = x;
        }
        return result;
    }
}
data {
    int n;  // num data
    int s;  // num states per year
    int num_years;
    vector[n] depth;  // depth data
    vector[n] y;  // concentration data
    vector[s * num_years] initial_probs;
}
transformed data {
    int n_st = s * num_years;  // total number of states
    vector[s] year_fractions;
    year_fractions = cumulative_sum(rep_vector(1.0/s, s));
    simplex[n_st] rho = initial_probs + 1e-10;
    rho = rho/sum(rho);
}
parameters {
    vector<lower=0, upper=1>[s] p_diag;
    real<lower=-3, upper=3> mu;
    real<lower=0, upper=1> sigma;
    real<lower=0, upper=2> scale;
}
transformed parameters {
    vector[s] cosine_term = cos(2.0*pi()*(year_fractions + 0.5));
    matrix[n_st, n] log_omega;

    for(i in 1:num_years){
        for(j in 1:s) {
            for(k in 1:n) {
                log_omega[(i-1)*s + j, k] =
                      normal_lpdf(y[k] | mu + cosine_term[j]*scale, sigma);
            }
        }
    }
}
model {
    target += hmm_marginal_banded(log_omega, tile(p_diag, num_years), rho);
}
generated quantities {
    matrix[n_st, n_st] p_full = diag_trans_to_full(tile(p_diag, num_years));
    matrix[n_st, n] posterior = hmm_hidden_state_prob(log_omega, p_full, rho);
    array[n] int sampled_states = hmm_latent_rng(log_omega, p_full, rho);

    vector[n_st] state_means = rep_vector(mu, n_st) + scale * tile(cosine_term, num_years);
    vector[n] y_hat;
    for (k in 1:n) y_hat[k] = state_means[sampled_states[k]];
}
"""

with open('hmm.stan', 'w') as f:
    f.write(model_code)

model = CmdStanModel(stan_file='hmm.stan')

##############################################################
# Data preperation

df = pd.read_csv('data.csv').iloc[:750]
depth, data, expert_labels = (np.asarray(x) for x in [df.depth, df.data, df.expert_labels])

s = 5  # number of states in every year
num_years = int(expert_labels.max() - expert_labels.min() + 10)

initial_probs = np.concatenate([
    np.ones(s)/s,  # uniform over the first year
    np.zeros((num_years - 1) * s)  # zero for other states
], axis=0)

pd.Series(dict(
    n=len(data), s=s, num_years=num_years, initial_probs=initial_probs, y=data, depth=depth
)).to_json('data.json')

##############################################################
# Inference and plotting

fit = model.optimize(data='data.json', seed=42, show_console=False)
posterior = fit.stan_variable('posterior')
sampled_states = fit.stan_variable('sampled_states')
y_hat = fit.stan_variable('y_hat')
year_preds = 2012 - sampled_states/s
state_years = 2012 - (np.arange(n_states) + 0.5) / s

fig, ax = plt.subplots(figsize=(5, 3))
im = ax.imshow(
    posterior,
    aspect="auto",
    origin="upper",
    extent=[depth.min(), depth.max(),
            state_years.min(), state_years.max()]
)

ax.plot(depth, expert_labels, "w-", lw=1, label="expert")
ax.set_xlabel("depth")
ax.set_ylabel("year")

plt.tight_layout()

plt.plot(depth, data, label="data")
plt.plot(depth, y_hat, label="hmm mean")
plt.xlabel("depth")
plt.ylabel("concentration")
plt.legend()
plt.tight_layout()
