#https://jonathan-guerne.medium.com/multi-objective-bayesian-optimization-with-botorch-3c5cf348c63b
#Source: https://github.com/JonathanGuerne/BoTorchMutliObjective/blob/main/Multi%20Objective%20Optimization.ipynb
#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function


# In[27]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.sampling.samplers import SobolQMCNormalSampler


# In[4]:


noise_level = 0.2

bounds = torch.tensor([[-1.2], [1.2]])

def f(x, noise_level=noise_level):
    results = []
    for x_i in x:   
       sub_results = [
         (np.sin(6 *x_i[0])**3 * (1 - np.tanh(x_i[0] ** 2))) + (-1 + torch.rand(1)[0] * 2) * noise_level,
         .5 - (np.cos(5 * x_i[0] + 0.7)**3 * (1 - np.tanh(x_i[0] ** 2))) + (-1 + torch.rand(1)[0] * 2) * noise_level,
      ]
       results.append(sub_results)
    return torch.tensor(results, dtype=torch.float32)

def f_no_noise(x):
   return f(x, noise_level=0)


# In[5]:


xs = np.linspace(bounds[0][0], bounds[1][0], 1000)
xs = xs.reshape((1000, -1))
ys_true = f(xs, noise_level=0).numpy()
plt.plot(xs, ys_true, label=["F1", "F2"])
# the fill method constructs a polygon of the specified color delimited by all the point
# in the xs and ys arrays.
plt.fill(np.concatenate([xs, xs[::-1]]),
         np.concatenate(([fx_i - 1 * noise_level for fx_i in ys_true],
                         [fx_i + 1 * noise_level for fx_i in ys_true[::-1]])),
         alpha=.2, ec="None")
plt.legend()
plt.grid()
plt.show()


# In[6]:


for i in range(100):
    x_init = torch.tensor(bounds[0]) + torch.rand(3, 1) * (torch.tensor(bounds[1]) - torch.tensor(bounds[0]))
    ys_true = f(x_init, noise_level=0)
    ys_evaluated = f(x_init, noise_level=0.2)

    assert torch.abs(ys_true - ys_evaluated).max() < 0.2, torch.abs(ys_true - ys_evaluated).max()


# In[7]:


def generate_initial_data(n_samples):
    x_init = torch.tensor(bounds[0]) + torch.rand(n_samples, 1) * (torch.tensor(bounds[1]) - torch.tensor(bounds[0]))
    y_init = f(x_init)
    y_init_true = f(x_init, noise_level=0)
    return x_init, y_init, y_init_true


# In[8]:


NUM_RESTARTS =  10
RAW_SAMPLES = 1024

standard_bounds = torch.tensor([[0.0], [1.0]])
MC_SAMPLES = 256


# In[9]:


def initialize_model(train_x, train_y):
    
    train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_y.shape[-1]):
        train_objective = train_y[:, i]
        models.append(
            SingleTaskGP(train_x, train_objective.unsqueeze(-1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# In[10]:


def generate_next_candidate(x, y, n_candidates=1):
    
    mll, model = initialize_model(x, y)
    fit_gpytorch_model(mll)

    sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

    train_x = normalize(x, bounds)
    with torch.no_grad():
        pred = model.posterior(normalize(train_x,bounds)).mean
    
    acq_fun_list = []
    for _ in range(n_candidates):
        
        weights = sample_simplex(2).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(
                weights,
                pred
            )
        )
        acq_fun = qNoisyExpectedImprovement(
            model=model,
            objective=objective,
            sampler=sampler,
            X_baseline=train_x,
            prune_baseline=True,
        )
        acq_fun_list.append(acq_fun)
    

    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_fun_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={
            "batch_limit": 5,
            "maxiter": 200,
        }
    )

    return unnormalize(candidates, bounds)


# In[11]:


def plot_candidates(candidates):
        xs = np.linspace(bounds[0][0], bounds[1][0], 1000)
        xs = xs.reshape((1000, -1))
        ys_true = f(xs, noise_level=0).numpy()
        plt.plot(xs, ys_true, label=["F1", "F2"])
        # the fill method constructs a polygon of the specified color delimited by all the point
        # in the xs and ys arrays.
        plt.fill(np.concatenate([xs, xs[::-1]]),
                np.concatenate(([fx_i - 1 * noise_level for fx_i in ys_true],
                                [fx_i + 1 * noise_level for fx_i in ys_true[::-1]])),
                alpha=.2, ec="None")

        plt.scatter(candidates[:, 0], candidates[:, 1], c='blue', label='Candidates')
        plt.scatter(candidates[:, 0], candidates[:, 2], c='orange', label='Candidates')

        plt.legend()
        plt.grid()
        plt.show()


# In[12]:


n_iter = 16
n_start = 3
n_samples = 1

x,y,y_true = generate_initial_data(n_start)

plot_candidates(torch.cat([x,y], dim=1))

for i in range(n_iter):
    print(f"Iteration {i}")

    candidates = generate_next_candidate(x, -y, n_candidates=n_samples)

    print(f"Candidates: {candidates}")

    x = torch.cat([x, candidates])
    y = torch.cat([y, f(candidates)], dim=0)

    plot_candidates(torch.cat([x,y], dim=1))


# In[ ]:




