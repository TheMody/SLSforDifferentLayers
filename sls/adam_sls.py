import copy
import time
import torch
import numpy as np

from .sls_base import StochLineSearchBase, get_grad_list, compute_grad_norm, random_seed_torch, try_sgd_update

#gets a nested list of parameters as input
class AdamSLS(StochLineSearchBase):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 c=0.1,
                 gamma=2.0,
                 beta=0.999,
                 momentum=0.9,
                 gv_option='per_param',
                 base_opt='adam',
                 pp_norm_method='pp_armijo',
                 strategy = "cycle",
                 mom_type='standard',
                 clip_grad=False,
                 beta_b=0.9,
                 beta_f=2.0,
                 reset_option=1,
                 timescale = 0.05,
                 first_step = True,
                 line_search_fn="armijo"):
        params = list(params)
        super().__init__(params,
                         n_batches_per_epoch=n_batches_per_epoch,
                         init_step_size=init_step_size,
                         c=c,
                         beta_b=beta_b,
                         gamma=gamma,
                         reset_option=reset_option,
                         line_search_fn=line_search_fn)
        self.mom_type = mom_type
        self.pp_norm_method = pp_norm_method

        self.init_step_sizes = [init_step_size for i in range(len(params))]
        # sps stuff
        # self.adapt_flag = adapt_flag

        # sls stuff
        self.beta_f = beta_f
        self.beta_b = beta_b
        self.reset_option = reset_option

        # others
        self.strategy = strategy
        self.nextcycle = 0
        self.params = params
        paramslist = []
        for param in self.params:
            paramslist = paramslist + param
        if self.mom_type == 'heavy_ball':
            self.params_prev = copy.deepcopy(params) 

        if self.strategy == "impact_mag":
            self.time_since_last_update = [0 for i in range(len(params))]
            self.importance = [0.0001 for i in range(len(params))]
            self.steps_taken = [0 for i in range(len(params))]

        self.momentum = momentum
        self.beta = beta
        self.first_step = first_step
        self.timescale = timescale
        # self.state['step_size'] = init_step_size

        self.clip_grad = clip_grad
        self.gv_option = gv_option
        self.base_opt = base_opt
        # self.step_size_method = step_size_method

        # gv options
        self.gv_option = gv_option
        if self.gv_option in ['scalar']:
            self.state['gv'] = 0.

        elif self.gv_option == 'per_param':
            self.state['gv'] = [[torch.zeros(p.shape).to(p.device) for p in params] for params in self.params]

            if self.base_opt in ['amsgrad', 'adam']:
                self.state['mv'] = [[torch.zeros(p.shape).to(p.device) for p in params] for params in self.params]
            
            if self.base_opt == 'amsgrad':
                self.state['gv_max'] = [[torch.zeros(p.shape).to(p.device) for p in params] for params in self.params]

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        start = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()

        loss = closure_deterministic()
        loss.backward()
        # print("time for backwards:", time.time()-start)
        # start = time.time()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1        
        # save the current parameters:
        params_current = [copy.deepcopy(param) for param in self.params]
        grad_current = [get_grad_list(param) for param in self.params]
        self.allgrad_current = []
        for grad in grad_current:
            for g in grad:
                self.allgrad_current.append(g)
        # print("time for copies:", time.time()-start)
        # start = time.time()

        grad_norm = [compute_grad_norm(grad) for grad in grad_current]
        # print("time for grad norm:", time.time()-start)
        # start = time.time()
        #  Gv options
        # =============
        if self.gv_option in ['scalar']:
            # update gv
            self.state['gv'] += (grad_norm.item())**2

        elif self.gv_option == 'per_param':
            # update gv
            for a, grad in enumerate(grad_current):
                for i, g in enumerate(grad):
                    if isinstance(g, torch.Tensor) and isinstance(self.state['gv'][a][i], torch.Tensor):
                        if g.device != self.state['gv'][a][i].device:
                            self.state['gv'][a][i] = self.state['gv'][a][i].to(g.device)
                    if isinstance(g, torch.Tensor) and isinstance(self.state['mv'][a][i], torch.Tensor):
                        if g.device != self.state['mv'][a][i].device:
                            self.state['mv'][a][i] = self.state['mv'][a][i].to(g.device)
                    if self.base_opt == 'adagrad':
                        self.state['gv'][a][i] += g**2 

                    elif self.base_opt == 'rmsprop':
                        self.state['gv'][a][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a][i]

                    elif self.base_opt in ['amsgrad', 'adam']:
                        self.state['gv'][a][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a][i]
                        self.state['mv'][a][i] = (1-self.momentum)*g + (self.momentum) * self.state['mv'][a][i]

                    else:
                        raise ValueError('%s does not exist' % self.base_opt)

        # print("time for gv and mv calcs:", time.time()-start)
        # start = time.time()
        pp_norm, pp_norms = self.get_pp_norm(grad_current=self.allgrad_current)
        step_sizes = self.state.get('step_sizes') or self.init_step_sizes
        step_sizes = [self.reset_step(step_size=step_size,
                                    n_batches_per_epoch=self.n_batches_per_epoch,
                                    gamma=self.gamma,
                                    reset_option=self.reset_option,
                                    init_step_size=self.init_step_size) for step_size in step_sizes]
        # print("time for ppnorm:", time.time()-start)
        # start = time.time()

        # compute step size and execute step
        # =================
        if self.pp_norm_method == "just_pp":
            
            orig_step = pp_norm*self.init_step_size
            step_size, loss_next = self.line_search(orig_step, params_current, grad_current, loss, closure_deterministic, grad_norm)
            try_sgd_update(self.params, step_size, params_current, grad_current)
        else:
            if self.first_step:
                step_size, loss_next = self.line_search(-1,step_sizes[0], params_current, grad_current, loss, closure_deterministic, grad_norm, non_parab_dec=pp_norm, precond=True)
                step_sizes = [step_size for i in range(len(step_sizes))]
          #      self.c  = self.c / len(step_sizes)
                self.first_step = False
            else:
                if self.strategy == "impact_mag":
                    probabilities= [(2**(self.time_since_last_update[i]*self.timescale))*0.2 +self.importance[i]/np.sum(self.importance) for i in range(len(step_sizes))]
                    probabilities = [p/np.sum(probabilities) for p in probabilities]
                  #  print(probabilities)
                    rand = np.random.choice([a for a in range(len(step_sizes))], p = probabilities)
                for i,step_size in enumerate(step_sizes):
                    if self.strategy == "impact_mag":
                        if rand == i:
                            step_size, loss_next = self.line_search(i,step_size, params_current[i], grad_current[i], loss, closure_deterministic, grad_norm[i], non_parab_dec=pp_norm, precond=True)
                            step_sizes[i] = step_size
                            self.importance[i] = (loss.item() - loss_next.item())*(1-self.momentum) + self.importance[i]*self.momentum
                            self.time_since_last_update[i] = 0
                        else:
                            self.try_sgd_precond_update(i,self.params[i], step_size, params_current[i], grad_current[i], self.momentum)
                            self.time_since_last_update[i] += 1
                        
                    if self.strategy == "cycle":
                        if i == self.nextcycle:
                            step_size, loss_next = self.line_search(i,step_size, params_current[i], grad_current[i], loss, closure_deterministic, grad_norm[i], non_parab_dec=pp_norm, precond=True)
                            step_sizes[i] = step_size
                        else:
                            self.try_sgd_precond_update(i,self.params[i], step_size, params_current[i], grad_current[i], self.momentum)
                self.nextcycle += 1
                if self.nextcycle >= len(self.params):
                    self.nextcycle = 0
        self.save_state(step_sizes, loss, loss_next, grad_norm)

        if torch.isnan(self.params[0][0]).sum() > 0:
            raise ValueError('nans detected')

        return loss

    def get_pp_norm(self, grad_current):
        if self.pp_norm_method in ['pp_armijo', "just_pp"]:
            pp_norm = 0
          #  pp_norms = []
            allstates = []
            for state in self.state['gv']:
                for s in state:
                    allstates.append(s)
            for i, (g_i, gv_i) in enumerate(zip(grad_current, allstates)):
                if self.base_opt in ['diag_hessian', 'diag_ggn_ex', 'diag_ggn_mc']:
                    pv_i = 1. / (gv_i+ 1e-8) # computing 1 / diagonal for using in the preconditioner

                elif self.base_opt == 'adam':
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                elif self.base_opt == 'amsgrad':
                    self.state['gv_max'][i] = torch.max(gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(self.state['gv_max'][i], self.beta, self.state['step']+1)

                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                elif self.base_opt in ['adagrad', 'rmsprop']:
                    pv_i = 1./(torch.sqrt(gv_i) + 1e-8)
                else:
                    raise ValueError('%s not found' % self.base_opt)
                if self.pp_norm_method == 'pp_armijo':
                    layer_norm = ((g_i**2) * pv_i).sum()
                elif self.pp_norm_method == "just_pp":
                    layer_norm = pv_i.sum()
                pp_norm += layer_norm
              #  pp_norms.append(layer_norm.item())

        elif self.pp_norm_method in ['pp_lipschitz']:
            pp_norm = 0

            for g_i in grad_current:
                if isinstance(g_i, float) and g_i == 0:
                    continue
                pp_norm += (g_i * (g_i + 1e-8)).sum()

        else:
            raise ValueError('%s does not exist' % self.pp_norm_method)

        return pp_norm, 0#pp_norms

    @torch.no_grad()
    def try_sgd_precond_update(self, i,params, step_size, params_current, grad_current, momentum):
        if self.gv_option in ['scalar']:
            zipped = zip(params, params_current, grad_current, self.state['gv'])
        
            for p_next, p_current, g_current, gv_i in zipped:
                p_next.data = p_current - (step_size / torch.sqrt(gv_i)) * g_current
        
        elif self.gv_option == 'per_param':
            if self.base_opt == 'adam':
                if i == -1:
                    zipped = zip([item for sublist in params for item in sublist], [item for sublist in params_current for item in sublist], [item for sublist in grad_current for item in sublist], 
                        [item for sublist in self.state['gv'] for item in sublist],[item for sublist in self.state['mv'] for item in sublist] )
                else:
                    zipped = zip(params, params_current, grad_current, self.state['gv'][i], self.state['mv'][i])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0. or  self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)

                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)
            
            elif self.base_opt == 'amsgrad':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])
                
                for i, (p_next, p_current, g_current, gv_i, mv_i) in enumerate(zipped):
                    self.state['gv_max'][i] = torch.max(gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(self.state['gv_max'][i], self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)
                    
                    if momentum == 0. or  self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)
                    else:
                        raise ValueError('does not exist')

                    # p_next.data = p_current - step_size * (pv_list *  mv_i_scaled)
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)

            elif (self.base_opt in ['rmsprop', 'adagrad']):
                zipped = zip(params, params_current, grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    pv_list = 1./ (torch.sqrt(gv_i) + 1e-8)
                    # p_next.data = p_current - step_size * (pv_list *  g_current)
    
                    p_next.data[:] = p_current.data
                    p_next.data.add_( (pv_list *  g_current), alpha=- step_size)

            elif (self.base_opt in ['diag_hessian', 'diag_ggn_ex', 'diag_ggn_mc']):
                zipped = zip(params, params_current, grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    pv_list = 1./ (gv_i+ 1e-8)  # adding 1e-8 to avoid overflow.
                    # p_next.data = p_current - step_size * (pv_list *  g_current)

                    # need to do this variant of the update for LSTM memory problems.
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  g_current), alpha=- step_size)
            

            else:
                raise ValueError('%s does not exist' % self.base_opt)

        else:
            raise ValueError('%s does not exist' % self.gv_option)

def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1-alpha**(max(1, step)))
    return vector / scale

