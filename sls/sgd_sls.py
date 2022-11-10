import copy
import time

from .sls_base import StochLineSearchBase, get_grad_list, compute_grad_norm, random_seed_torch


class SgdSLS(StochLineSearchBase):
    """Implements stochastic line search
    `paper <https://arxiv.org/abs/1905.09997>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        c (float, optional): armijo condition constant (default: 0.1)
        gamma (float, optional): factor used by Armijo for scaling the step-size at each line-search step (default: 2.0)
        reset_option (float, optional): sets the rest option strategy (default: 1)
        line_search_fn (float, optional): the condition used by the line-search to find the 
                    step-size (default: Armijo)
    """

    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 c=0.1,
                 beta_b=0.9,
                 strategy = "cycle",
                 gamma=2.0,
                 reset_option=1,
                 line_search_fn="armijo",):
        params = list(params)
        
        super().__init__(params,
                         n_batches_per_epoch=n_batches_per_epoch,
                         init_step_size=init_step_size,
                         c=c,
                         beta_b=beta_b,
                         gamma=gamma,
                         reset_option=reset_option,
                         line_search_fn=line_search_fn)
        
        self.strategy = strategy
        self.nextcycle = 0
        self.init_step_sizes = [init_step_size for i in range(len(params))]


    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()

        loss = closure_deterministic()
        loss.backward()

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # save the current parameters:
        params_current = [copy.deepcopy(param) for param in self.params]
        grad_current = [get_grad_list(param) for param in self.params]
        grad_norm = [compute_grad_norm(grad) for grad in grad_current]


        step_sizes = self.state.get('step_sizes') or self.init_step_sizes
        step_sizes = [self.reset_step(step_size=step_size,
                                    n_batches_per_epoch=self.n_batches_per_epoch,
                                    gamma=self.gamma,
                                    reset_option=self.reset_option,
                                    init_step_size=self.init_step_size) for step_size in step_sizes]
        

        for i,step_size in enumerate(step_sizes):
            if self.strategy == "cycle":
                if i == self.nextcycle:
                    step_size, loss_next = self.line_search(i,step_size, params_current[i], grad_current[i], loss, closure_deterministic, grad_norm[i])
                    step_sizes[i] = step_size
                else:
                    self.try_sgd_update(self.params[i], step_size, params_current[i], grad_current[i])
        self.nextcycle += 1
        if self.nextcycle >= len(self.params):
            self.nextcycle = 0
        self.save_state(step_sizes, loss, loss_next, grad_norm)

        return loss
