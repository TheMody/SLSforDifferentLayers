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
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
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


    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with random_seed_torch(seed):
                return closure()

        loss = closure_deterministic()
        loss.backward()

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = get_grad_list(self.params)
        grad_norm = compute_grad_norm(grad_current)

        step_size = self.reset_step(step_size=self.state.get('step_size') or self.init_step_size,
                                    n_batches_per_epoch=self.n_batches_per_epoch,
                                    gamma=self.gamma,
                                    reset_option=self.reset_option,
                                    init_step_size=self.init_step_size)

        step_size, loss_next = self.line_search(step_size, params_current, grad_current, loss, closure, grad_norm)
        self.save_state(step_size, loss, loss_next, grad_norm)
        return loss
