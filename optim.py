import numpy as np 

class Scheduleoptim():

    def __init__(self,optim,embd_dim,lr_mul,warmup_steps):
        self.optimizer=optim
        self.lr_mul=lr_mul
        self.d_model=embd_dim
        self.warmup_step=warmup_steps
        self.steps=0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        d_model=self.d_model
        warmup,n_steps=self.warmup_step,self.steps
        return (d_model**-0.5)*min(n_steps**-0.5,(n_steps*(warmup**(-1.5))))
    
    def update_learning_rate(self):

        self.steps+=1
        lr= self.lr_mul*self.get_lr_scale()

        for params in self.optimizer.param_groups:
            params["lr"]=lr

    def step_and_update(self):
        self.update_learning_rate()
        self.optimizer.step()
        