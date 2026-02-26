import torch
import torch.nn as nn
import math
import pytorch_lightning as ptl

def create_warmup_cosine_scheduler(optimizer, warmup_steps, max_steps, min_lr = 0):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(min(1,step)) / float(warmup_steps)
        
        progress = float(step - warmup_steps) / float(max_steps - warmup_steps)
        return max(min_lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CosineAnnealDecay():
	"""
	inspired by https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay/blob/main/linear_warmup_cosine_annealing_warm_restarts_weight_decay/lr_scheduler.py
	"""

	def __init__(
		self,
		T_start,
		T_interval,
		min_lr = 0.0001,
		start_lr = 0.1,
		T_mul = 1.):

		self.T_start = T_start
		self.T_mul = T_mul
		self.base_max_lr = start_lr
		self.max_lr = start_lr
		self.T_i = T_interval
		self.cycle = 0
		self.min_lr = min_lr
		self.T_warmup = T_start

	def get_lr(self, t):
		
		T_curr = t - self.T_start
		if T_curr >= self.T_i:
			self.cycle += 1
			T_curr = T_curr - self.T_i
			self.T_start += self.T_i
			self.T_i *= self.T_mul
		
			#Square-root cool-down of max_lr
			self.max_lr = max(self.min_lr , self.base_max_lr * self.T_warmup**2 * t**-2)

		new_lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * T_curr / self.T_i)) / 2
		return new_lr

def LayerParInit(layer, type = 'He'):

	if type == "He_normal":
		nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
		if layer.bias is not None:
			nn.init.zeros_(layer.bias)
	elif type == "He_uniform":
		nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
		if layer.bias is not None:
			nn.init.zeros_(layer.bias)
	elif type == "esm":
		nn.init.xavier_uniform_(layer.weight, gain=1 / 2**0.5)
		if layer.bias is not None:
			nn.init.zeros_(layer.bias)
	elif type == "xavier_normal":
		nn.init.xavier_normal_(layer.weight, gain=1 / 2**0.5)
		if layer.bias is not None:
			nn.init.zeros_(layer.bias)

"""

class ValidationCallback(plt.Callback):
	def __init__(self, partial_eval_interval = 0.1, full_eval_interval = 0.5, partial_eval_frac = 0.1):
		self.part_ti = partial_eval_interval
		self.full_ti = full_eval_interval
		self.part_frac = float(partial_eval_frac)
		self.last_part_val_ = 0
		self.last_full_val_ = 0
	
	def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
		curr_progress = (batch_idx + 1) / len(trainer.train_dataloader.dataset)

		# Trigger partial validation check if needed
		if curr_progress - self.last_part_val_ >= self.part_ti:
			self.run_partial_validation(trainer, pl_module)
			self.last_part_val_ = curr_progress
		
		# Trigger full validation check if needed
		if curr_progress - self.last_full_val_ >= self.full_ti:
			self.run_full_validation(trainer, pl_module)
			self.last_full_val_ = curr_progress
	
	def run_partial_validation(self, trainer, pl_module):
		val_dataloader = trainer.val_dataloaders[0]
		partial_size = int(len(val_dataloader.dataset) * self.part_frac)
		val_subset = torch.utils.data.subset(val_dataloader.dataset, range(partial_size))
		trainer.validate(pl_module, val_dataloaders = [torch.utils.data.DataLoader(val_subset)])
	
	def run_full_validation(self, trainer, pl_module):
		trainer.validate(pl_module)

"""
