import torch
import torch.nn as nn
from modules import *

def print_memory(note = None):
	for i in range(torch.cuda.device_count()):
			print(note)
			print(f"GPU {i}: Allocated Memory - {torch.cuda.memory_allocated(i) / 1e9} GB")

def LayerParInit(layer, std = 0.01):
	nn.init.normal_(layer.weight, std = std)
	if layer.bias is not None:
		nn.init.zeros_(layer.bias)

class ResBlock(nn.Module):
	def __init__(self, input_dim, out_dim, mid_dim):
		super().__init__()
		self.input_dim_ = input_dim
		self.out_dim_ = out_dim
		self.mid_dim_ = mid_dim
		self._init_modules()

	def _init_modules(self):
		self.FC1 = nn.Linear(self.input_dim_, self.mid_dim_)
		self.FC2 = nn.Linear(self.mid_dim_, self.out_dim_)
		self.ResProj = nn.Linear(self.input_dim_, self.out_dim_)

	def forward(self,x):
		res = x
		x = self.FC1(x)
		x = nn.functional.leaky_relu(x)
		x = self.FC2(x)
		x = x + self.ResProj(res)
		return x #last non-linearity is sigmoid which is within loss


class g2p_FC(nn.Module):
	def __init__(self, input_dim, use_gender = True):
		super().__init__()
		self.input_dim_ = input_dim
		self.use_gender_ = use_gender
		self._init_modules()

	def _init_modules(self):
		extra_dim = 2 if self.use_gender_ else 0
		self.FC1 = nn.Linear(self.input_dim_ + extra_dim, 1)

	def forward(self, doses, sex):
		if self.use_gender_:
			sex = sex.flatten().long()
			sex = nn.functional.one_hot(sex)
			doses = torch.cat([doses,sex], axis = 1)
		ret = self.FC1(doses)
		return ret

class g2p_residual(nn.Module):
	def __init__(self, input_dim, mid_dim = 256):
		super().__init__()
		self.res1 = ResBlock(input_dim = input_dim, out_dim = 1, mid_dim = mid_dim)

	def forward(self, doses):
		ret = self.res1(doses)
		return ret

class g2p_transformer(nn.Module):
	def __init__(
		self,
		seq_len,
		embed_dim,
		num_heads,
		dim_feedforward,
		num_layers,
		num_covars,
		num_phenos = 1,
		mean_imputed = False,
		kernel_size = None,
		dilation = 1,
		dlm_reprs = None):
		super().__init__()

		self.seq_len_ = seq_len
		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads
		self.attn_ffw_dim_ = dim_feedforward
		self.num_layers_ = num_layers
		self.dlm_reprs = dlm_reprs
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.mean_imputed_ = mean_imputed
		self.kernel_size_ = kernel_size
		self.dilation_ = dilation
		self._init_modules()

	def _init_modules(self):
		self.snp_embedding = nn.Embedding(
			num_embeddings = self.seq_len_, 
			embedding_dim = self.embed_dim_, 
			padding_idx = None)

		"""
		self.covar_embeddings = []
		if self.covars_embed_type_ is not None:
			for curr_type in self.covars_embed_type_:
				if "tabular" in curr_type:
					n_embed = int(curr_type.split('_')[-1])
					self.covar_embeddings.append(nn.Embedding(
															num_embeddings = n_embed, 
															embedding_dim = self.embed_dim_, 
															padding_idx = None
															))
				
				elif curr_type == 'linear_no_bias':
					self.covar_embeddings.append(nn.Linear(1,self.embed_dim_, bias = False))
				elif curr_type == 'linear_with_bias':
					self.covar_embeddings.append(nn.Linear(1,self.embed_dim_, bias = True))
		"""
		self.dose_embedding = None
		if not self.mean_imputed_:
			self.dose_embedding = nn.Embedding(
											num_embeddings = 4, #doses: 0,1,2,3 (0:NaN,1:0, 2:1, 3:2)
											embedding_dim = self.embed_dim_, 
											padding_idx = None
											)

		if self.dlm_reprs is not None:
			self.dlm_fc = nn.Linear(768,self.embed_dim_, bias = False)
		
		self.transformer_encoder = nn.ModuleList(
			[
				TransformerLayer(
					embed_dim = self.embed_dim_,
					num_heads = self.num_heads_,
					dim_feedforward = self.attn_ffw_dim_,
					kernel_size = self.kernel_size_,#this activates natten if not None
					dilation = self.dilation_,
					) for _ in range(self.num_layers_)
			])

		self.pheno_head = PhenoHead(
			embed_dim = self.embed_dim_,
			seq_len = self.seq_len_,
			num_covars = self.num_covars_,
			num_phenos = self.num_phenos_)

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)
		
		
		#Learnable feature embeddings
		x = torch.arange(self.seq_len_, device = doses.device)
		x = x.unsqueeze(0).repeat(doses.shape[0], 1)
		x = self.feature_embedding(x) #B x seq_len x embed_dim

		#DLM embeddings
		if self.dlm_reprs is not None:
			dlm_embd = self.dlm_reprs.unsqueeze(0).repeat(doses.shape[0], 1, 1).to('cuda')
			dlm_embd = nn.Parameter(dlm_embd, requires_grad = False)
			dlm_embd = self.dlm_fc(dlm_embd) #B x seq_len x embed_dim
			x = x + dlm_embd
		
		#Dose embedding
		if self.dose_embedding:
			doses = self.dose_embedding(doses+1)#added 1 to shift -1 (NaN) to 0
		else:
			doses = doses.unsqueeze(-1)

		#doses = x + doses
		x = x * doses #B x seq_len x embed_dim

		for layer in self.transformer_encoder:
			x = layer(x = x)

		"""
		if self.use_gender_:
			sex = sex.long()
			sex = self.gender_embedding(sex)
			sex = sex.unsqueeze(dim = 1)
			x = torch.cat((x,sex), dim = 1)

		for layer in self.transformer_encoder:
			x = layer(x = x)
		"""
		covars = covars if covars else []
		ret = self.pheno_head(doses = x, covars = covars)

		return ret


class g2p_transformer_ExplicitNaNDose(nn.Module):
	def __init__(
		self,
		seq_len,
		embed_dim,
		num_heads,
		dim_feedforward,
		num_layers,
		num_covars,
		num_phenos = 1,
		kernel_size = None,
		dilation = 1,
		dlm_reprs = None,
		weight_the_loss = False):
		super().__init__()

		self.seq_len_ = seq_len
		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads
		self.attn_ffw_dim_ = dim_feedforward
		self.num_layers_ = num_layers
		self.dlm_reprs = dlm_reprs
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.kernel_size_ = kernel_size
		self.dilation_ = dilation
		self.weight_the_loss_ = weight_the_loss
		self._init_modules()

	def _init_modules(self):
		self.snp_embedding = nn.Embedding(
			num_embeddings = 4 * self.seq_len_,#0,1,2,NaN doses X num_SNPs 
			embedding_dim = self.embed_dim_, 
			padding_idx = None)


		if self.dlm_reprs is not None:
			self.dlm_fc = nn.Linear(768,self.embed_dim_, bias = False)
		
		print("Number of layers is {}".format(self.num_layers_))
		self.transformer_encoder = nn.ModuleList(
			[
				TransformerLayer(
					embed_dim = self.embed_dim_,
					num_heads = self.num_heads_,
					dim_feedforward = self.attn_ffw_dim_,
					kernel_size = self.kernel_size_,#this activates natten if not None
					dilation = self.dilation_,
					) for _ in range(self.num_layers_)
			])
		
		"""Sep pheno heads
		self.pheno_heads = nn.ModuleList([PhenoHead_FC_on_flatten(
				embed_dim=self.embed_dim_,
				seq_len=self.seq_len_,
				num_covars = self.num_covars_,
				num_phenos = 1) for _ in range(self.num_phenos_)])
		"""
		
		#"""Unified pheno heads
		self.pheno_head = PhenoHead_FC_on_flatten(
				embed_dim=self.embed_dim_,
				seq_len=self.seq_len_,
				num_covars = self.num_covars_,
				num_phenos = self.num_phenos_)
		#"""
		
		self.log_sigma_2 = None
		if self.weight_the_loss_ and self.num_phenos_ > 1:
			self.log_sigma_2 = nn.Parameter(torch.ones(self.num_phenos_, requires_grad=True))

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)		
		
		#Learnable feature embeddings for (SNP,dose) pair
		## convert doses to 4 x col_idx + (doses + 1)
		doses = doses + 1 #To shift -1 (NaN) to 0
		R,C = doses.shape
		device = doses.device
		tmp = torch.arange(0, C * 4, 4, device = device).repeat(R, 1)
		doses = doses + tmp
		doses = doses.long().to(device)
		doses = self.snp_embedding(doses) #B x seq_len x embed_dim

		#DLM embeddings
		if self.dlm_reprs is not None:
			dlm_embd = self.dlm_reprs.unsqueeze(0).repeat(doses.shape[0], 1, 1).to('cuda')
			dlm_embd = nn.Parameter(dlm_embd, requires_grad = False)
			dlm_embd = self.dlm_fc(dlm_embd) #B x seq_len x embed_dim
			doses += dlm_embd
		
		for tmp,layer in enumerate(self.transformer_encoder):
			doses = layer(x = doses)

			
		covars = covars if covars is not None else []
		
		"""Sep Pheno heads
		if self.num_phenos_ == 1:
			return self.pheno_heads[0](doses = doses, covars = covars)
		else:
			#ret = [layer(doses = doses, covars = covars) for layer in self.pheno_heads]
			#return torch.cat(ret, axis = 1)
			return torch.cat([head(doses = doses, covars = covars) for head in self.pheno_heads], dim=1)  
		"""
		
		#"""Unified pheno heads
		return self.pheno_head(doses = doses, covars = covars)
		#"""




class g2p_transformer_ExplicitNaNDose2_withGlobalAtten(nn.Module):
	def __init__(
		self,
		seq_len,
		embed_dim,
		num_heads,
		dim_feedforward,
		num_layers,
		num_covars,
		num_phenos = 1,
		kernel_size = None,
		dilation = 1,
		dlm_reprs = None,
		weight_the_loss = False):
		super().__init__()

		self.seq_len_ = seq_len
		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads
		self.attn_ffw_dim_ = dim_feedforward
		self.num_layers_ = num_layers
		self.dlm_reprs = dlm_reprs
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.kernel_size_ = kernel_size
		self.dilation_ = dilation
		self.weight_the_loss_ = weight_the_loss
		self._init_modules()

	def _init_modules(self):
		## Define #phenos CLS tokens
		self.snp_embedding = nn.Embedding(
			num_embeddings = 2 * self.seq_len_ + self.num_phenos_, #NaN | non-NaN doses X num_SNPs + CLS (x#phenos)
			embedding_dim = self.embed_dim_, 
			padding_idx = None)


		if self.dlm_reprs is not None:
			self.dlm_fc = nn.Linear(768,self.embed_dim_, bias = False)
		
		print("Number of layers is {}".format(self.num_layers_))
		self.transformer_encoder = nn.ModuleList(
			[
				TransformerLayer_WithCLSToken(
					embed_dim = self.embed_dim_,
					num_heads = self.num_heads_,
					dim_feedforward = self.attn_ffw_dim_,
					kernel_size = self.kernel_size_,#this activates natten if not None
					dilation = self.dilation_,
					nCLS = self.num_phenos_
					) for _ in range(self.num_layers_)
			])
		
		"""Sep pheno heads
		self.pheno_heads = nn.ModuleList([PhenoHead_FC_on_flatten(
				embed_dim=self.embed_dim_,
				seq_len=self.seq_len_,
				num_covars = self.num_covars_,
				num_phenos = 1) for _ in range(self.num_phenos_)])
		"""
		
		#"""Unified pheno heads
		self.pheno_head = PhenoHead_FC_on_flatten(
				embed_dim=self.embed_dim_,
				seq_len=self.num_phenos_,# number of CLS tokens
				num_covars = self.num_covars_,
				num_phenos = self.num_phenos_)
		#"""
		
		self.log_sigma_2 = None
		if self.weight_the_loss_ and self.num_phenos_ > 1:
			self.log_sigma_2 = nn.Parameter(torch.ones(self.num_phenos_, requires_grad=True))

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)		
		
		#Learnable feature embeddings, 0.5 parameters compared to g2p_transformer_ExplicitNaNDose
		R,C = doses.shape
		device = doses.device
		mask_non_nan = ~doses.eq(-1)

		""" a better implementation below
		#col_indices = torch.arange(C).unsqueeze(0).expand(R, -1)
		#embds = torch.where(doses == -1, 2 * col_indices, 2 * col_indices + 1)
		#embds = embds.long().to(device)
		"""
		embds = 2 * torch.arange(C, device = device) + (doses != -1).long()
		cls_tokens = torch.arange(start = 2*C, end = 2*C + self.num_phenos_, device = device).repeat(R,1)
		cls_mask = torch.zeros_like(cls_tokens, dtype = torch.bool, device = device)
		embds = torch.cat([embds, cls_tokens], dim = 1)
		mask_non_nan = torch.cat([mask_non_nan, cls_mask], dim = 1)

		
		embds = self.snp_embedding(embds) #B x seq_len x embed_dim
		doses = doses.unsqueeze(-1)
		cls_fake_doses = torch.ones((R,self.num_phenos_,1), dtype = doses.dtype, device = device)
		doses = torch.cat([doses, cls_fake_doses], dim = 1)
		embds[mask_non_nan] *= doses[mask_non_nan]
		doses = embds

		#DLM embeddings
		if self.dlm_reprs is not None:
			dlm_embd = self.dlm_reprs.unsqueeze(0).repeat(doses.shape[0], 1, 1).to('cuda')
			dlm_embd = nn.Parameter(dlm_embd, requires_grad = False)
			dlm_embd = self.dlm_fc(dlm_embd) #B x seq_len x embed_dim
			doses += dlm_embd

		
		for tmp,layer in enumerate(self.transformer_encoder):
			doses = layer(x = doses)
			
		covars = covars if covars is not None else []
		
		"""Sep Pheno heads
		if self.num_phenos_ == 1:
			return self.pheno_heads[0](doses = doses, covars = covars)
		else:
			#ret = [layer(doses = doses, covars = covars) for layer in self.pheno_heads]
			#return torch.cat(ret, axis = 1)
			return torch.cat([head(doses = doses, covars = covars) for head in self.pheno_heads], dim=1)  
		"""
		
		#"""Unified pheno heads
		return self.pheno_head(doses = doses[:,-self.num_phenos_:,:], covars = covars)
		#"""



class g2p_transformer_ExplicitNaNDose2(nn.Module):
	def __init__(
		self,
		seq_len,
		embed_dim,
		num_heads,
		dim_feedforward,
		num_layers,
		num_covars,
		num_phenos = 1,
		kernel_size = None,
		dilation = 1,
		dlm_reprs = None,
		weight_the_loss = False,
		use_snp_annots = False,
		snp_indices = None):
		super().__init__()

		if isinstance(dilation, int):
			dilation = [dilation for _ in range(num_layers)]
		else:
			assert(len(dilation) == num_layers)
		
		if use_snp_annots and snp_indices is None:
			raise ValueError("Using snp annotations without providing a suitable SNP subset is not recommneded!")

		self.seq_len_ = seq_len
		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads
		self.attn_ffw_dim_ = dim_feedforward
		self.num_layers_ = num_layers
		self.dlm_reprs = dlm_reprs
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.kernel_size_ = kernel_size
		self.dilation_ = dilation
		self.weight_the_loss_ = weight_the_loss
		self.use_snp_annots_ = use_snp_annots
		self.snp_indices_ = snp_indices
		self._init_modules()

	def _init_modules(self):
		self.snp_embedding = nn.Embedding(
			num_embeddings = 2 * self.seq_len_,#NaN | non-NaN doses X num_SNPs 
			embedding_dim = self.embed_dim_, 
			padding_idx = None)


		if self.dlm_reprs is not None:
			self.dlm_fc = nn.Linear(768,self.embed_dim_, bias = False)
		
		if self.use_snp_annots_:
			self.annot_fc = nn.Linear(74,self.embed_dim_, bias = False)
			self.annot_tensor = torch.load("/home/payamd/RP/data/multitask/snps_functional_annots/aiprs_aligned_annotations_fp16.pt")
			self.annot_tensor = self.annot_tensor[self.snp_indices_]
			self.annot_tensor = self.annot_tensor.detach().clone()  # Prevent modification by references
			assert not torch.any(torch.isnan(self.annot_tensor)), "Tensor contains NaN values!"
			assert self.annot_tensor.shape[0] == self.seq_len_
			

		self.transformer_encoder = nn.ModuleList(
			[
				TransformerLayer(
					embed_dim = self.embed_dim_,
					num_heads = self.num_heads_,
					dim_feedforward = self.attn_ffw_dim_,
					kernel_size = self.kernel_size_,#this activates natten if not None
					dilation = layer_dilation,
					) for layer_dilation in self.dilation_
			])
		
		"""Sep pheno heads
		self.pheno_heads = nn.ModuleList([PhenoHead_FC_on_flatten(
				embed_dim=self.embed_dim_,
				seq_len=self.seq_len_,
				num_covars = self.num_covars_,
				num_phenos = 1) for _ in range(self.num_phenos_)])
		"""
		
		#"""Unified pheno heads
		self.pheno_head = PhenoHead_FC_on_flatten(
				embed_dim=self.embed_dim_,
				seq_len=self.seq_len_,
				num_covars = self.num_covars_,
				num_phenos = self.num_phenos_)
		#"""
		
		self.log_sigma_2 = None
		if self.weight_the_loss_ and self.num_phenos_ > 1:
			self.log_sigma_2 = nn.Parameter(torch.ones(self.num_phenos_, requires_grad=True))

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)		
		
		#Learnable feature embeddings, 0.5 parameters compared to g2p_transformer_ExplicitNaNDose
		R,C = doses.shape
		device = doses.device
		mask_non_nan = ~doses.eq(-1)

		""" a better implementation below
		#col_indices = torch.arange(C).unsqueeze(0).expand(R, -1)
		#embds = torch.where(doses == -1, 2 * col_indices, 2 * col_indices + 1)
		#embds = embds.long().to(device)
		"""
		embds = 2 * torch.arange(C, device = device) + (doses != -1).long()
		
		embds = self.snp_embedding(embds) #B x seq_len x embed_dim

		if self.use_snp_annots_:
			self.annot_tensor = self.annot_tensor.to(device)
			self.annot_tensor.requires_grad_(False)  # Disable gradients
			annot_embeds = self.annot_fc(self.annot_tensor) #seq_len X embed_dim
			embds += annot_embeds #B x seq_len x embed_dim


		doses = doses.unsqueeze(-1)
		embds[mask_non_nan] *= doses[mask_non_nan]
		doses = embds

		#DLM embeddings
		if self.dlm_reprs is not None:
			dlm_embd = self.dlm_reprs.unsqueeze(0).repeat(doses.shape[0], 1, 1).to('cuda')
			dlm_embd = nn.Parameter(dlm_embd, requires_grad = False)
			dlm_embd = self.dlm_fc(dlm_embd) #B x seq_len x embed_dim
			doses += dlm_embd
		
		
		
		for tmp,layer in enumerate(self.transformer_encoder):
			doses = layer(x = doses)


		covars = covars if covars is not None else []
		
		"""Sep Pheno heads
		if self.num_phenos_ == 1:
			return self.pheno_heads[0](doses = doses, covars = covars)
		else:
			#ret = [layer(doses = doses, covars = covars) for layer in self.pheno_heads]
			#return torch.cat(ret, axis = 1)
			return torch.cat([head(doses = doses, covars = covars) for head in self.pheno_heads], dim=1)  
		"""
		
		#"""Unified pheno heads
		return self.pheno_head(doses = doses, covars = covars)
		#"""

class g2p_linear_ExplicitNaNDose2(nn.Module):
	def __init__(
		self,
		seq_len,
		embed_dim,
		num_covars,
		num_phenos = 1,
		weight_the_loss = False):
		super().__init__()

		self.seq_len_ = seq_len
		self.embed_dim_ = embed_dim
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.weight_the_loss_ = weight_the_loss
		self._init_modules()

	def _init_modules(self):
		self.snp_embedding = nn.Embedding(
			num_embeddings = 2 * self.seq_len_,#NaN | non-NaN doses X num_SNPs 
			embedding_dim = self.embed_dim_, 
			padding_idx = None)
		
		#"""Unified pheno heads
		self.pheno_head = PhenoHead_FC_on_flatten(
				embed_dim=self.embed_dim_,
				seq_len=self.seq_len_,
				num_covars = self.num_covars_,
				num_phenos = self.num_phenos_)
		#"""
		
		self.log_sigma_2 = None
		if self.weight_the_loss_ and self.num_phenos_ > 1:
			self.log_sigma_2 = nn.Parameter(torch.ones(self.num_phenos_, requires_grad=True))

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)		
		
		#Learnable feature embeddings, 0.5 parameters compared to g2p_transformer_ExplicitNaNDose
		R,C = doses.shape
		device = doses.device
		mask_non_nan = ~doses.eq(-1)

		""" a better implementation below
		#col_indices = torch.arange(C).unsqueeze(0).expand(R, -1)
		#embds = torch.where(doses == -1, 2 * col_indices, 2 * col_indices + 1)
		#embds = embds.long().to(device)
		"""
		embds = 2 * torch.arange(C, device = device) + (doses != -1).long()
		
		embds = self.snp_embedding(embds) #B x seq_len x embed_dim
		doses = doses.unsqueeze(-1)
		embds[mask_non_nan] *= doses[mask_non_nan]
		doses = embds
			
		covars = covars if covars is not None else []
		
		"""Sep Pheno heads
		if self.num_phenos_ == 1:
			return self.pheno_heads[0](doses = doses, covars = covars)
		else:
			#ret = [layer(doses = doses, covars = covars) for layer in self.pheno_heads]
			#return torch.cat(ret, axis = 1)
			return torch.cat([head(doses = doses, covars = covars) for head in self.pheno_heads], dim=1)  
		"""
		
		#"""Unified pheno heads
		return self.pheno_head(doses = doses, covars = covars)
		#"""


class g2p_linear_OneHotDoses(nn.Module):
	def __init__(
		self,
		seq_len,
		num_covars,
		num_phenos = 1,
		weight_the_loss = False):
		super().__init__()

		self.seq_len_ = seq_len
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.weight_the_loss_ = weight_the_loss
		self._init_modules()

	def _init_modules(self):
		self.FC = nn.Linear(self.seq_len_ * 2 + self.num_covars_, self.num_phenos_)
		
		self.log_sigma_2 = None
		if self.weight_the_loss_ and self.num_phenos_ > 1:
			self.log_sigma_2 = nn.Parameter(torch.ones(self.num_phenos_, requires_grad=True))

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)	
		B, L = doses.shape
		device = doses.device
		embds = torch.zeros((B, 2 * L), device = device, dtype=torch.float)
		# Mask where input is -1 (B, L)
		mask = (doses == -1)  # Shape: (B, L)
		# Fill in the first part of the encoding: (B, 2L, 1)
		embds[:, 0::2] = mask.to(torch.float)  # First position of each pair
		# Fill in the second part of the encoding (value itself for non -1)
		embds[:, 1::2] = (~mask).to(torch.float) * abs(doses.to(torch.float))
		doses = embds
		
		covars = covars if covars is not None else []
		if covars is not None and len(covars) > 0 :
			assert(self.num_covars_ == covars.shape[1])
			doses = torch.cat((doses, covars), dim = -1) # B x (2 * seq_len + sum of additional covars dims)
		
		return self.FC(doses)


class g2p_transformer_baseV2_withPooling(nn.Module):
	def __init__(
		self,
		seq_len,
		embed_dim,
		num_heads,
		dim_feedforward,
		num_layers,
		num_covars,
		num_phenos = 1,
		kernel_size = None,
		dilation = 1,
		pool_size_stride = 2,
		dlm_reprs = None):
		super().__init__()

		self.seq_len_ = seq_len
		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads
		self.attn_ffw_dim_ = dim_feedforward
		self.num_layers_ = num_layers
		self.dlm_reprs = dlm_reprs
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.kernel_size_ = kernel_size
		self.dilation_ = dilation
		self.pool_size_stride = pool_size_stride
		self._init_modules()

	def _init_modules(self):
		self.snp_embedding = nn.Embedding(
			num_embeddings = 4 * self.seq_len_,#0,1,2,NaN doses X num_SNPs 
			embedding_dim = self.embed_dim_, 
			padding_idx = None)


		if self.dlm_reprs is not None:
			self.dlm_fc = nn.Linear(768,self.embed_dim_, bias = False)


		encoder_layers = []
		for _ in range(self.num_layers_):
			encoder_layers.append(TransformerLayer(
													embed_dim = self.embed_dim_,
													num_heads = self.num_heads_,
													dim_feedforward = self.attn_ffw_dim_,
													kernel_size = self.kernel_size_,#this activates natten if not None
													dilation = self.dilation_,
													))
			
			encoder_layers.append(TransposeLayer(1,2))
			encoder_layers.append(nn.MaxPool1d(kernel_size = pool_size_stride, stride = pool_size_stride))
			encoder_layers.append(TransposeLayer(1,2))

		
		self.transformer_encoder = nn.ModuleList(
			[
				TransformerLayer(
					embed_dim = self.embed_dim_,
					num_heads = self.num_heads_,
					dim_feedforward = self.attn_ffw_dim_,
					kernel_size = self.kernel_size_,#this activates natten if not None
					dilation = self.dilation_,
					) for _ in range(self.num_layers_)
			])
		
		"""Separate pheno heads
		self.pheno_heads = nn.ModuleList([PhenoHead(
			embed_dim = self.embed_dim_,
			seq_len = self.seq_len_,
			num_covars = self.num_covars_) for _ in range(self.num_phenos_)])
		"""
		self.pheno_head = PhenoHead(
				embed_dim=self.embed_dim_,
				seq_len=self.seq_len_,
				num_covars = self.num_covars_,
				num_phenos = self.num_phenos_)

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)		
		
		#Learnable feature embeddings for (SNP,dose) pair
		## convert doses to 4 x col_idx + (doses + 1)
		doses += 1 #To shift -1 (NaN) to 0
		R,C = doses.shape
		device = doses.device
		tmp = torch.arange(0, C * 4, 4, device = device).repeat(R, 1)
		doses += tmp
		doses = doses.long().to(device)
		doses = self.snp_embedding(doses) #B x seq_len x embed_dim

		#DLM embeddings
		if self.dlm_reprs is not None:
			dlm_embd = self.dlm_reprs.unsqueeze(0).repeat(doses.shape[0], 1, 1).to('cuda')
			dlm_embd = nn.Parameter(dlm_embd, requires_grad = False)
			dlm_embd = self.dlm_fc(dlm_embd) #B x seq_len x embed_dim
			doses += dlm_embd
		
		
		for tmp,layer in enumerate(self.transformer_encoder):
			doses = layer(x = doses)

		covars = covars if covars else []
		"""Separate pheno heads
		if self.num_phenos_ == 1:
			return self.pheno_heads[0](doses = doses, covars = covars)
		else:
			ret = [layer(doses = doses, covars = covars) for layer in self.pheno_heads]
			return torch.cat(ret, axis = 1)
		"""
		return self.pheno_head(doses = doses, covars = covars)




class g2p_transformer_with_CNNTower(nn.Module):
	def __init__(
		self,
		seq_len,
		conv_kernel_sizes,
		conv_kernel_channels,
		pool_size_stride,
		embed_dim,
		num_heads,
		dim_feedforward,
		num_layers,
		num_covars,
		num_phenos = 1,
		kernel_size = None,
		dilation = 1,
		dlm_reprs = None,
		cnn_norm = 'batch',
		cnn_activation = 'silu'):
		super().__init__()

		assert(isinstance(conv_kernel_sizes, list))
		assert(isinstance(conv_kernel_channels, list))
		assert(len(conv_kernel_channels) == len(conv_kernel_sizes))
		for w in conv_kernel_sizes:
			assert(w%2) #Let's not use even kernel sizes


		self.seq_len_ = seq_len
		self.conv_w_ = conv_kernel_sizes
		self.conv_c_ = conv_kernel_channels
		self.pool_s_ = pool_size_stride
		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads
		self.attn_ffw_dim_ = dim_feedforward
		self.num_layers_ = num_layers
		self.dlm_reprs = dlm_reprs
		self.num_covars_ = num_covars
		self.num_phenos_ = num_phenos
		self.kernel_size_ = kernel_size
		self.dilation_ = dilation
		self.cnn_norm_ = cnn_norm
		self.cnn_act_ = cnn_activation
		self._init_modules()

	def _init_modules(self):
		self.snp_embedding = CNNTower(
									seq_len = self.seq_len_,
									kernel_sizes = self.conv_w_,
									kernel_channels = self.conv_c_,
									pool_size_stride = self.pool_s_,
									norm = self.cnn_norm_,
									activation = self.cnn_act_
									)

		
		self.transformer_encoder = nn.ModuleList(
			[
				TransformerLayer(
					embed_dim = self.embed_dim_,
					num_heads = self.num_heads_,
					dim_feedforward = self.attn_ffw_dim_,
					kernel_size = self.kernel_size_,#this activates natten if not None
					dilation = self.dilation_,
					) for _ in range(self.num_layers_)
			])
		

		print("CNNTower reduces sequence length to {}".format(self.snp_embedding.final_len))
		self.pheno_heads = nn.ModuleList([PhenoHead(
			embed_dim = self.embed_dim_,
			seq_len = self.snp_embedding.final_len,
			num_covars = self.num_covars_) for _ in range(self.num_phenos_)])

	
	def forward(self, doses, covars = None):
		assert(doses.shape[1] == self.seq_len_)

		#get one-hot encode, prepare for CNNTower
		doses += 1 #To shift -1 (NaN) to 0
		doses = nn.functional.one_hot(doses.long()).to(torch.bfloat16)
		doses = doses.transpose(1,2) #B x 4 x seq_len

		#SNP embedding by CNNTower
		doses = self.snp_embedding(doses)
		doses = doses.transpose(1,2) #B x seq_len x embed_dim

		"""
		#DLM embeddings
		if self.dlm_reprs is not None:
			dlm_embd = self.dlm_reprs.unsqueeze(0).repeat(doses.shape[0], 1, 1).to('cuda')
			dlm_embd = nn.Parameter(dlm_embd, requires_grad = False)
			dlm_embd = self.dlm_fc(dlm_embd) #B x seq_len x embed_dim
			doses += dlm_embd
		"""
		
		#print_memory(note = "Before Transformer layers")
		
		for _,layer in enumerate(self.transformer_encoder):
			doses = layer(x = doses)

		covars = covars if covars else []
		if self.num_phenos_ == 1:
			return self.pheno_heads[0](doses = doses, covars = covars)
		else:
			ret = [layer(doses = doses, covars = covars) for layer in self.pheno_heads]
			return torch.cat(ret, axis = 1)