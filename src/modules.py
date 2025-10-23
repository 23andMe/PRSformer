import torch
import torch.nn as nn
from natten import NeighborhoodAttention1D
from natten import is_fna_enabled
import natten
import math
from utils import LayerParInit


class MultiHeadAttention_natten(nn.Module):
	def __init__(self, embed_dim, num_heads, kernel_size, dilation):
		super().__init__()
		natten.use_fused_na(True)
		if is_fna_enabled():
			print("NATTEN WILL USE FNA!")
		else:
			raise ValueError("FNA not working!")

		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads
		self.kernel_size_ = kernel_size
		self.dilation_ = dilation

		self._init_modules()

	def _init_modules(self):
		self.mha = NeighborhoodAttention1D(
			dim = self.embed_dim_ ,
			num_heads = self.num_heads_,
			kernel_size = self.kernel_size_,
			dilation = self.dilation_,
			is_causal = False,
			rel_pos_bias = False,
			qkv_bias = True,
			qk_scale = None,#uses default scale
			attn_drop = 0.0,
			proj_drop = 0.0,
			)
		

	def forward(self, x):
		"""
		x: B x L x E
		"""
		attn_output = self.mha(x = x)
		return attn_output


class MultiHeadAttention_torch(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super().__init__()
		self.embed_dim_ = embed_dim
		self.num_heads_ = num_heads

		self._init_modules()

	def _init_modules(self):
		self.mha = nn.MultiheadAttention(
			embed_dim = self.embed_dim_, 
			num_heads = self.num_heads_, 
			dropout = 0.0, 
			bias = True, 
			add_bias_kv = False, 
			add_zero_attn = False, 
			kdim = None, 
			vdim = None, 
			batch_first = True)
		

	def forward(self, query, key, value):
		attn_output, _ = self.mha(
			query = query, 
			key = key, 
			value = value, 
			key_padding_mask = None, 
			need_weights = False, 
			attn_mask = None, 
			average_attn_weights = False, 
			is_causal = False)

		return attn_output

class TransformerLayer_WithCLSToken(nn.Module):
	def __init__(
		self,
		embed_dim,
		num_heads,
		dim_feedforward,
		kernel_size,
		dilation = 1,
		nCLS = 1):
		super().__init__()


		self.local_MultiheadAttention = MultiHeadAttention_natten(
			embed_dim = embed_dim, 
			num_heads = num_heads, 
			kernel_size = kernel_size, 
			dilation = dilation
		)
		self.global_MultiheadAttention = MultiHeadAttention_torch(
			embed_dim = embed_dim, 
			num_heads = num_heads,)
		
		self.pre_atten_LayerNorm = nn.LayerNorm(embed_dim)
		self.pre_fc_LayerNorm = nn.LayerNorm(embed_dim)
		self.fc1 = nn.Linear(embed_dim, dim_feedforward)
		self.fc2 = nn.Linear(dim_feedforward, embed_dim)
		self.nCLS = nCLS
	
	def forward(self,x):

		res = x
		x = self.pre_atten_LayerNorm(x)
		x_local = self.local_MultiheadAttention(x = x[:,:-self.nCLS,:])
		tmp_key = torch.cat([x_local, x[:,-self.nCLS:,:]], dim = 1)
		x_cls = self.global_MultiheadAttention(query = x[:,-self.nCLS:,:], key = tmp_key, value = tmp_key)
		x = torch.cat([x_local, x_cls], dim = 1)
		x = res + x

		res = x
		x = self.pre_fc_LayerNorm(x)
		x = nn.functional.silu(self.fc1(x))
		x = self.fc2(x)
		x = res + x

		return x

class TransformerLayer(nn.Module):
	def __init__(
		self,
		embed_dim,
		num_heads,
		dim_feedforward,
		kernel_size = None,
		dilation = 1):
		super().__init__()

		if not kernel_size:
			self.MultiheadAttention = MultiHeadAttention_torch(
				embed_dim=embed_dim,
				num_heads=num_heads
				)
			self.use_natten_ = False
		else:
			self.MultiheadAttention = MultiHeadAttention_natten(
				embed_dim = embed_dim, 
				num_heads = num_heads, 
				kernel_size = kernel_size, 
				dilation = dilation
			)
			self.use_natten_ = True

		self.pre_atten_LayerNorm = nn.LayerNorm(embed_dim)
		self.pre_fc_LayerNorm = nn.LayerNorm(embed_dim)
		self.fc1 = nn.Linear(embed_dim, dim_feedforward)
		self.fc2 = nn.Linear(dim_feedforward, embed_dim)
		LayerParInit(self.fc1, type = "xavier_normal")
		LayerParInit(self.fc2, type = "xavier_normal")
	
	def forward(self,x):

		res = x
		x = self.pre_atten_LayerNorm(x)
		if self.use_natten_:
			x = self.MultiheadAttention(x = x)
		else:
			x = self.MultiheadAttention(query = x, key = x, value = x)
		x = res + x

		res = x
		x = self.pre_fc_LayerNorm(x)
		x = nn.functional.gelu(self.fc1(x))
		x = self.fc2(x)
		x = res + x

		return x


class CNNTower(nn.Module):
	"""
	Applies user-defined number of conv+MaxPool layers with specified params. 
	** The first layer is applied without normalization and non-linearity
	** Sequences are padded to maintain the sequence length after convolution
	** MaxPool layers use the same size and stride for length reduction
	** All convolutions use stride of 1 and dilation of 1
	"""
	def __init__(
		self,
		seq_len,
		kernel_sizes,
		kernel_channels,
		pool_size_stride = 2,
		norm = 'batch',
		activation = 'silu'):
		super().__init__()

		layers = []
		layers += [nn.Conv1d(in_channels = 4, out_channels = kernel_channels[0], kernel_size = kernel_sizes[0], padding = (kernel_sizes[0]-1)//2)]
		LayerParInit(layers[-1], type = "xavier_normal")
		layers += [nn.MaxPool1d(kernel_size = pool_size_stride, stride = pool_size_stride)]

		in_channel = kernel_channels[0]
		curr_len = self._dim_after_conv_pool(seq_len, kernel_sizes[0], (kernel_sizes[0]-1)//2, pool_size_stride)
		if len(kernel_channels) > 1:
			print("CNNTower has at least 2 layers!")
			for out_channel, w in zip(kernel_channels[1:], kernel_sizes[1:]):
				layers.append(ConvPoolBlock(in_channels = in_channel, 
											out_channels = out_channel, 
											conv_kernel_size = w, 
											pool_size_stride = pool_size_stride, 
											norm = norm, 
											activation = activation))
				in_channel = out_channel
				curr_len = self._dim_after_conv_pool(curr_len, w, (w-1)//2, pool_size_stride)
		
		self.layers = nn.ModuleList(layers)
		self.final_len = curr_len
	
	def _dim_after_conv_pool(self, L, w, p, s):
		""" calculate dim after one block of conv1d + Pool (assuming convolution stride of 1)
		L: input dim (i.e. sequence length) to block
		w: conv kernel size
		p: conv padding size
		s: pooling size and tride
		"""
		L_conv = L + 2 * p - w + 1
		L_out = math.floor((L_conv - s) / s + 1)
		return L_out
	
	def forward(self, doses):
		"""
		doses : B x E x L

		Where E is equal to 4 (Nan,0,1,2 doses)
		"""
		for layer in self.layers:
			doses = layer(doses)
		return doses

activations_dict = {
	"selu": nn.functional.selu,
	"celu": nn.functional.celu,
	"gelu": nn.functional.gelu,
	"silu": nn.functional.silu,  # Also known as Swish
}

class ConvPoolBlock(nn.Module):
	# Sequences are padded to maintain the sequence length after convolution
	def __init__(self, in_channels, out_channels, conv_kernel_size, pool_size_stride, norm = 'batch', activation = 'silu'):
		super().__init__()
		self.norm_type = norm
		if norm == 'batch':
			self.norm = nn.BatchNorm1d(in_channels) 
		elif norm == 'layer':
			self.norm = nn.LayerNorm(in_channels)
		else:
			raise ValueError("Not implemented!")
		
		assert(activation in activations_dict), "Specified activation function is not present in the activation dictionary"
		self.act_ = activations_dict[activation]
		self.conv = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = conv_kernel_size, padding = (conv_kernel_size-1)//2)
		LayerParInit(self.conv, type = "xavier_normal")
		self.pool = nn.MaxPool1d(kernel_size = pool_size_stride, stride = pool_size_stride)
	
	def forward(self, doses):
		"""
		doses : B x E x L
		"""
		if self.norm_type == 'layer':
			doses = doses.transpose(1,2)
			doses = self.norm(doses)
			doses = doses.transpose(1,2)
		elif self.norm_type == 'batch':
			doses = self.norm(doses)

		doses =  self.pool(self.conv(self.act_(doses)))
		return doses


class TransposeLayer(nn.Module):
	def __init__(self, dim0, dim1):
		super().__init__()
		self.dim0 = dim0
		self.dim1 = dim1

	def forward(self, x):
		return x.transpose(self.dim0, self.dim1)


class PhenoHead(nn.Module):
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0,
		num_phenos = 1):
		super().__init__()
		self.ln = nn.LayerNorm([embed_dim, seq_len])
		self.num_covars_ = num_covars
		self.fc1 = nn.Linear(seq_len, 1)
		self.fc2 = nn.Linear(embed_dim + num_covars, num_phenos)
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning!
		doses = doses.transpose(1,2) #B x embed_dim x L
		doses = self.ln(doses) # self.bn(doses)
		doses = nn.functional.leaky_relu(self.fc1(doses)).squeeze(-1) #B x embed_dim
		if covars is not None and len(covars) > 0 :
			assert(self.num_covars_ == covars.shape[1])
			doses = torch.cat((doses, covars), dim = -1) # B x (embed_dim + sum of additional covars dims)
		return self.fc2(doses) #B x num_phenos --> logit


class PhenoHead_FC_on_GlobalAvgPool(nn.Module):
	def __init__(
		self,
		embed_dim,
		seq_len,#Not needed
		num_covars = 0,
		num_phenos = 1):
		super().__init__()
		self.ln = nn.LayerNorm(embed_dim)
		self.num_covars_ = num_covars
		self.fc1 = nn.Linear(embed_dim + num_covars, num_phenos)
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning
		
		doses = self.ln(doses) # self.bn(doses)
		doses = doses.mean(dim = 1)
		if covars is not None and len(covars) > 0 :
			assert(self.num_covars_ == covars.shape[1])
			doses = torch.cat((doses, covars), dim = -1) # B x (embed_dim * seq_len + sum of additional covars dims)
		return self.fc1(doses) #B x num_phenos --> logit


class PhenoHead_FC_on_flatten(nn.Module):
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0,
		num_phenos = 1):
		super().__init__()
		self.ln = nn.LayerNorm(embed_dim)
		self.num_covars_ = num_covars
		self.fc1 = nn.Linear(seq_len * embed_dim + num_covars, num_phenos)
		LayerParInit(self.fc1, type = "xavier_normal")
		
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning
		
		doses = self.ln(doses) # self.bn(doses)
		doses = torch.flatten(doses, start_dim = 1)
		if covars is not None and len(covars) > 0 :
			assert(self.num_covars_ == covars.shape[1])
			doses = torch.cat((doses, covars), dim = -1) # B x (embed_dim * seq_len + sum of additional covars dims)
		return self.fc1(doses) #B x num_phenos --> logit

class PhenoHead_FC_on_flatten_maxpooled(nn.Module):
	import math
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0,
		num_phenos = 1):
		super().__init__()
		self.ln = nn.LayerNorm(embed_dim)
		self.num_covars_ = num_covars
		self.mp = nn.MaxPool1d(kernel_size = 64, stride = 64, padding=0, dilation=1, ceil_mode = True)#ceil mode makes sure each element is covered, automatically pads the right end if needed!
		pooled_len = math.ceil((seq_len - 64)/64 + 1)
		self.fc1 = nn.Linear(pooled_len * embed_dim, num_phenos)
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning
		
		doses = self.ln(doses) # self.bn(doses)
		doses = doses.transpose(1,2) #B x embed_dim x L
		doses = torch.flatten(self.mp(doses), start_dim = 1)
		if covars is not None and len(covars) > 0 :
			assert(self.num_covars_ == covars.shape[1])
			doses = torch.cat((doses, covars), dim = -1) # B x (embed_dim * seq_len + sum of additional covars dims)
		return self.fc1(doses) #B x num_phenos --> logit


class PhenoHead_FC_on_flatten_maxpooled_nonLin(nn.Module):
	import math
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0,
		num_phenos = 1):
		super().__init__()
		self.ln = nn.LayerNorm(embed_dim)
		self.num_covars_ = num_covars
		self.mp = nn.MaxPool1d(kernel_size = 64, stride = 64, padding=0, dilation=1, ceil_mode = True)#ceil mode makes sure each element is covered, automatically pads the right end if needed!
		pooled_len = math.ceil((seq_len - 64)/64 + 1)
		self.fc1 = nn.Linear(pooled_len * embed_dim, 32)
		self.fc2 = nn.Linear(32, num_phenos)
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning
		
		doses = self.ln(doses) # self.bn(doses)
		doses = doses.transpose(1,2) #B x embed_dim x L
		doses = torch.flatten(self.mp(doses), start_dim = 1)
		if covars is not None and len(covars) > 0 :
			assert(self.num_covars_ == covars.shape[1])
			doses = torch.cat((doses, covars), dim = -1) # B x (embed_dim * seq_len + sum of additional covars dims)
		return self.fc2(nn.functional.leaky_relu(self.fc1(doses))) #B x num_phenos --> logit
	
	

class PhenoHead_HidNeurons(nn.Module):
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0,
		num_phenos = 1):
		super().__init__()
		self.ln = nn.LayerNorm([embed_dim, seq_len])
		self.num_covars_ = num_covars
		self.fc1 = nn.Linear(seq_len, 16)
		self.fc2 = nn.Linear(embed_dim * 16 + num_covars, num_phenos)
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning! for now we experiment with batchnorm since the task is not language
		doses = doses.transpose(1,2) #B x embed_dim x L
		doses = self.ln(doses) # self.bn(doses)
		doses = nn.functional.leaky_relu(self.fc1(doses)).flatten(start_dim = 1) #B x embed_dim * 16
		if covars is not None and len(covars) > 0 :
			assert(self.num_covars_ == covars.shape[1])
			doses = torch.cat((doses, covars), dim = -1) # B x (embed_dim + sum of additional covars dims)
		return self.fc2(doses) #B x num_phenos --> logit


"""For Separate pheno heads
class PhenoHead(nn.Module):
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0):
		super().__init__()

		self.bn = nn.BatchNorm1d(embed_dim)
		self.num_covars_ = num_covars
		#self.ln = nn.LayerNorm(embed_dim)
		self.fc1 = nn.Linear(seq_len, 1)
		LayerParInit(self.fc1, type = "xavier_normal")
		self.fc2 = nn.Linear(embed_dim + num_covars, 1)
		LayerParInit(self.fc2, type = "xavier_normal")
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning! for now we experiment with barchnorm since the task is not language
		#doses = self.ln(doses)
		doses = doses.transpose(1,2) #B x embed_dim x L
		doses = self.bn(doses)
		doses = nn.functional.silu(self.fc1(doses)).squeeze(-1) #B x embed_dim
		if covars:
			assert(self.num_covars_ == len(covars))
			doses = [doses] + covars
			doses = torch.cat(doses, dim = -1) # B x (embed_dim + sum of additional covars dims)
		return self.fc2(doses) #B x 1 --> logit
"""

class PhenoHead2(nn.Module):
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0):
		super().__init__()

		self.bn = nn.BatchNorm1d(embed_dim)
		self.num_covars_ = num_covars
		self.fc1 = nn.Linear(seq_len, seq_len//8)
		self.fc2 = nn.Linear(seq_len//8, 1)
		LayerParInit(self.fc1, type = "xavier_normal")
		LayerParInit(self.fc2, type = "xavier_normal")
		self.fc3 = nn.Linear(embed_dim + num_covars, 1)
		LayerParInit(self.fc3, type = "xavier_normal")
	
	def forward(self, doses, covars):
		#ESM, NT and DLM would use a layernorm at the beginning! for now we experiment with barchnorm since the task is not language
		doses = doses.transpose(1,2) #B x embed_dim x L
		doses = nn.functional.leaky_relu(self.fc2(nn.functional.leaky_relu(self.fc1(doses)))).squeeze(-1) #B x embed_dim
		
		if covars:
			assert(self.num_covars_ == len(covars))
			doses = [doses] + covars
			doses = torch.cat(doses, dim = -1) # B x (embed_dim + sum of additional covars dims)
		return self.fc3(doses) #B x 1 --> logit









"""
The below code is for experiments
"""
class ConvPoolBlock_old(nn.Module):
	def __init__(self, in_channels, out_channels, conv_kernel_size, pool_kernel_size, pool_stride):
		super().__init__()
		self.conv = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = conv_kernel_size)
		self.pool = nn.MaxPool1d(kernel_size = pool_kernel_size, stride = pool_stride)
	
	def forward(self, doses):
		"""
		doses : B x E x L
		"""
		doses =  nn.functional.silu(self.pool(self.conv(doses)))
		return doses

class CNNPhenoHead_17(nn.Module):
	def __init__(
		self,
		embed_dim,
		seq_len,
		num_covars = 0,
		num_phenos = 1):
		super().__init__()
		assert num_covars == 0, "Error: Additional covariates not supported with CNNPhenoHead!"
		self.ln = nn.LayerNorm(embed_dim)
	   
		#we can later take these values as input! number of elemenst corresponds to the number of cnn+pooling  blocks
		self.kernel_channels_ = [64,16]
		self.kernel_sizes_ = [255,255]
		self.pool_sizes_ = [255,255]
		self.pool_strides_ = [85,85]

		self.conv_pool_blocks = []
		in_channel = embed_dim
		in_len = seq_len
		for out_channel, k, p, s in zip(self.kernel_channels_, self.kernel_sizes_, self.pool_sizes_, self.pool_strides_):
			self.conv_pool_blocks.append(ConvPoolBlock_old(in_channels = in_channel, 
													   out_channels = out_channel, 
													   conv_kernel_size = k, 
													   pool_kernel_size = p, 
													   pool_stride = s,
													   ))
			in_channel = out_channel
			in_len = self._dim_after_conv_pool(in_len, k, p, s)
		
		self.conv_pool_blocks = nn.ModuleList(self.conv_pool_blocks)
		
		flatten_dim = in_len * out_channel #for these numbers it should be: 115 * 16
		self.fc = nn.Linear(flatten_dim, num_phenos)
	
	def _dim_after_conv_pool(self, L, k, p, s):
		""" calculated dim after one block of conv1d + Pool (assuming convolution stride of 1)
		L: input dim to block
		k: conv kernel size
		p: pooling kernel size
		s: pooling stride
		"""
		L_conv = L - k + 1
		L_out = math.floor((L_conv - p) / s + 1)
	
	def forward(self, doses, covars):
		#ignoring covars for now
		doses = self.ln(doses) #apply on the last (embedding dimension) only
		doses = self.conv_pool_blocks(doses.transpose(1, 2))
		return self.fc(torch.flatten(doses)) #logits of size B x num_phenos
