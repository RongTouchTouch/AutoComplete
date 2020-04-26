import torch.nn as nn
from Sublayers import SublayerConnection, SublayerConnection4KG
from Utils import clones


# 每层有两个子层:第一层是多头自注意机制，第二层是一个简单的、位置导向的、全连接的前馈网络。
class EncoderLayer(nn.Module):
	"编码器由以下的自注意力和前馈网络组成"

	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	#输入为数据及对应的mask，因为encoder中不需要掩盖任何输入token，所以mask一般为[1，1...1]
	#self_attn即multiheadattention，前三个输入为Query，Key，Value。其中Query可以理解为查询语句，Key-Value对可以理解为键值对，
	#最后一个参数mask，在decoder中会产生作用，因为在decoder中mask不全为1
	def forward(self, x, mask):
		"按照论文中的图1（左）的方式进行连接"
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)


# 在每个编码器层中的两个子层外，解码器还插入第三个子层，该子层在编码器堆栈的输出上执行多头关注。与编码器类似，使用残差连接解码器的每个子层，然后进行层归一化。
class DecoderLayer(nn.Module):
	"解码器由以下的自注意力、源注意力和前馈网络组成"

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"按照论文中的图1（右）的方式进行连接"
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)


class EncoderLayer4KG(nn.Module):
	def __init__(self, size, intermediate_size, self_attn, self_attn_ent, feed_forward, dropout):
		super(EncoderLayer4KG, self).__init__()
		self.attention = self_attn
		self.attention_ent = self_attn_ent
		self.sublayer = SublayerConnection4KG(size, intermediate_size, dropout)
		self.feed_forward = feed_forward
		self.size = size

	def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent=None, ent_mask=None):
		hidden_states, hidden_states_ent = hidden_states, hidden_states_ent
		# print("hidden_states.shape: ", hidden_states.shape)
		hidden_states = self.attention(hidden_states, hidden_states, hidden_states, attention_mask)
		# print(hidden_states_ent.shape)
		hidden_states_ent = self.attention_ent(hidden_states_ent, hidden_states_ent, hidden_states_ent, attention_mask_ent)
		# TODO
		# hidden_states_ent = hidden_states_ent  # * ent_mask
		return self.sublayer(hidden_states, hidden_states_ent)
