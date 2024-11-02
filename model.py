import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model: int, vocab_size: int):
        super(InputEmbedding,self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        '''We use the embedding layer provided by pytorch to do that embeddding'''
        return self.embedding(x) * (self.d_model**0.5)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model :int ,max_length: int,dropout: float) -> None:
        super(PositionalEncoding,self).__init__()
        self.d_model = d_model
        self.max_length = max_length # THe maximum length of the sentence present in the vocab
        self.dropout = nn.Dropout(dropout)

        postional_encoding = torch.zeros(max_length,d_model)
        position = torch.arang(0,max_length,dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        postional_encoding[:,0::2] = torch.sin(position*div_term)
        postional_encoding[:,1::2] = torch.cos(position*div_term)

        postional_encoding = postional_encoding.unsqueeze(0) # Changing the tensor fron (max_length,d_model) to (batch_size,max_length,d_model)
        self.register_buffer('postional_encoding',postional_encoding)

    def forward(self,x):
        x = x + (self.postional_encoding[:,:x.shape[1],:]).required_grad(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 10**-6) -> None:
        super(LayerNormalization,self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean)/ (std+ self.eps) + self.beta
        
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model : int,d_ff: int,dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        # We have a input tensor with the dimenstion (batch_size,max_length,d_model) -> (batch_size,max_length,d_ff)
        x = self.linear2(self.dropout(self.linear1(x)))

class MultiHeadedAttention(nn.Module):
     def __init__(self,d_model: int,n_heads: int,dropout: float) -> None:
         super().__init__()
         self.d_model = d_model
         self.n_heads = n_heads
         self.dropout = nn.Dropout(dropout)
         assert d_model % n_heads == 0, "d_model is not divisible by n_heads"
         self.d_k = d_model // n_heads
         self.query = nn.Linear(d_model,d_model)
         self.values = nn.Linear(d_model,d_model)
         self.keys = nn.Linear(d_model,d_model)
         self.output = nn.Linear(d_model,d_model)
     
     @staticmethod
     def attention(query,key,value,mask, droput: nn.Dropout):
         
         d_k = query.shape[-1]
         attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
         if mask is not None:
             attention_score.masked_fill(mask == 0, float('-inf'))
         attention_score = attention_score.softmax(dim = -1) # (batch_size,n_heads,seq_len,seq_len)
         if dropout is not None:
             attention_score = dropout(attention_score)
         return (attention_score @ value), attention_score

     def forward(self,q,k,v,mask):
         query = self.query(q) # (batch_size,seq_len,d_model) x (d_model,d_model) -> (batch_size,seq_len,d_model)
         keys = self.keys(k) # (batch_size,seq_len,d_model) x (d_model,d_model) -> (batch_size,seq_len,d_model)
         values = self.values(v) # (batch_size,seq_len,d_model) x (d_model,d_model) -> (batch_size,seq_len,d_model)

         # (batch_size,seq_len,dmodel) -> (batch_size,seq_len,n_heads,d_k) -> (batch_size,n_heads,seq_len,d_k)
         query = query.view(query.shape[0],query.shape[1],self.n_heads,self.d_k).transpose(1,2)
         keys = keys.view(keys.shape[0],keys.shape[1],self.n_heads,self.d_k).transpose(1,2)
         values = values.view(values.shape[0],values.shape[1],self.n_heads,self.d_k).transpose(1,2)

         x,self.attention_score = MultiHeadedAttention.attention(query,keys,values,mask,self.dropout)
         x = x.transpose(1,2).contigious().view(x.shape[0],-1,self.d_model)
         return self.output(x)
class ResidualConnection(nn.Module):
    def __init__(self,droput: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block : MultiHeadedAttention,
                feed_forward_network : FeedForwardBlock,
                dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_network = feed_forward_network
        self.residual_connection = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
        ])
    def forward(self,x,src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x, self.feed_forward_network)
        return x
    
class Encoder(nn.Module):
    def __init__(self,num_layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = num_layers
        self.norm = LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock:
    def __init__(self,self_attention_block : MultiHeadedAttention,
                cross_attention_block: MultiHeadedAttention,
                feed_forward_block : FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(3)
        ])
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        '''
        x: Input to the decoder
        encoder_output: Output of the encoder layer
        src_mask: Mask applied to the encoder
        tgt_mask: Mask applied to the decoder
        '''
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x,self.feed_forward_block)
        return x
class Decoder(nn.Module):
    def __init__(self,layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self,d_model: int,vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim = -1)
    
class Transformer(nn.Module):
    def __init__(self,encoder: Encoder,decoder: Decoder,src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                src_pos : PositionalEncoding, tgt_pos : PositionalEncoding, project:ProjectionLayer) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.project = project

    def encoder(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decoder(self,encoder_output,src_mask,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decode(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.project(x)

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_max_length:int,tgt_max_length:int,
                     d_model: int = 512,n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model,tgt_vocab_size)

    src_pos_encoding = PositionalEncoding(d_model,src_max_length,dropout)
    tgt_pos_encoding = PositionalEncoding(d_model,tgt_max_length,dropout)

    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadedAttention(d_model,n_heads,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = Encoder(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention = MultiHeadedAttention(d_model,n_heads,dropout)
        decoder_cross_attention = MultiHeadedAttention(d_model,n_heads,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = Decoder(decoder_self_attention,decoder_cross_attention,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection = ProjectionLayer(d_model,tgt_vocab_size)

    transformer = Transformer(encoder,decoder,src_embedding,tgt_embedding,src_pos_encoding,tgt_pos_encoding,projection)

    for p in transformer.parameters:
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return transformer
        
    
