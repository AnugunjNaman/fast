from typing import Optional
import torch
import torch.nn as nn
from config import ASTConfig
import torch.nn.functional as F
from thop import profile

class ASTPatchEmbeddings(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        patch_size = config.patch_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = nn.Conv2d(1, config.hidden_size, kernel_size=(patch_size, patch_size), stride=(frequency_stride, time_stride))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(x).flatten(2).transpose(1, 2)
        return embeddings
    

class ASTEmebeddings(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ASTPatchEmbeddings(config)

        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        num_patches = frequency_out_dimension * time_out_dimension
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
    
    def get_shape(self, config: ASTConfig):
        # see Karpathy's cs231n blog on how to calculate the output dimensions
        # https://cs231n.github.io/convolutional-networks/#conv
        frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1
        return frequency_out_dimension, time_out_dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        embeddings = self.patch_embeddings(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    

class ASTSelfAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        attention_probs = torch.ones(2, 2)
        outputs = (context_layer, attention_probs)
        return outputs
    
class ASTSelfOutput(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class ASTAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.attention = ASTSelfAttention(config)
        self.output = ASTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor):
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ASTIntermediate(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class ASTOutput(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states
    
class ASTLayer(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.seq_len_dim = 1
        self.attention = ASTAttention(config)
        self.intermediate = ASTIntermediate(config)
        self.output = ASTOutput(config)
        #self.moe = MoELayer(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states))
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in AST, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        # layer_output, aux_loss = self.moe(layer_output)
        # layer_output = layer_output + hidden_states

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs

        return outputs, 0.0

class ASTEncoder(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ASTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor):
        all_hidden_states = () 
        all_self_attentions = ()

        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs, aux_loss = layer_module(hidden_states)
            hidden_states = layer_outputs[0]
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

        all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None), aux_loss
        
class ASTModel(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = ASTEmebeddings(config)
        self.encoder = ASTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def get_input_embeddings(self) -> ASTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(self, input_values: Optional[torch.Tensor]):
        if input_values is None:
            raise ValueError("You have to specify input_values")
        
        embedding_output = self.embeddings(input_values)
        encoder_outputs, aux_loss = self.encoder(embedding_output)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2
        return (sequence_output, pooled_output) + encoder_outputs[1:], aux_loss

class ASTMLPHead(nn.Module):
    def __init__(self, config: ASTConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state

class ASTClassification(nn.Module):
    def __init__(self, config: ASTConfig):
        super().__init__()
        self.num_labels = config.num_labels
        self.ast = ASTModel(config)
        self.classifier = ASTMLPHead(config)
        self.alpha = config.alpha

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs, aux_loss = self.ast(input_values)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = aux_loss if aux_loss else 0.0
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss += loss_fct(logits.squeeze(), labels.squeeze())
            
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss += loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        
        
        outputs = (logits,) + outputs[2:]
        return ((loss,) + outputs) if loss is not None else outputs

if __name__=="__main__":
    
    x = torch.randn(2, 1, 128, 1876)
    y = torch.randint(0, 1, (2, 1)).squeeze()
    model = ASTClassification(config=ASTConfig())

    z = model(x, y)[1]
    print(z)
    flops, params = profile(model, inputs=(x,y))

    print(f"Total FLOPS: {flops}")
    print(f"Total Parameters: {params}")
