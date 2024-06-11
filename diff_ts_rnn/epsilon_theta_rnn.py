import math
import torch
from torch import nn
import torch.nn.functional as F

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=256):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(dim, max_steps), persistent=False)
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="zeros",
        )
        self.diffusion_projection = nn.Linear(hidden_size * 256, residual_channels) 
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, diffusion_step):


       
        diffusion_step = diffusion_step.view(diffusion_step.size(0), -1)
        diffusion_step = self.diffusion_projection(diffusion_step)


       
        diffusion_step = diffusion_step.unsqueeze(-1).expand(-1, -1, x.size(2))
   

        y = x + diffusion_step
  
        y = self.dilated_conv(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip

class QuantileEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding = nn.Linear(1, embedding_dim)

    def forward(self, quantiles):
        return self.embedding(quantiles)

class EpsilonThetaRNN(nn.Module):
    def __init__(
        self,
        num_classes=5,
        time_emb_dim=32,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=32,
        class_emb_dim=32,
        target_dim=1,
        rnn_hidden_size=32,  
        quantile_embedding_dim=32 
    ):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, class_emb_dim)
        self.input_projection = nn.Conv1d(1 + class_emb_dim, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(time_emb_dim, proj_dim=residual_hidden)
        self.rnn = nn.LSTM(
            input_size=residual_hidden,
            hidden_size=rnn_hidden_size,
            num_layers=2,  
            batch_first=True,
            dropout=0.5  
        )
        self.quantile_embedding = QuantileEmbedding(quantile_embedding_dim)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                residual_channels=residual_channels,
                dilation=1,
                hidden_size=residual_hidden,
            ) for _ in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3, padding=1)
        self.output_projection = nn.Conv1d(residual_channels, target_dim, 3, padding=1)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, inputs, time, class_labels, quantiles):

        class_embeddings = self.class_embedding(class_labels)  
   
        class_embeddings = class_embeddings.unsqueeze(2).expand(-1, -1, inputs.size(2))
        inputs = torch.cat([inputs, class_embeddings], dim=1)
  
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)
 
        
        diffusion_step = self.diffusion_embedding(time)

        rnn_input = diffusion_step.unsqueeze(1) 
     
        rnn_output, _ = self.rnn(rnn_input)
        diffusion_step = rnn_output.squeeze(1) 


        quantile_emb = self.quantile_embedding(quantiles.unsqueeze(-1)).squeeze(1)  
     
        quantile_emb = quantile_emb.unsqueeze(2).expand(-1, -1, x.size(2))  
   


        diffusion_step = diffusion_step.unsqueeze(-1).expand(-1, -1, x.size(2))
     

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step + quantile_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)

        return x

