import numpy as np
import torch
from torch.nn import (
    Module,
    ModuleList,
    Embedding,
    Linear,
    TransformerEncoderLayer,
    TransformerEncoder,
    CrossEntropyLoss,
    LayerNorm,
    Dropout,
)

from torch import nn


import torch.nn.functional as F


class EmbeddingProductHead(Module):
    def __init__(self, hidden_dim=256, num_features=3, num_bins=(41, 41, 41)):
        super(EmbeddingProductHead, self).__init__()
        assert num_features == 3
        self.num_features = num_features
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.combined_bins = int(np.sum(num_bins))
        self.linear = Linear(hidden_dim, self.combined_bins * hidden_dim)
        self.act = torch.nn.Softplus()
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, emb):
        batch_size, seq_len, _ = emb.shape
        bin_emb = self.act(self.linear(emb))
        bin_emb = bin_emb.view(batch_size, seq_len, self.combined_bins, self.hidden_dim)
        bin_emb_x, bin_emb_y, bin_emb_z = torch.split(bin_emb, self.num_bins, dim=2)

        bin_emb_xy = bin_emb_x.unsqueeze(2) * bin_emb_y.unsqueeze(3)
        bin_emb_xy = bin_emb_xy.view(batch_size, seq_len, -1, self.hidden_dim)

        logits = bin_emb_xy @ bin_emb_z.transpose(2, 3)
        logits = self.logit_scale.exp() * logits.view(batch_size, seq_len, -1)
        return logits




class JetTransformerAL(Module):
    def __init__(
        self,
        original_model,
        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100
    ):
        super(JetTransformerAL, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.num_const = num_const

        # Feature embeddings are inherited from the original model
        self.feature_embeddings = original_model.feature_embeddings
        # Transformer layers are inherited from the original model
        self.layers = original_model.layers
        # Output normalization and dropout are inherited from the original model
        self.out_norm = original_model.out_norm
        self.dropout_layer = original_model.dropout

        self.flat = torch.nn.Flatten()
        # Classification head with MLP and Average Pooling

        # Classification head with MLP and Average Pooling
        self.jet_mlp = nn.Linear(hidden_dim*self.num_const, 64)  # Reduce hidden_dim to 64 per jet
      
        self.mlp1 = nn.Linear(64 * 2, 128)  # After concatenating both jet representations
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Pooling over feature dimension
        self.mlp2 = nn.Linear(1, 128)  # Hidden layer after pooling
        self.output = nn.Linear(128, 1)  # Binary classification
        
        # Criterion for binary classification
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, jet1, jet2, padding_mask1, padding_mask2):
        """
        Forward pass where jet1 and jet2 are processed separately and their representations are concatenated.
        """

        # Apply feature embeddings and transformer processing separately for each jet
        jet1_repr = self._process_jet(jet1, padding_mask1)  # Process jet1
        jet2_repr = self._process_jet(jet2, padding_mask2)  # Process jet2


        jet1_repr = self.jet_mlp(jet1_repr)  # (batch, 64)
        jet2_repr = self.jet_mlp(jet2_repr)  # (batch, 64)

        pooling='avg'
        # Concatenate the representations of jet1 and jet2
        combined_repr = torch.cat([jet1_repr, jet2_repr], dim=-1)  # Shape: [batch_size, hidden_dim * 2]
        if pooling=='avg':
            # MLP processing
            x = self.mlp1(combined_repr)
            x = torch.nn.Dropout(p=0.1)(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            #x = self.dropout_layer(x)

            # Average Pooling (across jet constituents)
            #x = x.unsqueeze(1)  # Add a dummy dimension for pooling
            #print('x before pooling')
            #print(x)
            #print(x.shape)
            x = self.avg_pool(x)
             # Perform average pooling
            #x = x.squeeze(-1)  # Remove the dummy dimension
            #print('x after pooling')
            #print(x)
            #print(x.shape)
            # Second MLP layer after pooling
            x = self.mlp2(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = torch.nn.Dropout(p=0.1)(x)
            #print('x after mpl2')
            #print(x)
            #print(x.shape)
            x=torch.nn.Flatten()(x)
            x = self.output(x)
        
        
     
        
        #print('x after output')
        #print(x)
        #print(x.shape)
        #x = torch.sigmoid(x)
        return x

    def _process_jet(self, jet_data, padding_mask):
        """
        Processes a single jet (either jet1 or jet2).
        Applies feature embeddings, transformer layers, and normalization.
        """
        batch_size, num_const, num_features = jet_data.shape  # (batch, n_const, n_features)
        jet_data[jet_data < 0] = 0  # Handle negative values

        # Apply feature embeddings for each feature dimension
        emb = self.feature_embeddings[0](jet_data[:, :, 0])  # Embedding the first feature
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet_data[:, :, i])  # Sum embeddings across features
        
        # Construct causal mask to restrict attention to preceding elements
        seq_len = jet_data.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=jet_data.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)  # Causal mask
        padding_mask = ~padding_mask  # Invert padding mask for transformer layers

        #exit()
        # Apply transformer layers
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # Normalize and apply dropout
        emb = self.out_norm(emb)
        #emb = self.dropout_layer(emb)
        #emb = self.flat(emb)
        # Take the representation of the [CLS] token or apply pooling if needed
        #jet_repr = emb.mean(dim=1)  # Taking mean pooling across all constituents
        
        return emb

    def loss(self, logits, true_bin):
        """
        Computes the loss for the given logits and ground truth binary labels.
        """
        
        #print('true bine')
        
        #print(true_bin)
        #print(true_bin.shape)
        return self.criterion(logits, true_bin)

##################################################################################


class JetTransformerALScratch(Module):
    def __init__(
        self,
        original_model,
        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100,
        jet_emb=False,
        pooling='False',
        direct='True'
        



    ):
        super(JetTransformerALScratch, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.num_const = num_const
        self.jet_emb = jet_emb
        self.pooling = pooling
        self.direct = direct
  
        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )

        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout_layer = Dropout(dropout)

        self.flat = torch.nn.Flatten()
     
        self.jet_mlp = nn.Linear(hidden_dim, 64)
        if self.direct=='False':
            self.mlp1 = nn.Linear(64 * 2, 128)
              # After concatenating both jet representations
        elif self.direct=='True':  
            self.mlp1 = nn.Linear( 64*2, 128)



        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Pooling over feature dimension
        self.mlp2 = nn.Linear(1, 128)  # Hidden layer after pooling
        if self.jet_emb=='False':
            self.output = nn.Linear(128*self.num_const, 1)  # Binary classification
        elif self.jet_emb=='True':
            self.output = nn.Linear(128, 1)  # Binary classification
        
        # Criterion for binary classification
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, jet1, jet2, padding_mask1, padding_mask2):
        """
        Forward pass where jet1 and jet2 are processed separately and their representations are concatenated.
        """

        # Apply feature embeddings and transformer processing separately for each jet
        jet1_repr = self._process_jet(jet1, padding_mask1)  # Process jet1
        jet2_repr = self._process_jet(jet2, padding_mask2)  # Process jet2

        if self.direct=='False':
            jet1_repr = self.jet_mlp(jet1_repr)  # (batch, 64)
            jet2_repr = self.jet_mlp(jet2_repr)  # (batch, 64)
        

       
        # Concatenate the representations of jet1 and jet2
        combined_repr = torch.cat([jet1_repr, jet2_repr], dim=-1)  # Shape: [batch_size, hidden_dim * 2]
        if self.pooling=='avg':
            # MLP processing
            x = self.mlp1(combined_repr)
            x = torch.nn.Dropout(p=0.1)(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            #x = self.dropout_layer(x)

            # Average Pooling (across jet constituents)
            #x = x.unsqueeze(1)  # Add a dummy dimension for pooling
            #print('x before pooling')
            #print(x)
            #print(x.shape)
            x = self.avg_pool(x)
             # Perform average pooling
            #x = x.squeeze(-1)  # Remove the dummy dimension
            #print('x after pooling')
            #print(x)
            #print(x.shape)
            # Second MLP layer after pooling
            x = self.mlp2(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = torch.nn.Dropout(p=0.1)(x)
            #print('x after mpl2')
            #print(x)
            #print(x.shape)
            if self.jet_emb=='False':
                x=torch.nn.Flatten()(x)


            x = self.output(x)
        
        elif self.pooling=='False':
            x = self.mlp1(combined_repr)
            x = torch.nn.Dropout(p=0.1)(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = self.output(x)
        
        #print('x after output')
        #print(x)
        #print(x.shape)
        #x = torch.sigmoid(x)
        return x

    def _process_jet(self, jet_data, padding_mask):
        """
        Processes a single jet (either jet1 or jet2).
        Applies feature embeddings, transformer layers, and normalization.
        """
        batch_size, num_const, num_features = jet_data.shape  # (batch, n_const, n_features)
        jet_data[jet_data < 0] = 0  # Handle negative values

        # Apply feature embeddings for each feature dimension
        emb = self.feature_embeddings[0](jet_data[:, :, 0])  # Embedding the first feature
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet_data[:, :, i])  # Sum embeddings across features
        
        # Construct causal mask to restrict attention to preceding elements
        seq_len = jet_data.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=jet_data.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)  # Causal mask
        padding_mask = ~padding_mask  # Invert padding mask for transformer layers

        #exit()
        # Apply transformer layers
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # Normalize and apply dropout
        jet_repr = self.out_norm(emb)
        #emb = self.dropout_layer(emb)
        #emb = self.flat(emb)
        # Take the representation of the [CLS] token or apply pooling if needed
        if self.jet_emb=='True':
            jet_repr = emb.mean(dim=1)  # Taking mean pooling across all constituents
        
        return jet_repr

    def loss(self, logits, true_bin):
        """
        Computes the loss for the given logits and ground truth binary labels.
        """
        
        #print('true bine')
        
        #print(true_bin)
        #print(true_bin.shape)
        return self.criterion(logits, true_bin)


##################################################################################





##################################################################################

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Embedding,
    TransformerEncoderLayer,
    TransformerEncoder,
    LayerNorm,
    Dropout
)

class JetTransformerALScratchCLS(Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100,
        # Extra: second downstream Transformer settings
        downstream_num_layers=2,
        downstream_num_heads=2,
        jet_last_emb='False'
    ):
        super(JetTransformerALScratchCLS, self).__init__()
        
        self.num_features = num_features
        self.dropout = dropout
        self.num_const = num_const
        self.hidden_dim = hidden_dim
        self.jet_last_emb = jet_last_emb
        # ===============  (1) ORIGINAL / PRE-TRAINED PART  ===============
        # Same as your original: feature embeddings + N-layers of Transformer
        self.feature_embeddings = nn.ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[i])
                for i in range(num_features)
            ]
        )
        
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = LayerNorm(hidden_dim)
        self.dropout_layer = Dropout(dropout)
        self.part_style='False'
        # This linear layer reduces the jet embedding from the first-stage
        # (only if you had something like that originally).
        # Or skip it if your old code didn't have it.   
        self.jet_mlp = nn.Linear(hidden_dim, hidden_dim)
        
        #ParTStyle transformations
        self.gelu = nn.GELU()
        self.out_norm_2 = LayerNorm(hidden_dim)
        self.jet_mlp_last = nn.Linear(hidden_dim, hidden_dim)
        # ===============  (2) DOWNSTREAM CLS HEAD  ===============
        # A second Transformer (small) that processes [CLS2, Jet1_repr, Jet2_repr]
        encoder_layer2 = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=downstream_num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.downstream_transformer = TransformerEncoder(encoder_layer2, num_layers=downstream_num_layers)
        
        # A new CLS token for the downstream classification
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Final classification
        self.final_classifier = nn.Linear(hidden_dim, 1)
        
        # Criterion
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, jet1, jet2, padding_mask1, padding_mask2, labels=None):
        """
        (A) Use the existing Transformer (unchanged) to get a single vector per jet
        (B) Combine those vectors in a second small Transformer with a new [CLS2] token
        (C) Output a single logit per event
        """
        # ========== (A) FIRST-STAGE TRANSFORMER (Unchanged) ==========
        jet1_repr = self._process_jet(jet1, padding_mask1)  # [B, hidden_dim]
        jet2_repr = self._process_jet(jet2, padding_mask2)  # [B, hidden_dim]
        
        # (Optional) pass through old MLP or standard dropout, etc. from your code
        
        if self.jet_last_emb=='True':
            jet1_repr = self.jet_mlp(jet1_repr)  # still [B, hidden_dim]
            jet2_repr = self.jet_mlp(jet2_repr)
        
        #ParT style
        
        if self.part_style=='True':
            jet1_repr = self.gelu(jet1_repr)  
            jet2_repr = self.gelu(jet2_repr)
            jet1_repr = self.out_norm_2(jet1_repr)  
            jet2_repr = self.out_norm_2(jet2_repr)
            jet1_repr = self.jet_mlp_last(jet1_repr)  
            jet2_repr = self.jet_mlp_last(jet2_repr)
        # ========== (B) DOWNSTREAM CLS-BASED TRANSFORMER ==========
        # Put the two jet embeddings side by side
        # shape = [B, 2, hidden_dim]
        combined_jets = torch.stack([jet1_repr, jet2_repr], dim=1)
        #combined_jets = torch.cat([jet1_repr, jet2_repr], dim=1) 
        
        # Insert the second CLS token at the front
        B = jet1_repr.size(0)
        cls2 = self.cls_token2.expand(B, -1, -1)  # [B, 1, hidden_dim]
        
        # shape = [B, 3, hidden_dim]
        downstream_input = torch.cat([cls2, combined_jets], dim=1)
      
        # Pass through small downstream Transformer
        out2 = self.downstream_transformer(downstream_input)
        
        # The [CLS2] representation is in index 0
        cls_output = out2[:, 0, :]  # [B, hidden_dim]
        
        # Final classification
        logits = self.final_classifier(cls_output)  # [B, 1]

        # If labels are provided, return loss
        if labels is not None:
            return logits, self.loss(logits, labels)
        return logits

    def _process_jet(self, jet_data, padding_mask):
        """
        This is your *original* method that processes a single jet.
        It could do mean pooling or an existing [CLS], etc. 
        We'll keep it exactly as you had it, minus references to HLF.
        """
        batch_size, num_const, _ = jet_data.shape
        jet_data[jet_data < 0] = 0

        # Feature embeddings
        emb = self.feature_embeddings[0](jet_data[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet_data[:, :, i])

        # Build a causal mask if you originally did that
        seq_len = emb.shape[1]
        seq_idx = torch.arange(seq_len, device=emb.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)

        # Invert padding mask if you used True=valid
        padding_mask = ~padding_mask

        # Pass through all original Transformer layers
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # Normalization + dropout
        emb = self.out_norm(emb)
        emb = self.dropout_layer(emb)
        #jet_repr=emb
        # Suppose the old code ended with:
        # return emb.mean(dim=1)
        # We'll keep that. So each jet is [B, hidden_dim].
        jet_repr = emb.mean(dim=1)
       
       
        return jet_repr

    def loss(self, logits, labels):
        return self.criterion(logits, labels)
##################################################################################


class JetTransformerALFineTuneCLS(Module):
    def __init__(
        self,
        original_model,
        pretrain='scratch',
      
        use_sep_token='False',
        use_hlf='linear',

        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100,
        # Extra: second downstream Transformer settings
        downstream_num_layers=2,
        downstream_num_heads=2,
        jet_last_emb='linear',
        hlf_dim=5,
        num_layers_hlf=4,
        hidden_dim_hlf=256
    ):
        super(JetTransformerALFineTuneCLS, self).__init__()
        
        self.num_features = num_features
        self.dropout = dropout
        self.num_const = num_const
        self.hidden_dim = hidden_dim
        self.jet_last_emb = jet_last_emb
        self.hlf_dim=hlf_dim
        self.pretrain=pretrain
       
        self.use_sep_token=use_sep_token
        self.use_hlf=use_hlf


        # ===============  (1) ORIGINAL / PRE-TRAINED PART  ===============
        # Same as your original: feature embeddings + N-layers of Transformer
        
        if self.pretrain=='scratch':
            print('training from scratch')
            
            self.feature_embeddings = nn.ModuleList(
                [
                    Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[i])
                    for i in range(num_features)
                ]
            )
            
            self.layers = nn.ModuleList(
                [
                    TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim,
                        batch_first=True,
                        norm_first=True,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.out_norm = LayerNorm(hidden_dim)
            self.dropout_layer = Dropout(dropout)
        elif self.pretrain=='fine' or self.pretrain=='freeze':
        
        

            # Feature embeddings are inherited from the original model
            self.feature_embeddings = original_model.feature_embeddings
            # Transformer layers are inherited from the original model
            self.layers = original_model.layers
            # Output normalization and dropout are inherited from the original model
            self.out_norm = original_model.out_norm
            self.dropout_layer = original_model.dropout
            #self.dropout_layer = Dropout(dropout)

      
        # This linear layer reduces the jet embedding from the first-stage
        # (only if you had something like that originally).
        # Or skip it if your old code didn't have it.   
        self.jet_mlp = nn.Linear(hidden_dim, hidden_dim)
        
        #ParTStyle transformations
        self.gelu = nn.GELU()
        self.out_norm_2 = LayerNorm(hidden_dim)
        self.jet_mlp_last = nn.Linear(hidden_dim, hidden_dim)
        


        #=========HIGH LEVEL FEATURES=====================#
        self.hlf_mlp1 = nn.Linear(self.hlf_dim, hidden_dim_hlf)
        # build transformer layers
        transformer_layer = TransformerEncoderLayer(d_model=hidden_dim_hlf, nhead=2,dropout=.5,activation='gelu',batch_first=True,
            norm_first=True)
        self.hlf_transformer = TransformerEncoder(transformer_layer, num_layers=num_layers_hlf)
        self.hlf_out_norm = LayerNorm(hidden_dim_hlf)
        
        self.hlf_mlp2 = nn.Linear(hidden_dim_hlf, hidden_dim)


        self.mlp_scale_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp_shift_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp_scale_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp_shift_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )



        # ===============  (2) DOWNSTREAM CLS HEAD  ===============
        # A second Transformer (small) that processes [CLS2, Jet1_repr, Jet2_repr]
        encoder_layer2 = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=downstream_num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.downstream_transformer = TransformerEncoder(encoder_layer2, num_layers=downstream_num_layers)
        
        # A new CLS token for the downstream classification
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # Final classification
        self.final_classifier = nn.Linear(hidden_dim, 1)
        
        # Criterion
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, jet1, jet2, padding_mask1, padding_mask2,hlf, labels=None):
        """
        (A) Use the existing Transformer (unchanged) to get a single vector per jet
        (B) Combine those vectors in a second small Transformer with a new [CLS2] token
        (C) Output a single logit per event
        """
        # ========== (A) FIRST-STAGE TRANSFORMER (Unchanged) ==========
        jet1_repr = self._process_jet(jet1, padding_mask1)  # [B, hidden_dim]
        jet2_repr = self._process_jet(jet2, padding_mask2)  # [B, hidden_dim]
        
        # (Optional) pass through old MLP or standard dropout, etc. from your code
        
        if self.jet_last_emb=='linear':
        
            #Similar to what ParT does
            jet1_repr = self.jet_mlp(jet1_repr)  
            jet2_repr = self.jet_mlp(jet2_repr)
           
            jet1_repr = self.gelu(jet1_repr)  
            jet2_repr = self.gelu(jet2_repr)

            jet1_repr = self.out_norm_2(jet1_repr)  
            jet2_repr = self.out_norm_2(jet2_repr)

            jet1_repr = self.jet_mlp_last(jet1_repr)  
            jet2_repr = self.jet_mlp_last(jet2_repr)
        


        #===========HIGH LEVEL FEATURES=========#
        #####HLF stage

        #print(hlf.size())
        if self.use_hlf=='linear' or self.use_hlf=='mean':
            hlf_1 = hlf[:, 0]  # Shape will be [128, 5]
            hlf_2 = hlf[:, 1]
            #print(hlf_1.size())
            
            #hlf = hlf.view(hlf.shape[0], -1)

    
            #print(hlf)
            #print(hlf.shape)
        
            hlf_1=self.hlf_mlp1(hlf_1)
        
            hlf_1 = torch.nn.LeakyReLU(negative_slope=0.01)(hlf_1)
            hlf_1 = self.hlf_transformer(hlf_1) # Transformer

            if self.use_hlf=='linear':
                hlf_1 = self.hlf_out_norm(hlf_1)
                hlf_1 = self.hlf_mlp2(hlf_1)
                hlf_1_repr = torch.nn.LeakyReLU(negative_slope=0.01)(hlf_1)
                
            elif self.use_hlf=='mean':
                hlf_1_repr =hlf_1.mean(dim=0)
            


            hlf_2=self.hlf_mlp1(hlf_2)
        
            hlf_2 = nn.GELU()(hlf_2)
            hlf_2 = self.hlf_transformer(hlf_2) # Transformer


            if self.use_hlf=='linear':
                
                hlf_2 = self.hlf_out_norm(hlf_2)
                hlf_2 = self.hlf_mlp2(hlf_2)
                hlf_2_repr = nn.GELU()(hlf_2)
                
            elif self.use_hlf=='mean':
                hlf_2_repr =hlf_1.mean(dim=0)
            
        

            #print(hlf_1_repr.size())
            #print(hlf_2_repr.size())
            
            scale_factor_1 = self.mlp_scale_1(hlf_1_repr)  # Shape: [128, hidden_dim]
            shift_value_1 = self.mlp_shift_1(hlf_1_repr)   # Shape: [128, hidden_dim]
            
            #exit()
            if self.jet_last_emb=='linear' and self.use_hlf=='linear':
                
                scale_factor_1 = scale_factor_1.unsqueeze(1)   # Shape: [128, 1, 256]
                shift_value_1 = shift_value_1.unsqueeze(1) 
    

            scaled_and_shifted_jets1 = jet1_repr * scale_factor_1 + shift_value_1
         
            # Scale and shift for jet 2
            scale_factor_2 = self.mlp_scale_2(hlf_2_repr)  # Shape: [128, hidden_dim]
            shift_value_2 = self.mlp_shift_2(hlf_2_repr)   # Shape: [128, hidden_dim]
            if self.jet_last_emb=='linear' and self.use_hlf=='linear':
                scale_factor_2 = scale_factor_2.unsqueeze(1)   # Shape: [128, 1, 256]
                shift_value_2 = shift_value_2.unsqueeze(1)
            
            scaled_and_shifted_jets2 = jet2_repr * scale_factor_2 + shift_value_2
            
            
            if self.use_sep_token=='True':
                B = jet1_repr.size(0)
                sep_token = self.sep_token.expand(B, -1,-1)
                if self.jet_last_emb=='mean':
                    combined_jets = torch.stack([scaled_and_shifted_jets1, sep_token.squeeze(1),scaled_and_shifted_jets2], dim=1)
                else:
                    combined_jets = torch.cat([scaled_and_shifted_jets1,sep_token, scaled_and_shifted_jets2], dim=1)
            else:
                if self.jet_last_emb=='mean':
                    combined_jets = torch.stack([scaled_and_shifted_jets1, scaled_and_shifted_jets2], dim=1)
                else:
                    combined_jets = torch.cat([scaled_and_shifted_jets1, scaled_and_shifted_jets2], dim=1)


        if self.use_hlf=='False':
            if self.use_sep_token=='True':
                B = jet1_repr.size(0)
                sep_token = self.sep_token.expand(B, -1, -1)
                if self.jet_last_emb=='mean':
                    
                    combined_jets = torch.stack([jet1_repr, sep_token.squeeze(1),jet2_repr], dim=1)
                else:
                    combined_jets = torch.cat([jet1_repr, sep_token,jet2_repr], dim=1)
            else:
                if self.jet_last_emb=='mean':
                    combined_jets = torch.stack([jet1_repr, jet2_repr], dim=1)
                else:
                    combined_jets = torch.cat([jet1_repr, jet2_repr], dim=1)
            

        #combined_jets = torch.cat([scaled_and_shifted_jets1, sep_token,scaled_and_shifted_jets2], dim=1)
        
        '''

        hlf_1_repr=hlf_1_repr.view(hlf.shape[0], 1, 256)
        hlf_2_repr=hlf_2_repr.view(hlf.shape[0], 1, 256)
        #print(hlf_1_repr.size())
        #print(jet1_repr.size())
        
        # ========== (B) DOWNSTREAM CLS-BASED TRANSFORMER ==========
        # Put the two jet embeddings side by side
        # shape = [B, 2, hidden_dim]
        #combined_jets = torch.stack([jet1_repr, jet2_repr], dim=1)
        
        
        #combined_jets = torch.cat([jet1_repr,hlf_1_repr,SEP_token, jet2_repr,hlf_2_repr], dim=1)
        B = jet1_repr.size(0)
        sep_token = self.sep_token.expand(B, -1, -1)
        combined_jets = torch.cat([jet1_repr,sep_token, jet2_repr], dim=1)
        '''
        #combined_jets = torch.cat([jet1_repr,hlf_1_repr, jet2_repr,hlf_2_repr], dim=1)
        #combined_jets = torch.cat([jet1_repr,sep_token, jet2_repr], dim=1)
        #print(combined_jets.size())
        
        # Insert the second CLS token at the front
        B = jet1_repr.size(0)
        cls2 = self.cls_token2.expand(B, -1, -1)  # [B, 1, hidden_dim]
        
        # shape = [B, 3, hidden_dim]
        downstream_input = torch.cat([cls2, combined_jets], dim=1)
        
        # Pass through small downstream Transformer
        out2 = self.downstream_transformer(downstream_input)
        
        # The [CLS2] representation is in index 0
        cls_output = out2[:, 0, :]  # [B, hidden_dim]
        
        # Final classification
        logits = self.final_classifier(cls_output)  # [B, 1]

        # If labels are provided, return loss
        if labels is not None:
            return logits, self.loss(logits, labels)
        return logits

    def _process_jet(self, jet_data, padding_mask):
        """
        This is your *original* method that processes a single jet.
        It could do mean pooling or an existing [CLS], etc. 
        We'll keep it exactly as you had it, minus references to HLF.
        """
        batch_size, num_const, _ = jet_data.shape
        jet_data[jet_data < 0] = 0

        # Feature embeddings
        emb = self.feature_embeddings[0](jet_data[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet_data[:, :, i])

        # Build a causal mask if you originally did that
        seq_len = emb.shape[1]
        seq_idx = torch.arange(seq_len, device=emb.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)

        # Invert padding mask if you used True=valid
        padding_mask = ~padding_mask

        # Pass through all original Transformer layers
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # Normalization + dropout
        emb = self.out_norm(emb)
        emb = self.dropout_layer(emb)
        jet_repr=emb
        # Suppose the old code ended with:
        # return emb.mean(dim=1)
        # We'll keep that. So each jet is [B, hidden_dim].
        #jet_repr = emb.mean(dim=1)
        if self.jet_last_emb=='mean':
            jet_repr = jet_repr.mean(dim=1)
        return jet_repr

    def loss(self, logits, labels):
        return self.criterion(logits, labels)


###########################################################################################

class JetTransformerALwHLF(Module):
    def __init__(
        self,
        original_model,
        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100,
        hlf_dim=(2,5),
        num_layers_hlf=4,
        hidden_dim_hlf=128
    ):
        super(JetTransformerALwHLF, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.num_const = num_const

        # Feature embeddings are inherited from the original model
        self.feature_embeddings = original_model.feature_embeddings
        # Transformer layers are inherited from the original model
        self.layers = original_model.layers
        # Output normalization and dropout are inherited from the original model
        self.out_norm = original_model.out_norm
        self.dropout_layer = original_model.dropout
        self.flat = torch.nn.Flatten()
        # Classification head with MLP and Average Pooling

        # Classification head with MLP and Average Pooling
        self.jet_mlp = nn.Linear(hidden_dim*self.num_const, 64)
 
        #HLF layers
        #self.hlf_mlp1 = nn.Linear(hlf_dim, hidden_dim)  # First MLP layer
        
        self.hlf_mlp1 = nn.Linear(10, hidden_dim_hlf)
        # build transformer layers
        transformer_layer = TransformerEncoderLayer(d_model=hidden_dim_hlf, nhead=2)
        self.hlf_transformer = TransformerEncoder(transformer_layer, num_layers=num_layers_hlf)
        
        
        self.hlf_mlp2 = nn.Linear(hidden_dim_hlf, 64)  # Final MLP

      
        self.mlp1 = nn.Linear(64 * 3, 128)  # After concatenating both jet representations
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Pooling over feature dimension
        self.mlp2 = nn.Linear(1, 128)  # Hidden layer after pooling
        self.output = nn.Linear(128, 1)  # Binary classification
        
        # Criterion for binary classification
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, jet1, jet2, padding_mask1, padding_mask2,hlf):
        """
        Forward pass where jet1 and jet2 are processed separately and their representations are concatenated.
        """

        # Apply feature embeddings and transformer processing separately for each jet
        jet1_repr = self._process_jet(jet1, padding_mask1)  # Process jet1
        jet2_repr = self._process_jet(jet2, padding_mask2)  # Process jet2


        jet1_repr = self.jet_mlp(jet1_repr)  # (batch, 64)
        jet2_repr = self.jet_mlp(jet2_repr)  # (batch, 64)



        #####HLF stage
        hlf = hlf.view(hlf.shape[0], -1)
        #print(hlf)
        #print(hlf.shape)
        hlf=self.hlf_mlp1(hlf)
        hlf = torch.nn.LeakyReLU(negative_slope=0.01)(hlf)
        hlf = self.hlf_transformer(hlf) # Transformer
        hlf = self.hlf_mlp2(hlf)
        hlf_repr = torch.nn.LeakyReLU(negative_slope=0.01)(hlf)


        pooling='avg'
        # Concatenate the representations of jet1 and jet2
        combined_repr = torch.cat([jet1_repr, jet2_repr,hlf_repr], dim=-1)  # Shape: [batch_size, hidden_dim * 2]
        if pooling=='avg':
            # MLP processing
            x = self.mlp1(combined_repr)
            x = torch.nn.Dropout(p=0.1)(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            #x = self.dropout_layer(x)

            # Average Pooling (across jet constituents)
            #x = x.unsqueeze(1)  # Add a dummy dimension for pooling
            #print('x before pooling')
            #print(x)
            #print(x.shape)
            x = self.avg_pool(x)
             # Perform average pooling
            #x = x.squeeze(-1)  # Remove the dummy dimension
            #print('x after pooling')
            #print(x)
            #print(x.shape)
            # Second MLP layer after pooling
            x = self.mlp2(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = torch.nn.Dropout(p=0.1)(x)
            #print('x after mpl2')
            #print(x)
            #print(x.shape)
            x=torch.nn.Flatten()(x)
            x = self.output(x)
        
        
     
        
        #print('x after output')
        #print(x)
        #print(x.shape)
        #x = torch.sigmoid(x)
        return x

    def _process_jet(self, jet_data, padding_mask):
        """
        Processes a single jet (either jet1 or jet2).
        Applies feature embeddings, transformer layers, and normalization.
        """
        batch_size, num_const, num_features = jet_data.shape  # (batch, n_const, n_features)
        jet_data[jet_data < 0] = 0  # Handle negative values

        # Apply feature embeddings for each feature dimension
        emb = self.feature_embeddings[0](jet_data[:, :, 0])  # Embedding the first feature
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet_data[:, :, i])  # Sum embeddings across features
        
        # Construct causal mask to restrict attention to preceding elements
        seq_len = jet_data.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=jet_data.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)  # Causal mask
        padding_mask = ~padding_mask  # Invert padding mask for transformer layers

        #exit()
        # Apply transformer layers
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # Normalize and apply dropout
        emb = self.out_norm(emb)
        #emb = self.dropout_layer(emb)
        #emb = self.flat(emb)
        # Take the representation of the [CLS] token or apply pooling if needed
        #jet_repr = emb.mean(dim=1)  # Taking mean pooling across all constituents
        
        return emb

    def loss(self, logits, true_bin):
        """
        Computes the loss for the given logits and ground truth binary labels.
        """
        
        #print('true bine')
        
        #print(true_bin)
        #print(true_bin.shape)
        return self.criterion(logits, true_bin)






##################################################################################

class JetTransformerALwHLFScratch(Module):
    def __init__(
        self,
        original_model,
        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100,
        hlf_dim=(2,5),
        num_layers_hlf=4,
        hidden_dim_hlf=128
    ):
        super(JetTransformerALwHLFScratch, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.num_const = num_const

        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )

        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout_layer = Dropout(dropout)

        self.flat = torch.nn.Flatten()
        # Classification head with MLP and Average Pooling

        # Classification head with MLP and Average Pooling
        self.jet_mlp = nn.Linear(hidden_dim*self.num_const, 64)
        #HLF layers
        #self.hlf_mlp1 = nn.Linear(hlf_dim, hidden_dim)  # First MLP layer
        
        self.hlf_mlp1 = nn.Linear(10, hidden_dim_hlf)
        # build transformer layers
        transformer_layer = TransformerEncoderLayer(d_model=hidden_dim_hlf, nhead=2)
        self.hlf_transformer = TransformerEncoder(transformer_layer, num_layers=num_layers_hlf)
        
        
        self.hlf_mlp2 = nn.Linear(hidden_dim_hlf, 64)  # Final MLP
 
 
        # Classification head with MLP and Average Pooling

        # Classification head with MLP and Average Pooling
   
      
        self.mlp1 = nn.Linear(64 * 3, 64*3)  # After concatenating both jet representations
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Pooling over feature dimension
        self.mlp2 = nn.Linear(1, 128)  # Hidden layer after pooling
        self.output = nn.Linear(128, 1)  # Binary classification
        
        # Criterion for binary classification
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, jet1, jet2, padding_mask1, padding_mask2,hlf):
        """
        Forward pass where jet1 and jet2 are processed separately and their representations are concatenated.
        """

        # Apply feature embeddings and transformer processing separately for each jet
        jet1_repr = self._process_jet(jet1, padding_mask1)  # Process jet1
        jet2_repr = self._process_jet(jet2, padding_mask2)  # Process jet2


        jet1_repr = self.jet_mlp(jet1_repr)  # (batch, 64)
        jet2_repr = self.jet_mlp(jet2_repr)  # (batch, 64)



        #####HLF stage
        hlf = hlf.view(hlf.shape[0], -1)
        #print(hlf)
        #print(hlf.shape)
        hlf=self.hlf_mlp1(hlf)
        hlf = torch.nn.LeakyReLU(negative_slope=0.01)(hlf)
        hlf = self.hlf_transformer(hlf) # Transformer
        hlf = self.hlf_mlp2(hlf)
        hlf_repr = torch.nn.LeakyReLU(negative_slope=0.01)(hlf)


        pooling='avg'
        # Concatenate the representations of jet1 and jet2
        combined_repr = torch.cat([jet1_repr, jet2_repr,hlf_repr], dim=-1)  # Shape: [batch_size, hidden_dim * 2]
        if pooling=='avg':
            # MLP processing
            x = self.mlp1(combined_repr)
            x = torch.nn.Dropout(p=0.1)(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            #x = self.dropout_layer(x)

            # Average Pooling (across jet constituents)
            #x = x.unsqueeze(1)  # Add a dummy dimension for pooling
            #print('x before pooling')
            #print(x)
            #print(x.shape)
            x = self.avg_pool(x)
             # Perform average pooling
            #x = x.squeeze(-1)  # Remove the dummy dimension
            #print('x after pooling')
            #print(x)
            #print(x.shape)
            # Second MLP layer after pooling
            x = self.mlp2(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = torch.nn.Dropout(p=0.1)(x)
            #print('x after mpl2')
            #print(x)
            #print(x.shape)
            x=torch.nn.Flatten()(x)
            x = self.output(x)
        
        
     
        
        #print('x after output')
        #print(x)
        #print(x.shape)
        #x = torch.sigmoid(x)
        return x

    def _process_jet(self, jet_data, padding_mask):
        """
        Processes a single jet (either jet1 or jet2).
        Applies feature embeddings, transformer layers, and normalization.
        """
        batch_size, num_const, num_features = jet_data.shape  # (batch, n_const, n_features)
        jet_data[jet_data < 0] = 0  # Handle negative values

        # Apply feature embeddings for each feature dimension
        emb = self.feature_embeddings[0](jet_data[:, :, 0])  # Embedding the first feature
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet_data[:, :, i])  # Sum embeddings across features
        
        # Construct causal mask to restrict attention to preceding elements
        seq_len = jet_data.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=jet_data.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)  # Causal mask
        padding_mask = ~padding_mask  # Invert padding mask for transformer layers

        #exit()
        # Apply transformer layers
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # Normalize and apply dropout
        emb = self.out_norm(emb)
        #emb = self.dropout_layer(emb)
        #emb = self.flat(emb)
        # Take the representation of the [CLS] token or apply pooling if needed
        #jet_repr = emb.mean(dim=1)  # Taking mean pooling across all constituents
        
        return emb

    def loss(self, logits, true_bin):
        """
        Computes the loss for the given logits and ground truth binary labels.
        """
        
        #print('true bine')
        
        #print(true_bin)
        #print(true_bin.shape)
        return self.criterion(logits, true_bin)





###################################################################################################################################################################

import torch
import torch.nn as nn
from torch.nn import Module, Embedding, TransformerEncoderLayer, TransformerEncoder, LayerNorm, Dropout

class JetTransformerALwHLFSeparate(Module):
    def __init__(
        self,
        original_model,
        hidden_dim=256,
        num_layers=8,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100,
        # HLF dims
        # We assume each jet has 5 HLFs, so total 10. We'll process them as 5 + 5
        hlf_per_jet=5,
        hidden_dim_hlf=64,
        # Additional
        final_hidden=128,
    ):
        super(JetTransformerALwHLFSeparate, self).__init__()
        
        self.num_features = num_features
        self.dropout = dropout
        self.num_const = num_const
        self.hidden_dim = hidden_dim
        
        # --------------------------
        # 1) Feature embeddings (for pt, eta, phi)
        # --------------------------
        self.feature_embeddings = nn.ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )
        
        # --------------------------
        # 2) Transformer for constituents
        # --------------------------
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout_layer = Dropout(dropout)
        
        # Reduce the final representation of each jet’s constituent transformer to 64
        self.jet_const_mlp = nn.Linear(hidden_dim, 64) 
        
        # --------------------------
        # 3) Two separate MLPs for HLF
        #    Each jet has 5 HLF features → embed to 64
        # --------------------------
        self.hlf_mlp = nn.Sequential(
            nn.Linear(hlf_per_jet, hidden_dim_hlf),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_hlf, 64),
            nn.LeakyReLU(0.01),
        )

        
        # --------------------------
        # 4) Final classification head
        #    We now have 4 × 64 = 256 dims after concatenating:
        #    (jet1_const, jet2_const, jet1_hlf, jet2_hlf)
        # --------------------------
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64*4, final_hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(final_hidden, 1),
        )
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, jet1, jet2, mask1, mask2, hlf):
        """
        jet1, jet2: shape (batch_size, n_const, 3). 
        padding_mask1, padding_mask2: shape (batch_size, n_const) 
          with True/False indicating valid constituents.
        hlf: shape (batch_size, 10) if you store jammed 5 HLFs for jet1 and 5 for jet2 in one Tensor.
        """
        # ------------------------------------------------
        # 1) Process each jet’s constituents with the transformer
        # ------------------------------------------------
        jet1_const_repr = self._process_jet(jet1, mask1)  # → (batch, hidden_dim)
        jet2_const_repr = self._process_jet(jet2, mask2)  # → (batch, hidden_dim)
        
        # Reduce to 64
        jet1_const_repr = self.jet_const_mlp(jet1_const_repr)  # → (batch, 64)
        jet2_const_repr = self.jet_const_mlp(jet2_const_repr)  # → (batch, 64)
        
        # ------------------------------------------------
        # 2) Split the HLF for each jet
        # ------------------------------------------------
        # We assume hlf.shape is (batch_size, 10) → first 5 for jet1, last 5 for jet2
        batch_size = hlf.shape[0]
        hlf_jet1 = hlf[:,0, :]  # (batch, 5)
        hlf_jet2 = hlf[:,1,]  # (batch, 5)
        
        # Pass each set of HLFs through its MLP
        hlf_repr_jet1 = self.hlf_mlp(hlf_jet1)  # → (batch, 64)
        hlf_repr_jet2 = self.hlf_mlp(hlf_jet2)  # → (batch, 64)
        
        # ------------------------------------------------
        # 3) Concatenate all 4 embeddings: shape (batch, 256)
        # ------------------------------------------------
        combined = torch.cat([jet1_const_repr, jet2_const_repr,
                              hlf_repr_jet1, hlf_repr_jet2], dim=-1)
        
        # ------------------------------------------------
        # 4) Final classification MLP
        # ------------------------------------------------
        logits = self.fusion_mlp(combined)  # → (batch, 1)
        return logits

    def _process_jet(self, jet_data, padding_mask):
        """
        Applies feature embeddings + transformer layers + mean pooling.
        Returns shape: (batch_size, hidden_dim).
        """
        batch_size, num_const, num_features = jet_data.shape
        jet_data[jet_data < 0] = 0  # Just as you had it

        # Feature embeddings
        emb = self.feature_embeddings[0](jet_data[:, :, 0])  # embedding for 1st feature
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet_data[:, :, i])
        
        # Build causal mask
        seq_len = emb.shape[1]
        seq_idx = torch.arange(seq_len, device=emb.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        
        # invert the padding mask for the transformer’s src_key_padding_mask
        padding_mask = ~padding_mask
        
        # Apply each transformer layer
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # out_norm + dropout
        emb = self.out_norm(emb)
        emb = self.dropout_layer(emb)
        
        # Mean pool across constituents
        #jet_repr = emb.mean(dim=1)
        return emb

    def loss(self, logits, true_bin):
        return self.criterion(logits, true_bin)





class JetTransformerClassifier(Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        num_const=100
    ):
        super(JetTransformerClassifier, self).__init__()
        self.num_features = num_features
        self.dropout = dropout

        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )

        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)

        # output projection and loss criterion
        self.flat = torch.nn.Flatten()
        #self.out = Linear(hidden_dim * 100, 1)
        self.out = Linear(hidden_dim * num_const, 1)
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, x, padding_mask):
        # construct causal mask to restrict attention to preceding elements
        seq_len = x.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=x.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        padding_mask = ~padding_mask

        # project x to initial embedding
        x[x < 0] = 0
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])

        # apply transformer layer
        for layer in self.layers:
            emb = layer(
                src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask
            )

        emb = self.out_norm(emb)
        emb = self.dropout(emb)
        emb = self.flat(emb)
        out = self.out(emb)
        return out

    def loss(self, logits, true_bin):

        loss = self.criterion(logits, true_bin)
        return loss


class JetTransformer(Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        output="linear",
        classifier=False,
        tanh=False,
        end_token=False,
    ):
        super(JetTransformer, self).__init__()
        self.num_features = num_features
        self.dropout = dropout
        self.total_bins = int(np.prod(num_bins))
        if end_token:
            self.total_bins += 1
        self.classifier = classifier
        self.tanh = tanh
        print(f"Bins: {self.total_bins}")

        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )

        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)

        # output projection and loss criterion
        if output == "linear":
            self.out_proj = Linear(hidden_dim, self.total_bins)
        else:
            self.out_proj = EmbeddingProductHead(hidden_dim, num_features, num_bins)
        self.criterion = CrossEntropyLoss()

    def forward(self, x, padding_mask):
        # construct causal mask to restrict attention to preceding elements
        seq_len = x.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=x.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        padding_mask = ~padding_mask

        # project x to initial embedding
        x[x < 0] = 0
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])

        # apply transformer layer
        for layer in self.layers:
            emb = layer(
                src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask
            )

        emb = self.out_norm(emb)
        emb = self.dropout(emb)

        # project final embedding to logits (not normalized with softmax)
        logits = self.out_proj(emb)
        if self.tanh:
            return 13 * torch.tanh(0.1 * logits)
        else:
            return logits

    def loss(self, logits, true_bin):
        # ignore final logits
        logits = logits[:, :-1].reshape(-1, self.total_bins)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        loss = self.criterion(logits, true_bin)
        return loss

    def probability(
        self,
        logits,
        padding_mask,
        true_bin,
        perplexity=False,
        logarithmic=False,
        topk=False,
    ):
        batch_size, padded_seq_len, num_bin = logits.shape
        seq_len = padding_mask.long().sum(dim=1)

        # ignore final logits
        logits = logits[:, :-1]
        probs = torch.softmax(logits, dim=-1)

        if topk:
            vals, idx = torch.topk(probs, topk, dim=-1, sorted=False)
            probs = torch.zeros_like(probs, device=probs.device)
            probs[
                torch.arange(probs.shape[0])[:, None, None],
                torch.arange(probs.shape[1])[None, :, None],
                idx,
            ] = vals

            probs = probs / probs.sum(dim=-1, keepdim=True)

        probs = probs.reshape(-1, self.total_bins)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        # select probs of true bins
        sel_idx = torch.arange(probs.shape[0], dtype=torch.long, device=probs.device)
        probs = probs[sel_idx, true_bin].view(batch_size, padded_seq_len - 1)
        probs[~padding_mask[:, 1:]] = 1.0
        if perplexity:
            probs = probs ** (1 / seq_len.float().view(-1, 1))

        if logarithmic:
            probs = torch.log(probs).sum(dim=1)
        else:
            probs = probs.prod(dim=1)
        return probs

    def sample(self, starts, device, len_seq, trunc=None):
        def select_idx():
            # Select bin at random according to probabilities
            rand = torch.rand((len(jets), 1), device=device)
            preds_cum = torch.cumsum(preds, -1)
            preds_cum[:, -1] += 0.01  # If rand = 1, sort it to the last bin
            idx = torch.searchsorted(preds_cum, rand).squeeze(1)
            return idx

        if not trunc is None and trunc >= 1:
            trunc = torch.tensor(trunc, dtype=torch.long)

        jets = -torch.ones((len(starts), len_seq, 3), dtype=torch.long, device=device)
        true_bins = torch.zeros((len(starts), len_seq), dtype=torch.long, device=device)

        # Set start bins and constituents
        num_prior_bins = torch.cumprod(torch.tensor([1, 41, 31]), -1).to(device)
        bins = (starts * num_prior_bins.reshape(1, 1, 3)).sum(axis=2)
        true_bins[:, 0] = bins
        jets[:, 0] = starts
        padding_mask = jets[:, :, 0] != -1

        self.eval()
        finished = torch.ones(len(starts)) != 1
        with torch.no_grad():
            for particle in range(len_seq - 1):
                if all(finished):
                    break
                # Get probabilities for the next particles
                preds = self.forward(jets, padding_mask)[:, particle]
                preds = torch.nn.functional.softmax(preds[:, :], dim=-1)

                # Remove low probs
                if not trunc is None:
                    if trunc < 1:
                        preds = torch.where(
                            preds < trunc, torch.zeros(1, device=device), preds
                        )
                    else:
                        preds, indices = torch.topk(preds, trunc, -1, sorted=False)

                preds = preds / torch.sum(preds, -1, keepdim=True)

                idx = select_idx()
                if not trunc is None and trunc >= 1:
                    idx = indices[torch.arange(len(indices)), idx]
                finished[idx == 39401] = True

                # Get tuple from found bin and set next particle properties
                true_bins[~finished, particle + 1] = idx[~finished]
                bins = self.idx_to_bins(idx[~finished])
                for ind, tmp_bin in enumerate(bins):
                    jets[~finished, particle + 1, ind] = tmp_bin

                padding_mask[~finished, particle + 1] = True
        return jets, true_bins

    def idx_to_bins(self, x):
        pT = x % 41
        eta = torch.div((x - pT), 41, rounding_mode="trunc") % torch.div(
            1271, 41, rounding_mode="trunc"
        )
        phi = torch.div((x - pT - 41 * eta), 1271, rounding_mode="trunc")
        return pT, eta, phi


class CNNclass(Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = torch.nn.Sequential(
            # Input = 1 x 30 x 30, Output = 32 x 30 x 30
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 32 x 30 x 30, Output = 32 x 15 x 15
            torch.nn.MaxPool2d(kernel_size=2),
            # Input = 32 x 15 x 15, Output = 64 x 15 x 15
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 64 x 15 x 15, Output = 64 x 7 x 7
            torch.nn.MaxPool2d(kernel_size=2),
            # Input = 64 x 7 x 7, Output = 64 x 7 x 7
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 64 x 7 x 7, Output = 64 x 3 x 3
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 3 * 3, 512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(x, y)


class ParticleNet(Module):
    pass
