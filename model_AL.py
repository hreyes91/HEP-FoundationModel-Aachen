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

 
        # Classification head with MLP and Average Pooling

        # Classification head with MLP and Average Pooling
        self.jet_mlp = nn.Linear(hidden_dim, 64)  # Reduce hidden_dim to 64 per jet
      
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
            #x=torch.nn.Flatten()(x)
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
        emb = self.dropout_layer(emb)

        # Take the representation of the [CLS] token or apply pooling if needed
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

 
        #HLF layers
        #self.hlf_mlp1 = nn.Linear(hlf_dim, hidden_dim)  # First MLP layer
        
        self.hlf_mlp1 = nn.Linear(10, hidden_dim)
        # build transformer layers
        transformer_layer = TransformerEncoderLayer(d_model=hidden_dim_hlf, nhead=2)
        self.hlf_transformer = TransformerEncoder(transformer_layer, num_layers=num_layers_hlf)
        
        
        self.hlf_mlp2 = nn.Linear(hidden_dim, hidden_dim)  # Final MLP
 
 
        # Classification head with MLP and Average Pooling

        # Classification head with MLP and Average Pooling
        self.jet_mlp = nn.Linear(hidden_dim, 64)  # Reduce hidden_dim to 64 per jet
      
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
        hlf=torch.nn.Flatten()(hlf)
        print(hlf)
        print(hlf.shape)
        hlf = F.leaky_relu(self.hlf_mlp1(hlf))
        hlf = self.hlf_transformer(hlf) # Transformer
        hlf_repr = F.leaky_relu(self.hlf_mlp2(hlf))


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
            #x=torch.nn.Flatten()(x)
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
        emb = self.dropout_layer(emb)

        # Take the representation of the [CLS] token or apply pooling if needed
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



###################################################################################################################################################################


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
