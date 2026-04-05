"""
Martingale Posterior Neural Process (MPNP) Implementation

This module implements the Martingale Posterior Neural Process, which extends
the standard Neural Process with principled uncertainty quantification through
predictive resampling (pseudo-context generation).
"""

import sys
from pathlib import Path

# Add src directory to path for sibling package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import nn
from torch.distributions import Normal
from math import pi
import math

# Import base Neural Process components from sibling package
from neural_process import Encoder, MuSigmaEncoder, Decoder


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for autoregressive pseudo-context generation."""
    
    def __init__(self, dim_hidden=128, num_heads=8):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.head_dim = dim_hidden // num_heads
        
        self.fc_q = nn.Linear(dim_hidden, dim_hidden)
        self.fc_k = nn.Linear(dim_hidden, dim_hidden)
        self.fc_v = nn.Linear(dim_hidden, dim_hidden)
        self.fc_out = nn.Linear(dim_hidden, dim_hidden)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, n_query, dim]
            key: [batch, n_key, dim]
            value: [batch, n_value, dim]
            mask: [batch, n_key] - mask for keys
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        Q = self.fc_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.fc_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.fc_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # mask: [batch, n_key] -> [batch, 1, 1, n_key]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        # Handle NaN from all-masked rows
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_hidden)
        output = self.fc_out(attn_output)
        
        return output


class MAB(nn.Module):
    """Multihead Attention Block with residual connection and layer normalization."""
    
    def __init__(self, dim_in=128, dim_out=128, dim_hidden=128, num_heads=8):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.attention = MultiHeadAttention(dim_hidden, num_heads)
        self.fc_out = nn.Linear(dim_hidden, dim_hidden)
        self.fc_real_out = nn.Linear(dim_hidden, dim_out)
        self.ln1 = nn.LayerNorm(dim_hidden)
        self.ln2 = nn.LayerNorm(dim_hidden)
        
        # Project inputs to dim_hidden if needed
        self.project_query = nn.Linear(dim_in, dim_hidden) if dim_in != dim_hidden else nn.Identity()
        self.project_kv = nn.Linear(dim_in, dim_hidden) if dim_in != dim_hidden else nn.Identity()
        
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: [batch, n_query, dim_in]
            key_value: [batch, n_kv, dim_in]
            mask: [batch, n_kv]
        """
        # Project to dim_hidden
        query_proj = self.project_query(query)
        kv_proj = self.project_kv(key_value)
        
        attn_out = self.attention(query_proj, kv_proj, kv_proj, mask)
        out = self.ln1(query_proj + attn_out)
        out = self.ln2(out + torch.relu(self.fc_out(out)))
        out = self.fc_real_out(out)
        return out


class ISAB(nn.Module):
    """Induced Set Attention Block for pseudo-context generation."""
    
    def __init__(self, dim_in=128, dim_out=128, dim_hidden=128, num_heads=8):
        super().__init__()
        # First MAB: context (dim_in) attends to generate_sample (dim_in)
        self.mab0 = MAB(dim_in=dim_in, dim_out=dim_hidden, dim_hidden=dim_hidden, num_heads=num_heads)
        # Second MAB: generate_sample (dim_in) attends to h (dim_hidden)  
        self.mab1 = MAB(dim_in=dim_in, dim_out=dim_out, dim_hidden=dim_hidden, num_heads=num_heads)
        
    def forward(self, context, generate_sample, mask_context=None):
        """
        Args:
            context: [batch, n_context, dim_in] - encoded context representations
            generate_sample: [batch, n_generate, dim_in] - initial noise samples
            mask_context: [batch, n_context] - mask for context
        """
        # First MAB: context attends to generate_sample
        h = self.mab0(context, generate_sample, mask=None)
        # Second MAB: generate_sample attends to h (which is influenced by context)
        out = self.mab1(generate_sample, h, mask_context)
        return out

class MartingalePosteriorNeuralProcess(nn.Module):
    """
    Martingale Posterior Neural Process (MPNP) implementation, extending a Neural Process 
    to enforce martingale-consistent posterior updates via predictive resampling.
    
    This model generates 'pseudo' context points from its current predictive distribution 
    and incorporates them with the true context to update the latent function belief, 
    ensuring that posterior updates form a martingale (i.e., the updated predictions 
    on average equal prior predictions given self-sampled data).
    
    Parameters
    ----------
    x_dim : int
        Dimension of x values (input domain).
    y_dim : int
        Dimension of y values (output range).
    r_dim : int
        Dimension of the context representation vector (encoder output).
    z_dim : int
        Dimension of the latent function embedding (latent variable theta).
    h_dim : int
        Dimension of hidden layers in encoder and decoder networks.
    num_pseudo_points : int, optional
        Number of pseudo context points to generate for each task during training (default 20).
    num_pseudo_samples : int, optional
        Number of pseudo-context sets (K) to sample per task for the marginal likelihood estimate (default 5).
    x_range : tuple, optional
        Range (min, max) for pseudo-point x coordinates (default (-1, 1)).
    loss_weights : dict, optional
        Weights for loss terms: {'marg': 1.0, 'amort': 1.0, 'pseudo': 0.1}
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, 
                 num_pseudo_points=20, num_pseudo_samples=5,
                 use_autoregressive=True, num_heads=8,
                 x_range=(-1, 1), loss_weights=None):
        super(MartingalePosteriorNeuralProcess, self).__init__()
        # Save model dimensions
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.num_pseudo_points = num_pseudo_points
        self.num_pseudo_samples = num_pseudo_samples
        self.use_autoregressive = use_autoregressive
        self.x_range = x_range
        
        # Loss weights with sensible defaults
        # Reduce pseudo weight to prevent it from hurting generalization
        default_weights = {'marg': 1.0, 'amort': 1.0, 'pseudo': 0.1}
        self.loss_weights = loss_weights if loss_weights is not None else default_weights
        
        # Initialize base NP components (encoder, latent param mapper, decoder)
        self.xy_encoder = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.latent_encoder = MuSigmaEncoder(r_dim, z_dim)
        self.decoder = Decoder(x_dim, z_dim, h_dim, y_dim)
        
        # Autoregressive module for pseudo-context generation
        if use_autoregressive:
            # ISAB takes concatenated (x, y) representations and generates pseudo-contexts
            self.autoregressive = ISAB(dim_in=r_dim, dim_out=x_dim + y_dim, dim_hidden=r_dim, num_heads=num_heads)
    
    def aggregate(self, r_i):
        """
        Aggregate per-point representations into a single context representation.
        (Using mean pooling for permutation invariance, as in standard NP.)
        """
        # r_i shape: (batch_size, num_points, r_dim)
        return torch.mean(r_i, dim=1)
    
    def encode_set(self, x, y):
        """
        Encode a set of points (x_i, y_i) into a latent distribution (mu, sigma) for theta.
        Returns Normal distribution q(theta | set).
        """
        batch_size, num_pts, _ = x.size()
        # Flatten inputs for encoder
        x_flat = x.view(batch_size * num_pts, self.x_dim)
        y_flat = y.view(batch_size * num_pts, self.y_dim)
        # Encode each (x_i, y_i) to representation r_i
        r_i_flat = self.xy_encoder(x_flat, y_flat)  # shape: (batch_size * num_pts, r_dim)
        r_i = r_i_flat.view(batch_size, num_pts, self.r_dim)
        # Aggregate representations into a single r
        r = self.aggregate(r_i)  # shape: (batch_size, r_dim)
        # Map to latent parameter distribution (mu, sigma)
        mu, sigma = self.latent_encoder(r) 
        return Normal(mu, sigma)
    
    def generate_pseudo_context(self, q_context, x_context=None, y_context=None):
        """
        Draw a pseudo context set from the model's current predictive (martingale) posterior.
        This simulates drawing new data from the model's predictive distribution given the current context.
        
        Parameters
        ----------
        q_context : Normal (batch_size x Normal(mu_context, sigma_context))
            Latent distribution given the true context (approximates p(theta | Z_c)).
        x_context : torch.Tensor of shape (batch_size, n_context, x_dim), optional
            Context inputs (required if use_autoregressive=True)
        y_context : torch.Tensor of shape (batch_size, n_context, y_dim), optional
            Context outputs (required if use_autoregressive=True)
        
        Returns
        -------
        x_pseudo : torch.Tensor of shape (batch_size, num_pseudo_points, x_dim)
        y_pseudo : torch.Tensor of shape (batch_size, num_pseudo_points, y_dim)
            A set of pseudo context points sampled from the model's predictive distribution.
        """
        batch_size = q_context.loc.size(0)
        device = q_context.loc.device
        
        if self.use_autoregressive and x_context is not None and y_context is not None:
            # Autoregressive generation using ISAB
            # Encode context to get representations
            batch_size_ctx, num_ctx, _ = x_context.size()
            x_ctx_flat = x_context.view(batch_size_ctx * num_ctx, self.x_dim)
            y_ctx_flat = y_context.view(batch_size_ctx * num_ctx, self.y_dim)
            r_ctx_flat = self.xy_encoder(x_ctx_flat, y_ctx_flat)
            r_context = r_ctx_flat.view(batch_size_ctx, num_ctx, self.r_dim)
            
            # Initialize random samples for generation
            generate_initial = torch.randn(batch_size, self.num_pseudo_points, self.r_dim, device=device)
            
            # Create mask for context (all valid)
            mask_context = torch.ones(batch_size, num_ctx, dtype=torch.bool, device=device)
            
            # Generate pseudo-context using ISAB
            # Output is in (x, y) space
            pseudo_xy = self.autoregressive(r_context, generate_initial, mask_context)
            
            # Split into x and y components
            x_pseudo = pseudo_xy[..., :self.x_dim]
            # y component is encoded in the second half but we'll regenerate y from decoder
            
            # For x coordinates, apply sigmoid to get [0,1] then scale to [-2pi, 2pi] for sine waves
            x_pseudo = 4 * pi * (torch.sigmoid(x_pseudo) - 0.5)
            
            # For y, decode to get proper predictions with uncertainty
            z_sample = q_context.rsample()  # sample latent
            y_pred_mu, y_pred_sigma = self.decoder(x_pseudo, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)
            
            # Sample y with clipping to prevent extreme values
            noise = torch.randn_like(y_pred_mu).clamp(-1, 1)
            y_pseudo = y_pred_mu + y_pred_sigma * noise
            
        else:
            # Original strategy: sample x uniformly and decode
            z_sample = q_context.rsample()  # shape: (batch_size, z_dim)
            # Sample random x locations for pseudo points within configured range
            x_min, x_max = self.x_range
            x_pseudo = torch.rand(batch_size, self.num_pseudo_points, self.x_dim, device=device)
            x_pseudo = x_pseudo * (x_max - x_min) + x_min
            # Decode to get predictive distribution for each pseudo x
            y_pred_mu, y_pred_sigma = self.decoder(x_pseudo, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)
            # Sample y values from the predictive distribution
            y_pseudo = p_y_pred.rsample()
        
        return x_pseudo, y_pseudo
    
    def compute_loss(self, x_context, y_context, x_target, y_target):
        """
        Compute the MPNP loss for a batch.
        
        If num_pseudo_samples == 0, uses simplified CNP-style loss (just amortization term).
        Otherwise, combines:
         - Marginal likelihood term L_marg (via K predictive samples),
         - Context-only likelihood (amortization) term L_amort,
         - Pseudo-context likelihood term L_pseudo.

        Each term is derived from the MPNP training objective (Eq. 16-19 in the paper).
        """
        # Encode true context to get context latent distribution q(theta | Z_c)
        q_context = self.encode_set(x_context, y_context)  # Normal dist for theta given true context
        
        # Sample from the latent distribution (stochastic) rather than using mean (deterministic)
        # This adds regularization and prevents overfitting
        z_c = q_context.rsample()  # stochastic sampling for regularization
        
        # Amortization term: log p(Y_target | Z_c) using the context encoding
        y_ctx_mu, y_ctx_sigma = self.decoder(x_target, z_c) 
        p_y_ctx = Normal(y_ctx_mu, y_ctx_sigma)
        # Compute negative log-likelihood of true targets under context-only predictive
        # Use mean over points (not sum) for more stable gradients
        neg_loglik_context = -p_y_ctx.log_prob(y_target).mean(dim=(1,2))  # shape: (batch_size,)
        L_amort = neg_loglik_context.mean()
        
        # If no pseudo-samples, use simplified CNP-style loss
        if self.num_pseudo_samples == 0:
            return L_amort
        
        # Initialize accumulators for marginal likelihood and pseudo-context terms
        log_weights = []    # will store log p(Y_target | theta_k) for each pseudo-sample k (for L_marg)
        neg_loglik_pseudo = []  # will store -log p(Y_target | theta(Z0_k)) for each pseudo context k (for L_pseudo)
        
        # Generate K pseudo contexts and evaluate predictions
        for k in range(self.num_pseudo_samples):
            # Draw one pseudo context set from the predictive distribution given true context
            x_pseudo, y_pseudo = self.generate_pseudo_context(q_context, x_context, y_context)
            # Combine true and pseudo context sets (Z_c union Z0^(k))
            x_comb = torch.cat([x_context, x_pseudo], dim=1)
            y_comb = torch.cat([y_context, y_pseudo], dim=1)
            # Encode combined set to get q(theta | Z_c union Z0^(k))
            q_comb = self.encode_set(x_comb, y_comb)
            # Use the mean of q_comb as the function parameter theta_k (deterministic representation of combined context)
            z_comb = q_comb.loc
            # Decode target points using theta_k to get predictive distribution for Y_target
            y_comb_mu, y_comb_sigma = self.decoder(x_target, z_comb)
            p_y_comb = Normal(y_comb_mu, y_comb_sigma)
            # Compute log-likelihood of true targets under this combined-context function
            log_p_Y_given_theta = p_y_comb.log_prob(y_target).mean(dim=(1,2))  # per-pixel log-likelihood (batch,)
            log_weights.append(log_p_Y_given_theta)  # store for marginal likelihood estimation
            
            # **Pseudo-context term**: also evaluate how well the *pseudo context alone* could predict Y_target.
            # Encode only the pseudo context Z0^(k) to get q(theta | Z0_k)
            q_pseudo = self.encode_set(x_pseudo, y_pseudo)
            z_pseudo = q_pseudo.loc
            y_pseudo_mu, y_pseudo_sigma = self.decoder(x_target, z_pseudo)
            p_y_pseudo = Normal(y_pseudo_mu, y_pseudo_sigma)
            # Negative log-likelihood of targets using only pseudo context's function estimate
            neg_ll_pseudo = -p_y_pseudo.log_prob(y_target).mean(dim=(1,2))  # shape: (batch_size,)
            neg_loglik_pseudo.append(neg_ll_pseudo)
        
        # Convert collected lists to tensors (shape: num_samples x batch_size)
        log_weights = torch.stack(log_weights, dim=0)  # shape (K, batch_size)
        neg_loglik_pseudo = torch.stack(neg_loglik_pseudo, dim=0)  # shape (K, batch_size)
        
        # Marginal likelihood term: 
        # Approximate log p(Y_target | Z_c) = log(1/K * sum_k exp(log_p_Y_given_theta_k))
        # Compute log-mean-exp in a stable way across K samples for each task in batch.
        max_log = torch.max(log_weights, dim=0).values  # shape (batch_size,)
        # log_mean_exp = max_log + log( mean(exp(log_weights - max_log)) )
        log_mean_exp = max_log + torch.log(torch.mean(torch.exp(log_weights - max_log), dim=0))
        # Negative of the log marginal likelihood (to minimize)
        neg_log_marg = -log_mean_exp  # shape: (batch_size,)
        L_marg = neg_log_marg.mean()  # average over batch
        
        # Pseudo-context term: average negative log-likelihood of targets under each pseudo-context alone
        # Take mean over K pseudo sets for each task, then mean over batch.
        neg_loglik_pseudo = neg_loglik_pseudo.mean(dim=0)  # mean over K, shape (batch_size,)
        L_pseudo = neg_loglik_pseudo.mean()

        # Total MPNP loss with configurable weights
        # Default weights reduce pseudo term to prevent hurting generalization
        w_marg = self.loss_weights.get('marg', 1.0)
        w_amort = self.loss_weights.get('amort', 1.0)
        w_pseudo = self.loss_weights.get('pseudo', 0.1)
        
        loss = w_marg * L_marg + w_amort * L_amort + w_pseudo * L_pseudo
        return loss
    
    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Forward pass for inference. During training, use compute_loss() instead.
        
        If y_target is None (inference mode), returns a Normal distribution over y_target 
        given the current context. We draw one function sample from the posterior (via context encoding).
        If y_target is provided, this forward does not compute the full MPNP losses (it will behave similar to a standard NP forward pass).
        """
        if self.training and y_target is not None:
            # In training mode, use compute_loss for proper objective.
            # Here we fallback to a standard NP-like forward if needed (not typically used).
            q_context = self.encode_set(x_context, y_context)
            z = q_context.rsample()  # sample a latent function theta from context posterior
            y_pred_mu, y_pred_sigma = self.decoder(x_target, z)
            return Normal(y_pred_mu, y_pred_sigma)
        else:
            # At test time, produce a predictive distribution for targets given context.
            q_context = self.encode_set(x_context, y_context)  # latent dist from true context
            z = q_context.rsample()  # sample one latent function from posterior
            y_pred_mu, y_pred_sigma = self.decoder(x_target, z)
            return Normal(y_pred_mu, y_pred_sigma)
