"""KmerDPVAE: k-mer-only VAE with truncated DP-GMM prior.

Simplified variant that uses only k-mer profiles as both input and
reconstruction target. No CNN sequence branch, no autoregressive decoder.

Encoder: MLP -> (mu, sigma2)
Decoder: MLP -> reconstructed k-mer profile
Prior:   Truncated stick-breaking DP-GMM
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from embedders.base import BaseEmbedder, EmbeddingResult
from features.kmer import KmerFeatureExtractor


# ---------------------------------------------------------------------------
# Truncated DP-GMM prior (NOT nn.Module — closed-form updates)
# ---------------------------------------------------------------------------

class DPGMM:
    """Truncated stick-breaking DP-GMM with variational inference.

    All parameters are updated via closed-form coordinate ascent, NOT SGD.

    Generative prior:
        v_k ~ Beta(1, alpha),  pi_k = v_k * prod_{l<k}(1-v_l)
        c_i ~ Cat(pi),         z_i | c_i=k ~ N(m_k, diag(s2_k))

    Variational family:
        q(v_k) = Beta(a_k, b_k)
        q(c_i) = Cat(r_i)
        m_k, S_k are point estimates updated in closed form.
    """

    def __init__(self, K_max: int = 500, d_z: int = 64, alpha: float = 1.0,
                 s_min: float = 1e-6, device: str = "cpu"):
        self.K = K_max
        self.d_z = d_z
        self.alpha = alpha
        self.s_min = s_min
        self.device = device

        # Stick-breaking Beta parameters
        self.a = torch.ones(K_max, device=device)
        self.b = torch.full((K_max,), alpha, device=device)

        # Component means and diagonal variances
        self.m = torch.randn(K_max, d_z, device=device) * 0.01
        self.S = torch.ones(K_max, d_z, device=device)

    def to(self, device):
        self.device = device
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.m = self.m.to(device)
        self.S = self.S.to(device)
        return self

    def init_from_data(self, mu_all: torch.Tensor):
        """Initialize components from encoder means via k-means.

        Args:
            mu_all: (N, d_z) encoder posterior means for all training data.
        """
        from sklearn.cluster import KMeans

        N = mu_all.shape[0]
        n_clusters = min(self.K, N)
        mu_np = mu_all.detach().cpu().numpy()

        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=0)
        km.fit(mu_np)

        centers = torch.from_numpy(km.cluster_centers_).float().to(self.device)
        global_var = torch.from_numpy(mu_np.var(axis=0)).float().to(self.device)

        self.m[:n_clusters] = centers
        if n_clusters < self.K:
            global_mean = mu_all.mean(dim=0)
            self.m[n_clusters:] = global_mean.unsqueeze(0) + \
                torch.randn(self.K - n_clusters, self.d_z, device=self.device) * 0.1

        self.S[:] = global_var.unsqueeze(0).clamp(min=self.s_min)
        self.a[:] = 1.0
        self.b[:] = self.alpha

    def compute_expected_log_pi(self) -> torch.Tensor:
        """Compute E[log pi_k] from stick-breaking Beta parameters.

        Returns: (K,) expected log mixture weights.
        """
        digamma_a = torch.special.digamma(self.a)
        digamma_b = torch.special.digamma(self.b)
        digamma_ab = torch.special.digamma(self.a + self.b)

        E_log_v = digamma_a - digamma_ab
        E_log_1mv = digamma_b - digamma_ab

        cumsum_log_1mv = torch.cumsum(E_log_1mv, dim=0)
        cumsum_shifted = torch.zeros_like(cumsum_log_1mv)
        cumsum_shifted[1:] = cumsum_log_1mv[:-1]

        return E_log_v + cumsum_shifted

    def compute_responsibilities(self, mu: torch.Tensor,
                                 sigma2: torch.Tensor) -> torch.Tensor:
        """Compute responsibilities r_ik = q(c_i = k).

        Args:
            mu: (N, d_z) encoder posterior means
            sigma2: (N, d_z) encoder posterior diagonal variances
        Returns:
            r: (N, K) responsibilities (sum to 1 over k)
        """
        E_log_pi = self.compute_expected_log_pi()

        combined_var = sigma2.unsqueeze(1) + self.S.unsqueeze(0)
        diff = mu.unsqueeze(1) - self.m.unsqueeze(0)

        log_gauss = -0.5 * (
            torch.log(combined_var).sum(dim=2)
            + (diff ** 2 / combined_var).sum(dim=2)
        )

        log_r = E_log_pi.unsqueeze(0) + log_gauss
        r = torch.softmax(log_r, dim=1)
        return r

    def update_parameters(self, mu: torch.Tensor, sigma2: torch.Tensor,
                          r: torch.Tensor):
        """Closed-form M-step updates for all mixture parameters.

        Args:
            mu: (N, d_z) encoder posterior means
            sigma2: (N, d_z) encoder posterior diagonal variances
            r: (N, K) responsibilities
        """
        N_k = r.sum(dim=0)
        N_k_safe = N_k.clamp(min=1e-10)

        # Stick-breaking updates
        self.a = 1.0 + N_k
        cumsum_rev = torch.flip(torch.cumsum(torch.flip(N_k, [0]), dim=0), [0])
        self.b = self.alpha + cumsum_rev - N_k

        # Component means
        self.m = (r.T @ mu) / N_k_safe.unsqueeze(1)

        # Component diagonal covariance
        diff = mu.unsqueeze(1) - self.m.unsqueeze(0)
        weighted = r.unsqueeze(2) * (sigma2.unsqueeze(1) + diff ** 2)
        self.S = weighted.sum(dim=0) / N_k_safe.unsqueeze(1)
        self.S = self.S.clamp(min=self.s_min)

    def kl_stick_breaking(self) -> torch.Tensor:
        """KL[q(v) || p(v)] = sum_k KL[Beta(a_k, b_k) || Beta(1, alpha)]."""
        log_B_prior = torch.lgamma(torch.tensor(1.0, device=self.device)) + \
            torch.lgamma(torch.tensor(self.alpha, device=self.device)) - \
            torch.lgamma(torch.tensor(1.0 + self.alpha, device=self.device))
        log_B_q = torch.lgamma(self.a) + torch.lgamma(self.b) - \
            torch.lgamma(self.a + self.b)

        psi_a = torch.special.digamma(self.a)
        psi_b = torch.special.digamma(self.b)
        psi_ab = torch.special.digamma(self.a + self.b)

        kl = log_B_prior - log_B_q + \
            (self.a - 1) * psi_a + \
            (self.b - self.alpha) * psi_b - \
            (self.a + self.b - 1 - self.alpha) * psi_ab

        return kl.sum()

    def expected_log_component_density(self, mu: torch.Tensor,
                                       sigma2: torch.Tensor,
                                       r: torch.Tensor) -> torch.Tensor:
        """Weighted E_q(z)[log N(z | m_k, S_k)] summed over i, k."""
        diff = mu.unsqueeze(1) - self.m.unsqueeze(0)
        inv_S = 1.0 / self.S.unsqueeze(0)

        log_det = torch.log(self.S).sum(dim=1)
        mahal = ((sigma2.unsqueeze(1) + diff ** 2) * inv_S).sum(dim=2)

        log_prob = -0.5 * (
            self.d_z * math.log(2 * math.pi) + log_det.unsqueeze(0) + mahal
        )

        return (r * log_prob).sum()

    def expected_log_pi_term(self, r: torch.Tensor) -> torch.Tensor:
        """sum_{i,k} r_ik E[log pi_k]. Returns scalar."""
        E_log_pi = self.compute_expected_log_pi()
        return (r * E_log_pi.unsqueeze(0)).sum()

    def assignment_entropy(self, r: torch.Tensor) -> torch.Tensor:
        """H[q(c)] = -sum_{i,k} r_ik log r_ik. Returns scalar."""
        return -(r * torch.log(r.clamp(min=1e-30))).sum()

    def posterior_entropy(self, sigma2: torch.Tensor) -> torch.Tensor:
        """H[q(z)] = sum_i 0.5 * sum_d log(2*pi*e * sigma2_id). Returns scalar."""
        return 0.5 * torch.log(2 * math.pi * math.e * sigma2).sum()

    def overlap_penalty(self) -> torch.Tensor:
        """Bhattacharyya-based Gaussian overlap penalty between components.

        L_overlap = sum_{k<l} pi_k * pi_l * exp(-D_B(k, l))

        where D_B is the Bhattacharyya distance between diagonal Gaussians:
        D_B(k,l) = 0.125*(m_k-m_l)^T S_avg^{-1} (m_k-m_l) + 0.5*log(|S_avg|/sqrt(|S_k||S_l|))
        with S_avg = (S_k + S_l) / 2.

        Returns: scalar penalty (higher = more overlap).
        """
        # Expected pi from stick-breaking
        E_log_pi = self.compute_expected_log_pi()
        pi = torch.softmax(E_log_pi, dim=0)  # (K,)

        # Only consider occupied components (pi > threshold) for efficiency
        mask = pi > 1e-4
        idx = torch.where(mask)[0]
        if len(idx) < 2:
            return torch.tensor(0.0, device=self.device)

        pi_sel = pi[idx]            # (M,)
        m_sel = self.m[idx]         # (M, d_z)
        S_sel = self.S[idx]         # (M, d_z)

        # Pairwise Bhattacharyya distance (vectorized)
        # S_avg_{kl} = (S_k + S_l) / 2
        S_k = S_sel.unsqueeze(1)    # (M, 1, d_z)
        S_l = S_sel.unsqueeze(0)    # (1, M, d_z)
        S_avg = (S_k + S_l) / 2    # (M, M, d_z)

        diff = m_sel.unsqueeze(1) - m_sel.unsqueeze(0)  # (M, M, d_z)

        # Mahalanobis term: 0.125 * sum_d (m_k - m_l)^2 / S_avg_d
        mahal = 0.125 * (diff ** 2 / S_avg).sum(dim=2)  # (M, M)

        # Log-det term: 0.5 * [sum_d log(S_avg_d) - 0.5*(sum_d log(S_k_d) + sum_d log(S_l_d))]
        log_det_avg = torch.log(S_avg).sum(dim=2)        # (M, M)
        log_det_k = torch.log(S_sel).sum(dim=1)          # (M,)
        log_det_l = log_det_k
        log_det_term = 0.5 * (log_det_avg - 0.5 * (log_det_k.unsqueeze(1) + log_det_l.unsqueeze(0)))

        D_B = mahal + log_det_term  # (M, M)

        # pi_k * pi_l * exp(-D_B) for k < l
        pi_outer = pi_sel.unsqueeze(1) * pi_sel.unsqueeze(0)  # (M, M)
        overlap_matrix = pi_outer * torch.exp(-D_B)

        # Upper triangle only (k < l)
        penalty = torch.triu(overlap_matrix, diagonal=1).sum()
        return penalty

    def state_dict(self) -> dict:
        return {
            "a": self.a.cpu(), "b": self.b.cpu(),
            "m": self.m.cpu(), "S": self.S.cpu(),
            "K": self.K, "d_z": self.d_z,
            "alpha": self.alpha, "s_min": self.s_min,
        }

    def load_state_dict(self, d: dict):
        self.a = d["a"].to(self.device)
        self.b = d["b"].to(self.device)
        self.m = d["m"].to(self.device)
        self.S = d["S"].to(self.device)


# ---------------------------------------------------------------------------
# Encoder / Decoder / Model
# ---------------------------------------------------------------------------

class KmerDPVAEEncoder(nn.Module):
    """Encoder: k-mer profile -> (mu, sigma2).

    Backbone: Linear(256,512) -> BatchNorm -> Sigmoid -> Linear(512, d_z)
    Same architecture as NonLinearEmbedder (without dropout), so contrastive
    pretrained weights can be loaded directly.

    The variance head (W_sigma) is separate and only used during VAE training.
    """

    def __init__(self, kmer_dim: int = 256, d_z: int = 16, eps: float = 1e-6):
        super().__init__()
        # Backbone (matches NonLinearEmbedder architecture)
        self.linear1 = nn.Linear(kmer_dim, 512)
        self.batch1 = nn.BatchNorm1d(512)
        self.activation1 = nn.Sigmoid()
        self.linear2 = nn.Linear(512, d_z)

        # Variance head (VAE only, not used in contrastive pretraining)
        self.W_sigma = nn.Linear(512, d_z)
        self.eps = eps

    def forward(self, kmer_profile: torch.Tensor):
        """
        Returns:
            z: (B, d_z) reparameterized sample (or mu at eval)
            mu: (B, d_z) posterior mean
            sigma2: (B, d_z) diagonal variance
        """
        h = self.activation1(self.batch1(self.linear1(kmer_profile)))
        mu = self.linear2(h)
        sigma2 = F.softplus(self.W_sigma(h)) + self.eps

        if self.training:
            z = mu + torch.sqrt(sigma2) * torch.randn_like(sigma2)
        else:
            z = mu

        return z, mu, sigma2

    def load_contrastive_weights(self, nonlinear_embedder):
        """Load backbone weights from a pretrained NonLinearEmbedder.

        Copies linear1, batch1, linear2 weights. W_sigma is left random.
        """
        self.linear1.load_state_dict(nonlinear_embedder.linear1.state_dict())
        self.batch1.load_state_dict(nonlinear_embedder.batch1.state_dict())
        self.linear2.load_state_dict(nonlinear_embedder.linear2.state_dict())


class KmerDPVAEDecoder(nn.Module):
    """MLP decoder: z -> reconstructed k-mer counts."""

    def __init__(self, d_z: int = 16, h_dim: int = 256, kmer_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 512),
            nn.GELU(),
            nn.Linear(512, kmer_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns predicted k-mer counts (B, kmer_dim)."""
        return self.net(z)


class KmerDPVAE(BaseEmbedder, nn.Module):
    """K-mer-only VAE with truncated DP-GMM prior.

    Input and reconstruction target are both normalized k-mer profiles.
    """

    def __init__(self, k: int = 4, d_z: int = 16, h_dim: int = 256,
                 K_max: int = 64, alpha: float = 1.0,
                 device: str = "cpu", seed: int = 26042024):
        nn.Module.__init__(self)

        self._k = k
        self._d_z = d_z
        self._h_dim = h_dim
        self._K_max = K_max
        self._alpha = alpha
        self._device_str = device
        self._seed = seed

        kmer_dim = 4 ** k
        self._kmer_dim = kmer_dim

        self._feature_extractor = KmerFeatureExtractor(
            k=k, alphabet=["A", "C", "G", "T"], normalized=False
        )

        self.encoder = KmerDPVAEEncoder(kmer_dim=kmer_dim, d_z=d_z)
        self.decoder = KmerDPVAEDecoder(d_z=d_z, h_dim=h_dim, kmer_dim=kmer_dim)
        self.dpgmm = DPGMM(K_max=K_max, d_z=d_z, alpha=alpha, device=device)

        torch.manual_seed(seed)

    def _get_kwargs(self) -> dict:
        return {
            "k": self._k, "d_z": self._d_z, "h_dim": self._h_dim,
            "K_max": self._K_max, "alpha": self._alpha, "seed": self._seed,
        }

    def reconstruction_loss(self, pred: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and target k-mer counts.

        Args:
            pred: (B, kmer_dim) decoder output
            target: (B, kmer_dim) raw k-mer counts
        Returns:
            scalar mean MSE.
        """
        return F.mse_loss(pred, target)

    def compute_elbo(self, kmer_profiles: torch.Tensor, use_dpgmm: bool = True,
                     beta: float = 1.0,
                     overlap_lambda: float = 0.0) -> tuple[torch.Tensor, dict]:
        """Compute negative ELBO for a batch of k-mer profiles.

        Args:
            kmer_profiles: (B, kmer_dim) k-mer profiles
            use_dpgmm: if False, use N(0,I) prior (warm-up phase)
            beta: KL weight for warm-up annealing
            overlap_lambda: weight for Bhattacharyya overlap penalty
        Returns:
            (neg_elbo, metrics_dict)
        """
        z, mu, sigma2 = self.encoder(kmer_profiles)
        pred = self.decoder(z)
        recon_loss = self.reconstruction_loss(pred, kmer_profiles)

        metrics = {"recon_loss": recon_loss.item()}

        if use_dpgmm:
            r = self.dpgmm.compute_responsibilities(mu.detach(), sigma2.detach())

            component_ll = self.dpgmm.expected_log_component_density(mu, sigma2, r)
            stick_term = self.dpgmm.expected_log_pi_term(r)
            post_entropy = self.dpgmm.posterior_entropy(sigma2)
            assign_entropy = self.dpgmm.assignment_entropy(r)
            stick_kl = self.dpgmm.kl_stick_breaking()

            B = mu.shape[0]
            neg_elbo = recon_loss - (1.0 / B) * (
                component_ll + stick_term + post_entropy + assign_entropy - stick_kl
            )

            # Overlap penalty: penalize Gaussian component overlap
            if overlap_lambda > 0:
                overlap = self.dpgmm.overlap_penalty()
                neg_elbo = neg_elbo + overlap_lambda * overlap
                metrics["overlap"] = overlap.item()
            else:
                metrics["overlap"] = 0.0

            metrics.update({
                "component_ll": component_ll.item() / B,
                "stick_term": stick_term.item() / B,
                "posterior_entropy": post_entropy.item() / B,
                "assignment_entropy": assign_entropy.item() / B,
                "stick_kl": stick_kl.item(),
            })
        else:
            kl = 0.5 * (sigma2 + mu ** 2 - 1 - torch.log(sigma2)).sum(dim=1).mean()
            neg_elbo = recon_loss + beta * kl
            metrics["kl_standard"] = kl.item()
            metrics["overlap"] = 0.0

        metrics["neg_elbo"] = neg_elbo.item()
        return neg_elbo, metrics

    def embed(self, inputs: list[str]) -> EmbeddingResult:
        """Encode DNA sequences via k-mer profiles."""
        device = next(self.parameters()).device
        self.eval()

        kmer_profiles = self._feature_extractor.extract_batch(inputs)
        kmer_tensor = torch.from_numpy(kmer_profiles.astype(np.float32)).to(device)

        with torch.no_grad():
            _, mu, sigma2 = self.encoder(kmer_tensor)

        return EmbeddingResult(
            mean=mu.cpu().numpy(),
            variance=sigma2.cpu().numpy(),
            distribution="gaussian",
        )

    def get_cluster_assignments(self, inputs: list[str]) -> np.ndarray:
        device = next(self.parameters()).device
        result = self.embed(inputs)
        mu = torch.from_numpy(result.mean).to(device)
        sigma2 = torch.from_numpy(result.variance).to(device)
        r = self.dpgmm.compute_responsibilities(mu, sigma2)
        return r.argmax(dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save([
            self._get_kwargs(),
            self.state_dict(),
            self.dpgmm.state_dict(),
        ], path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "KmerDPVAE":
        data = torch.load(path, map_location=device, weights_only=False)
        model_kwargs, nn_state, dpgmm_state = data
        model_kwargs["device"] = device
        model_kwargs.update(kwargs)
        model = cls(**model_kwargs)
        model.load_state_dict(nn_state)
        model.dpgmm.load_state_dict(dpgmm_state)
        model.dpgmm.to(device)
        model.to(device)
        return model

    @property
    def default_metric(self) -> str:
        return "l2"


def warmup_phase(
    model: KmerDPVAE,
    kmer_profiles: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    kl_anneal_epochs: int = 20,
    verbose: bool = True,
    print_every: int = 10,
) -> list[float]:
    """Phase 1: Train encoder+decoder with N(0,I) prior and KL annealing.

    Returns list of per-epoch losses.
    """
    model = model.to(device)
    model.train()

    dataloader = DataLoader(
        TensorDataset(kmer_profiles), batch_size=batch_size,
        shuffle=True, drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr,
    )

    losses = []
    if verbose:
        print(f"=== Phase 1: Warm-up ({epochs} epochs) ===")

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        beta = min(1.0, (epoch + 1) / max(kl_anneal_epochs, 1))

        for (batch_kmer,) in dataloader:
            batch_kmer = batch_kmer.to(device)
            optimizer.zero_grad()
            neg_elbo, metrics = model.compute_elbo(
                batch_kmer, use_dpgmm=False, beta=beta
            )
            neg_elbo.backward()
            optimizer.step()
            epoch_loss += metrics["neg_elbo"]
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and (epoch + 1) % print_every == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  beta={beta:.3f}")

    return losses


def init_dpgmm_phase(
    model: KmerDPVAE,
    kmer_profiles: torch.Tensor,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
):
    """Phase 2: Collect encoder means and initialize DP-GMM via k-means."""
    model = model.to(device)
    model.dpgmm.to(device)
    model.eval()

    eval_loader = DataLoader(
        TensorDataset(kmer_profiles), batch_size=batch_size,
        shuffle=False, drop_last=False,
    )

    all_mu = []
    with torch.no_grad():
        for (batch_kmer,) in eval_loader:
            batch_kmer = batch_kmer.to(device)
            _, mu, _ = model.encoder(batch_kmer)
            all_mu.append(mu)
    all_mu = torch.cat(all_mu, dim=0)
    model.dpgmm.init_from_data(all_mu)

    if verbose:
        print(f"=== Phase 2: Initialized {model.dpgmm.K} components "
              f"from {all_mu.shape[0]} samples ===")


def joint_phase(
    model: KmerDPVAE,
    kmer_profiles: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    mixture_update_interval: int = 1,
    overlap_lambda: float = 0.0,
    save_path: str = None,
    verbose: bool = True,
    print_every: int = 10,
) -> dict:
    """Phase 3: Joint SGD on ELBO + closed-form DP-GMM updates.

    Returns dict of per-epoch metric lists.
    """
    model = model.to(device)
    model.dpgmm.to(device)
    model.train()

    dataset = TensorDataset(kmer_profiles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr,
    )

    history = {"neg_elbo": [], "recon_loss": [], "component_ll": [],
               "stick_term": [], "posterior_entropy": [],
               "assignment_entropy": [], "stick_kl": [], "overlap": []}

    if verbose:
        print(f"=== Phase 3: Joint training ({epochs} epochs, "
              f"overlap_lambda={overlap_lambda}) ===")

    for epoch in range(epochs):
        epoch_metrics = {k: 0.0 for k in history}
        n_batches = 0

        for (batch_kmer,) in dataloader:
            batch_kmer = batch_kmer.to(device)
            optimizer.zero_grad()
            neg_elbo, metrics = model.compute_elbo(
                batch_kmer, use_dpgmm=True, beta=1.0,
                overlap_lambda=overlap_lambda,
            )
            neg_elbo.backward()
            optimizer.step()
            for k in history:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            n_batches += 1

        for k in history:
            history[k].append(epoch_metrics[k] / max(n_batches, 1))

        if verbose and (epoch + 1) % print_every == 0:
            h = history
            print(f"  Epoch {epoch+1}/{epochs}  "
                  f"elbo={h['neg_elbo'][-1]:.2f}  "
                  f"recon={h['recon_loss'][-1]:.2f}  "
                  f"comp_ll={h['component_ll'][-1]:.2f}  "
                  f"stick={h['stick_term'][-1]:.2f}  "
                  f"H_post={h['posterior_entropy'][-1]:.2f}  "
                  f"H_assign={h['assignment_entropy'][-1]:.2f}  "
                  f"kl_stick={h['stick_kl'][-1]:.2f}  "
                  f"overlap={h['overlap'][-1]:.4f}")

        # Periodic mixture update
        if (epoch + 1) % mixture_update_interval == 0:
            model.eval()
            all_mu_list, all_sigma2_list = [], []
            with torch.no_grad():
                for (batch_kmer,) in eval_loader:
                    batch_kmer = batch_kmer.to(device)
                    _, mu, sigma2 = model.encoder(batch_kmer)
                    all_mu_list.append(mu)
                    all_sigma2_list.append(sigma2)
            all_mu_t = torch.cat(all_mu_list, dim=0)
            all_sigma2_t = torch.cat(all_sigma2_list, dim=0)
            r = model.dpgmm.compute_responsibilities(all_mu_t, all_sigma2_t)
            model.dpgmm.update_parameters(all_mu_t, all_sigma2_t, r)
            model.train()

    if save_path:
        model.save(save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return history
