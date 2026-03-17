import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from embedders.base import EmbeddingResult
from utils.progress import pbar

class LaplaceLastLayerEmbedder:
    def __init__(self, base_model, prior_precision: float = 1.0):
        if not hasattr(base_model, "linear2"):
            raise ValueError("base_model must have a 'linear2' attribute")
        
        self.base_model = base_model
        self.prior_precision = prior_precision
        self._fitted = False
        
        # Storage for Eigen-decomposition factors
        self.Q_A = None
        self.Q_B = None
        self.S_A = None
        self.S_B = None

    def fit(self, dataset, loss_fn, batch_size: int = 64,
            device: str = "cpu", loss_name: str = "bern", verbose: bool = True,
            hessian_factorization: str = "ggn"):
        """Fit the KFAC Laplace approximation on a contrastive dataset.

        Parameters
        ----------
        hessian_factorization : str
            How to compute the B (output) factor of KFAC:
            - ``"ggn"``: Use the analytical Hessian d²L/dz² (GGN).
              Gives non-degenerate curvature for bern/poisson losses
              even at convergence. **Recommended.**
            - ``"ef"``: Use the empirical Fisher g·gᵀ (gradient outer
              product). Can underestimate curvature at convergence
              because contrastive loss gradients vanish for
              well-classified pairs.
        """
        valid_hf = ("ggn", "ef")
        if hessian_factorization not in valid_hf:
            raise ValueError(
                f"hessian_factorization must be one of {valid_hf}, "
                f"got {hessian_factorization!r}"
            )

        self._loss_name = loss_name

        model = self.base_model
        model.eval()
        dev = torch.device(device) if isinstance(device, str) else device
        model.to(dev)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Accumulators (Using Sums, not Means, to scale correctly with N)
        d_in = model.linear2.in_features + 1 # +1 for bias
        d_out = model.linear2.out_features

        A_sum = torch.zeros(d_in, d_in, device=dev)
        B_sum = torch.zeros(d_out, d_out, device=dev)
        n_samples = 0

        use_ggn = hessian_factorization == "ggn"
        if use_ggn and loss_name not in ("bern", "poisson"):
            warnings.warn(
                f"GGN Hessian not implemented for '{loss_name}' loss; "
                "falling back to empirical Fisher.",
                stacklevel=2,
            )
            use_ggn = False

        for batch in pbar(loader, desc="LLA fit", disable=not verbose):
            # 1. Handle Inputs — flatten (batch, 1+neg, feat) → (batch*(1+neg), feat)
            #    Same reshape as train_contrastive() in train.py.
            left, right, labels = batch
            left = left.reshape(-1, left.shape[-1]).to(dev)
            right = right.reshape(-1, right.shape[-1]).to(dev)
            labels = labels.reshape(-1).to(dev)

            # 2. Hook to capture Inputs AND Outputs of the linear layer.
            #    encoder() is called twice (left then right), so we accumulate both.
            captured = {'h': [], 'z': []}
            def _hook(module, inp, out):
                captured['h'].append(inp[0].detach())
                captured['z'].append(out.detach())

            handle = model.linear2.register_forward_hook(_hook)

            # Forward pass (no grad needed for model weights)
            with torch.no_grad():
                _ = model(left, right)
            handle.remove()

            # 3. Process captured data — concatenate left + right activations
            h_batch = torch.cat(captured['h'], dim=0)  # (2*M, hidden)
            z_batch = torch.cat(captured['z'], dim=0)  # (2*M, d_out)

            # Update A (Covariance of inputs to linear2, with bias column)
            h_aug = torch.cat([h_batch, torch.ones(h_batch.shape[0], 1, device=dev)], dim=1)
            A_sum += h_aug.T @ h_aug

            M = left.shape[0]
            z_left = z_batch[:M]
            z_right = z_batch[M:]

            if use_ggn:
                # ---- GGN: analytical Hessian d²L/dz² per pair ----
                # For a linear last layer, the GGN B factor is the average
                # Hessian of the loss w.r.t. the layer output z, NOT the
                # gradient outer product.
                #
                # Derivation (bern loss, L = BCE(exp(-||d||²), y)):
                #   H_m = 4·c_m · d_m·d_mᵀ + 2·s_m · I
                #   c = (1-y)·p/(1-p)²     s = y - (1-y)·p/(1-p)
                #
                # Key: for y=1 (positive pair), H = 2·I regardless of
                # convergence, giving non-zero curvature that the empirical
                # Fisher misses entirely.
                d_vec = z_left - z_right                    # (M, d_out)
                dist_sq = (d_vec ** 2).sum(dim=1)           # (M,)
                p = torch.clamp(torch.exp(-dist_sq), 1e-7, 1.0 - 1e-7)
                y = labels.float()

                if loss_name == "bern":
                    one_m_p = 1.0 - p
                    c = (1.0 - y) * p / (one_m_p ** 2)     # (M,)
                    s = y - (1.0 - y) * p / one_m_p         # (M,)
                else:  # poisson
                    c = p                                    # (M,)
                    s = y - p                                # (M,)

                # Clamp for numerical stability (large c when p→1 on negatives)
                c = torch.clamp(c, max=1000.0)

                # The full Hessian H_m = 4·c·d·dᵀ + 2·s·I is NOT PSD for
                # negative pairs (s < 0 when y=0).  With 20 negatives per
                # positive, the negative diagonal terms dominate, making
                # B_mean non-PSD and destroying curvature after clamping.
                #
                # Fix: keep only the PSD part by clamping s >= 0.
                #   y=1 (positive): s=1 → 2·I    (the key non-zero curvature)
                #   y=0 (negative): s<0 → 0       (keep rank-1 4·c·d·dᵀ only)
                s = torch.clamp(s, min=0.0)

                # B_batch = Σ_m H_m^+ = 4·Σ c_m·d_m·d_mᵀ + 2·(Σ s_m)·I
                d_weighted = d_vec * torch.sqrt(c).unsqueeze(1)  # (M, d_out)
                B_batch = (4.0 * d_weighted.T @ d_weighted
                           + 2.0 * s.sum() * torch.eye(d_out, device=dev))

                # Each pair → 2 forward passes (left, right) with same Hessian
                B_sum += 2.0 * B_batch
            else:
                # ---- Empirical Fisher: B = Σ g_i·g_iᵀ ----
                z_batch.requires_grad = True
                z_left_g = z_batch[:M]
                z_right_g = z_batch[M:]

                loss = loss_fn(z_left_g, z_right_g, labels, name=loss_name)
                g_combined = torch.autograd.grad(loss, z_batch)[0]
                # Undo mean reduction: loss averages over M pairs
                g_combined = g_combined * M

                B_sum += g_combined.T @ g_combined

            n_samples += h_batch.shape[0]

            # Cleanup
            del h_batch, z_batch, captured

        # 5. Eigendecomposition of KFAC factors.
        # Standard KFAC Laplace: precision P = N * A_mean ⊗ B_mean + prior * I
        A_mean = (A_sum / n_samples).cpu()
        B_mean = (B_sum / n_samples).cpu()

        # Symmetrise and add small jitter for numerical stability —
        # BatchNorm can produce near-singular input covariance.
        A_mean = 0.5 * (A_mean + A_mean.T)
        B_mean = 0.5 * (B_mean + B_mean.T)
        jitter = 1e-6 * torch.eye(A_mean.shape[0])
        A_mean = A_mean + jitter
        jitter_b = 1e-6 * torch.eye(B_mean.shape[0])
        B_mean = B_mean + jitter_b

        s_a, q_a = torch.linalg.eigh(A_mean)
        s_b, q_b = torch.linalg.eigh(B_mean)

        # Enforce positivity for numerical stability
        s_a = torch.clamp(s_a, min=1e-6)
        s_b = torch.clamp(s_b, min=1e-6)

        self.S_A = s_a
        self.S_B = s_b
        self.Q_A = q_a
        self.Q_B = q_b
        self.n_data = n_samples
        self._fitted = True

        if verbose:
            hf_label = "GGN" if use_ggn else "EF"
            print(f"LLA fitted on {n_samples} samples (B: {hf_label}).")
            print(f"  S_A range: [{s_a.min():.2e}, {s_a.max():.2e}]")
            print(f"  S_B range: [{s_b.min():.2e}, {s_b.max():.2e}]")

    def optimize_prior(self, n_steps: int = 100, verbose: bool = True,
                        method: str = "mackay", calibration_data=None,
                        tau_range=(1e-2, 1e4), n_tau: int = 60):
        """Tune prior_precision after fitting.

        Parameters
        ----------
        method : str
            ``"mackay"`` — MacKay (1992) fixed-point iteration maximizing the
            log marginal likelihood.  Fast but can over-regularize when
            last-layer weights are small (e.g. after BatchNorm).

            ``"mackay_corrected"`` — Like MacKay but uses the function-space
            weight norm ``||w_eff||^2 = tr(W A_mean W^T)`` instead of
            ``||w||^2``, accounting for the input distribution.

            ``"cv"`` — Maximize the coefficient of variation (CV) of total
            predictive variance across calibration samples.  Picks the tau
            that makes variance most discriminative across inputs.
            Requires ``calibration_data``.
        calibration_data : array-like, optional
            Input data (same format as ``embed()``).  Required for
            ``method="cv"``.  A random subset of 500 is used if larger.
        tau_range : tuple of float
            (min, max) range for the log-spaced tau sweep (``"cv"`` only).
        n_tau : int
            Number of tau candidates in the sweep (``"cv"`` only).
        n_steps : int
            Max iterations for MacKay (``"mackay"`` only).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before optimize_prior()")

        if method == "mackay":
            self._optimize_prior_mackay(n_steps=n_steps, verbose=verbose)
        elif method == "mackay_corrected":
            self._optimize_prior_mackay_corrected(n_steps=n_steps, verbose=verbose)
        elif method == "cv":
            if calibration_data is None:
                raise ValueError("calibration_data is required for method='cv'")
            self._optimize_prior_cv(
                calibration_data, tau_range=tau_range, n_tau=n_tau,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"method must be 'mackay', 'mackay_corrected', or 'cv', "
                f"got {method!r}"
            )

    # ------------------------------------------------------------------
    def _optimize_prior_mackay(self, n_steps: int = 100, verbose: bool = True):
        """MacKay fixed-point iteration on log marginal likelihood."""
        if self._loss_name == "hinge":
            warnings.warn(
                "optimize_prior: hinge loss is not a log-likelihood. "
                "Marginal likelihood optimisation is an approximation "
                "and may produce poorly calibrated prior_precision.",
                stacklevel=2,
            )

        W = self.base_model.linear2.weight.detach().cpu()
        b = self.base_model.linear2.bias.detach().cpu()
        w_norm_sq = (W ** 2).sum().item() + (b ** 2).sum().item()

        N = self.n_data
        kron_eigs = N * torch.outer(self.S_A, self.S_B)
        d = kron_eigs.numel()

        tau = self.prior_precision
        for step in range(n_steps):
            gamma = (kron_eigs / (tau + kron_eigs)).sum().item()
            tau_new = gamma / max(w_norm_sq, 1e-12)

            if abs(tau_new - tau) / max(abs(tau), 1e-12) < 1e-6:
                if verbose:
                    print(f"  MacKay converged at step {step + 1}: "
                          f"tau={tau_new:.4f}, gamma={gamma:.1f}/{d}")
                break
            tau = tau_new
        else:
            if verbose:
                print(f"  MacKay after {n_steps} steps: "
                      f"tau={tau:.4f}, gamma={gamma:.1f}/{d}")

        self.prior_precision = tau
        if verbose:
            print(f"  optimized prior_precision = {self.prior_precision:.4f}")

    # ------------------------------------------------------------------
    def _optimize_prior_mackay_corrected(self, n_steps: int = 100,
                                          verbose: bool = True):
        """MacKay iteration with function-space weight norm correction.

        Replaces ``||w||^2`` with ``||w_eff||^2 = tr(W_aug @ A_mean @ W_aug^T)``
        which measures the actual output energy accounting for the input
        distribution (important when BatchNorm makes raw weights small).
        """
        W = self.base_model.linear2.weight.detach().cpu()
        b = self.base_model.linear2.bias.detach().cpu()
        W_aug = torch.cat([W, b.unsqueeze(1)], dim=1)  # (d_out, d_in)

        A_mean = self.Q_A @ torch.diag(self.S_A) @ self.Q_A.T
        WA = W_aug @ A_mean
        w_eff_sq = (WA * W_aug).sum().item()  # tr(W_aug A_mean W_aug^T)

        N = self.n_data
        kron_eigs = N * torch.outer(self.S_A, self.S_B)
        d = kron_eigs.numel()

        tau = self.prior_precision
        for step in range(n_steps):
            gamma = (kron_eigs / (tau + kron_eigs)).sum().item()
            tau_new = gamma / max(w_eff_sq, 1e-12)

            if abs(tau_new - tau) / max(abs(tau), 1e-12) < 1e-6:
                if verbose:
                    print(f"  MacKay-corrected converged at step {step + 1}: "
                          f"tau={tau_new:.4f}, gamma={gamma:.1f}/{d}")
                break
            tau = tau_new
        else:
            if verbose:
                print(f"  MacKay-corrected after {n_steps} steps: "
                      f"tau={tau:.4f}, gamma={gamma:.1f}/{d}")

        self.prior_precision = tau
        if verbose:
            w_norm_sq = (W ** 2).sum().item() + (b ** 2).sum().item()
            print(f"  ||w||^2={w_norm_sq:.2f}  ||w_eff||^2={w_eff_sq:.2f}")
            print(f"  optimized prior_precision = {self.prior_precision:.4f}")

    # ------------------------------------------------------------------
    def _optimize_prior_cv(self, calibration_data, tau_range=(1e-2, 1e4),
                           n_tau: int = 60, verbose: bool = True):
        """Pick tau that maximizes CV of total predictive variance.

        Only the cheap eigenvalue inversion changes per candidate tau;
        one forward pass through the network suffices for all candidates.
        """
        # Subsample for speed
        cal = calibration_data
        if len(cal) > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(cal), 500, replace=False)
            cal = cal[idx]

        # Single forward pass to capture last-layer activations
        model = self.base_model
        model.eval()
        captured = {}
        def _hook(module, inp, out):
            captured['h'] = inp[0].detach().cpu()
        handle = model.linear2.register_forward_hook(_hook)
        with torch.no_grad():
            model.embed(cal)
        handle.remove()

        H = captured['h']
        H_aug = torch.cat([H, torch.ones(H.shape[0], 1)], dim=1)
        H_rot_sq = (H_aug @ self.Q_A) ** 2   # (n_cal, d_in)
        Q_B_sq = self.Q_B ** 2                # (d_out, d_out)

        N = self.n_data
        kron_base = N * torch.outer(self.S_A, self.S_B)  # (d_in, d_out)

        taus = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_tau)
        best_cv, best_tau = -1.0, taus[0]

        for tau_cand in taus:
            eigenvals_inv = 1.0 / (kron_base + tau_cand)
            var_out = H_rot_sq @ eigenvals_inv @ Q_B_sq.T  # (n_cal, d_out)
            total_var = var_out.sum(dim=1)
            mean_tv = total_var.mean().item()
            if mean_tv > 0:
                cv = (total_var.std() / total_var.mean()).item()
            else:
                cv = 0.0
            if cv > best_cv:
                best_cv = cv
                best_tau = float(tau_cand)

        self.prior_precision = best_tau
        if verbose:
            print(f"  CV-optimized prior: tau={best_tau:.4f} (CV={best_cv:.4f})")

    def embed(self, inputs) -> EmbeddingResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")

        model = self.base_model
        model.eval()
        
        captured = {}
        def _hook(module, inp, out):
            captured['h'] = inp[0].detach().cpu()
            
        handle = model.linear2.register_forward_hook(_hook)
        emb_res = model.embed(inputs) # Get deterministic mean
        handle.remove()
        
        # H: (N, d_in)
        H = captured['h']
        H_aug = torch.cat([H, torch.ones(H.shape[0], 1)], dim=1)
        
        # Predictive Variance Calculation
        # Var(f(x)) = diag( H_aug @ Sigma @ H_aug.T )
        # where Sigma = (A \otimes B + prior * I)^-1
        
        # Ensure KFAC factors are on CPU (H_aug is on CPU from the hook)
        Q_A = self.Q_A.cpu()
        Q_B = self.Q_B.cpu()
        S_A = self.S_A.cpu()
        S_B = self.S_B.cpu()

        # We calculate this using the eigen-basis to avoid forming large matrices.
        # 1. Project inputs into A's eigen-basis
        # H_rot = H_aug @ Q_A   [N, d_in]
        H_rot = H_aug @ Q_A
        
        # 2. Compute the inverse eigenvalues of the Hessian
        # Lambda_inv = 1 / (S_A[i] * S_B[j] + prior)
        # We need the specific interaction for the diagonal variance.
        
        # Precision eigenvalues: N * outer(s_a, s_b) + prior
        # s_a, s_b are eigenvalues of A_mean and B_mean respectively
        eigenvals = self.n_data * torch.outer(S_A, S_B) + self.prior_precision
        eigenvals_inv = 1.0 / eigenvals # (d_in, d_out)
        
        # 3. Compute per-dimension variance
        # Var_j(x) = sum_i ( (H_rot_i)^2 * (Q_B_jj)^2 * inv_lambda_ij )
        # However, we want the variance for the output basis (usually canonical).
        # Approximating that output basis Q_B aligns with canonical (often true for diagonal B) 
        # or computing full diagonal.
        
        # Exact diagonal computation:
        # Var(x)_k = sum_{i, j} (H_rot_{ni}^2) * (Q_B_{kj}^2) * (1 / (S_A_i * S_B_j + prior))
        
        H_rot_sq = H_rot ** 2 # (N, d_in)
        Q_B_sq = Q_B ** 2 # (d_out, d_out)
        
        # (N, d_in) @ (d_in, d_out) -> (N, d_out)
        # Matrix multiply handles the summation over i (index of A)
        variance = H_rot_sq @ eigenvals_inv 
        
        # Now apply the output rotation (mixing B's modes into output dimensions)
        # If we want variance in the CANONICAL output space:
        variance = variance @ Q_B_sq.T
        
        return EmbeddingResult(mean=emb_res.mean, variance=variance.numpy())

    @property
    def is_probabilistic(self):
        return True
