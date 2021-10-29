import torch


def kl_coefficients(num_scales, max_groups_per_scale, min_groups_per_scale):
    coeffs = []
    for i in range(num_scales):
        scale_powers_of_two = 2 ** (num_scales - 1 - i)
        num_groups = int(max(max_groups_per_scale // scale_powers_of_two, min_groups_per_scale))
        coeffs.append((2 ** i) ** 2 / num_groups * torch.ones(num_groups))
    coeffs = torch.cat(coeffs, dim=0)
    coeffs = coeffs / torch.min(coeffs)
    return coeffs.unsqueeze(0)


def differentiable_clamp_five(x):
    return torch.tanh(x / 5.) * 5.


def exp_plus_eps(x, eps=1e-2):
    return torch.exp(x) + eps


def calc_kl_and_z(mu_p, log_sig_p, mu_q, log_sig_q):
    mu_q = differentiable_clamp_five(mu_q)
    log_sig_q = differentiable_clamp_five(log_sig_q)
    sig_q = exp_plus_eps(log_sig_q)

    mu_p = differentiable_clamp_five(mu_p)
    log_sig_p = differentiable_clamp_five(log_sig_p)
    sig_p = exp_plus_eps(log_sig_p)

    first_term = (mu_q - mu_p) / sig_p
    second_term = sig_q / sig_p

    kl_per_var = 0.5 * (first_term * first_term + second_term * second_term) - 0.5 - torch.log(second_term)
    return torch.sum(kl_per_var, dim=[1, 2, 3]), torch.randn_like(mu_q) * sig_q + mu_q