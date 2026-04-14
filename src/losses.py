import torch
import torch.nn as nn

class QLIKELoss(nn.Module):
    # qlike (gaussian quasi-likelihood) adapted for log-variance inputs
    # standard form: L = (RV/h) - log(RV/h) - 1
    # since inputs are already log-vars: ratio RV/h = exp(y - y_hat)
    # so: L = exp(y - y_hat) - (y - y_hat) - 1
    def __init__(self):
        super(QLIKELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # residual in log space
        log_diff = y_true - y_pred          
        loss = torch.exp(log_diff) - log_diff - 1
        return torch.mean(loss)

class OriginalAFBRLLoss(nn.Module):
    # original bounded fuzzy loss (baseline variant)
    # uses a relative error e_t bounded strictly in [0,1]:
    #   e_t = |y - y_hat| / (|y - y_hat| + |y - y_bench|)
    # e_t=0 means perfect pred, e_t=1 means pred is worse than the garch bench
    # known flaw: as error -> inf, gradient -> 0 (saturation)
    def __init__(self, strict_sigma=0.2, loose_sigma=0.2):
        super(OriginalAFBRLLoss, self).__init__()
        self.strict_sigma = strict_sigma  # narrow gaussian -> harsher penalty for underpreds
        self.loose_sigma = loose_sigma    # wider gaussian -> softer penalty for overpreds
        self.eps = 1e-8

    def forward(self, y_pred, y_true, y_bench):
        # bounded relative error vs garch benchmark
        numerator = torch.abs(y_true - y_pred)
        denominator = numerator + torch.abs(y_true - y_bench) + self.eps
        e_t = numerator / denominator

        # asymmetric sigma: underpreds get strict_sigma, overpreds get loose_sigma
        sigmas = torch.where(y_pred < y_true,
                             torch.tensor(self.strict_sigma, device=y_pred.device),
                             torch.tensor(self.loose_sigma, device=y_pred.device))

        # gaussian membership: mu=1 when e_t=0 (perfect), mu->0 as error grows
        mu = torch.exp(-(e_t**2) / (2 * sigmas**2))

        # minimising loss = maximising fuzzymembership
        return torch.mean(1.0 - mu)