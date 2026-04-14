import torch
import torch.nn as nn


class AFBRLLoss(nn.Module):
    
    def __init__(self, strict_sigma=0.15, loose_sigma=0.3, epsilon=1e-8, recovery_weight=1.0):
        super(AFBRLLoss, self).__init__()
        self.strict_sigma = strict_sigma 
        self.loose_sigma = loose_sigma 
        self.epsilon = epsilon 
        self.recovery_weight = recovery_weight

    def forward(self, y_pred, y_true, y_benchmark):
        """
        compute afbrl loss.
        
        args:
            y_pred: model predictions (log variance)
            y_true: ground truth (log variance)
            y_benchmark: garch predictions (log variance)
        
        returns:
            scalar loss value.
        """
        # compute absolute errors
        model_error = torch.abs(y_true - y_pred)
        benchmark_error = torch.abs(y_true - y_benchmark)
        
        # relative error bounded in [0, 1]
        e_t = model_error / (model_error + benchmark_error + self.epsilon)
        
        # asymmetric penalty that is stricter for underprediction
        sigma_t = torch.where(
            y_pred < y_true,
            torch.tensor(self.strict_sigma, device=y_pred.device),
            torch.tensor(self.loose_sigma, device=y_pred.device)
        )
        
        # gaussian fuzzy membership function 
        membership = torch.exp(-(e_t ** 2) / (2 * (sigma_t ** 2)))
        
        # fuzzy loss component
        loss_fuzzy = 1.0 - membership
        
        # recovery loss prevents vanishing gradients during extreme events
        loss_recovery = (1.0 - membership) * model_error
        
        return torch.mean(loss_fuzzy + self.recovery_weight * loss_recovery)