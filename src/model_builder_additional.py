import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianConv2d, BayesianLinear
from blitz.utils import variational_estimator

##########################################################################################
#####                                                                                #####
#####                             FREQUENTIST MODEL                                  #####
#####                                                                                #####
##########################################################################################

# --- Separable Convolution Module ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
    
# --- Residual Block ---
class XceptionResBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        self.residual = nn.Sequential(
            SeparableConv2d(input_channels, output_channels), 
            nn.BatchNorm2d(output_channels, affine=True),
            nn.ReLU(),
            SeparableConv2d(output_channels, output_channels), 
            nn.BatchNorm2d(output_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=1, 
            stride=2, 
            padding=0
        )
        
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


# --- Customized Xception Model ---
class XceptionCustom(nn.Module): 
    def __init__(self, input_channels=3, filter_num=[8, 16, 32, 64, 128, 256, 512]):
        super().__init__()
        self.filter_num = filter_num 

        # Entry block
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8, 
                     kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU()
        )
        
        # Residual block sequence 
        blocks = []
        input_channels = 8
        for out_filters in self.filter_num:
            blocks.append(XceptionResBlock(input_channels, out_filters))
            input_channels = out_filters
        self.blocks = nn.ModuleList(blocks)

        # Final separable conv
        self.final_sepconv = nn.Sequential(
            SeparableConv2d(self.filter_num[-1], self.filter_num[-1]*2),
            nn.BatchNorm2d(self.filter_num[-1]*2, affine=True),
            nn.ReLU()
        )

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=self.filter_num[-1]*2, out_features=2, bias=False)
        )

    def forward(self, x: torch.Tensor):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_sepconv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


##########################################################################################
#####                                                                                #####
#####                               BAYESIAN MODELS                                  #####
#####                                                                                #####
##########################################################################################

# --- Helper function to create priors ---
def create_bayesian_layer(layer_type, *args, prior_sigma=0.1, **kwargs):
    """Create Bayesian layers with proper prior configuration"""
    try:
        # Try with explicit prior parameters
        if layer_type == 'linear':
            return BayesianLinear(*args, prior_mu=0.0, prior_sigma=prior_sigma, 
                                prior_pi=1.0, posterior_mu_init=0.0, 
                                posterior_rho_init=-3.0, **kwargs)
        elif layer_type == 'conv2d':
            return BayesianConv2d(*args, prior_mu=0.0, prior_sigma=prior_sigma,
                                prior_pi=1.0, posterior_mu_init=0.0, 
                                posterior_rho_init=-3.0, **kwargs)
    except TypeError:
        # Fallback to default BLiTZ parameters if custom prior fails
        if layer_type == 'linear':
            return BayesianLinear(*args, **kwargs)
        elif layer_type == 'conv2d':
            return BayesianConv2d(*args, **kwargs)


def create_prior_dict(prior_mu=0.0, prior_sigma=0.1, prior_pi=1.0, posterior_rho_init=-3.0):
    """Create a prior dictionary for Bayesian layers - may not be used if BLiTZ API differs"""
    return {
        'prior_mu': prior_mu,
        'prior_sigma': prior_sigma, 
        'prior_pi': prior_pi,
        'posterior_mu_init': prior_mu,
        'posterior_rho_init': posterior_rho_init
    }


# --- Monte Carlo Dropout Version (Simple Baseline) ---
class XceptionMCDropout(nn.Module):
    """Xception with MC Dropout for uncertainty estimation"""
    def __init__(self, input_channels=3, filter_num=[8, 16, 32, 64, 128, 256, 512], dropout_p=0.1):
        super().__init__()
        self.filter_num = filter_num
        self.dropout_p = dropout_p

        # Entry block
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8, 
                     kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p)  # Spatial dropout
        )
        
        # Residual blocks with dropout
        blocks = []
        input_channels = 8
        for out_filters in self.filter_num:
            blocks.append(XceptionResBlockDropout(input_channels, out_filters, dropout_p))
            input_channels = out_filters
        self.blocks = nn.ModuleList(blocks)

        # Final layers
        self.final_sepconv = nn.Sequential(
            SeparableConv2d(self.filter_num[-1], self.filter_num[-1]*2),
            nn.BatchNorm2d(self.filter_num[-1]*2, affine=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=self.filter_num[-1]*2, out_features=2, bias=False)
        )

    def forward(self, x: torch.Tensor, mc_samples=1, return_all=False):
        if mc_samples == 1:
            return self._single_forward(x)
        
        # Multiple forward passes for uncertainty
        outputs = []
        for _ in range(mc_samples):
            outputs.append(self._single_forward(x))
        
        if return_all:
            return torch.stack(outputs)
        else:
            return torch.mean(torch.stack(outputs), dim=0)
    
    def _single_forward(self, x):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_sepconv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class XceptionResBlockDropout(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout_p=0.1):
        super().__init__()
        
        self.residual = nn.Sequential(
            SeparableConv2d(input_channels, output_channels), 
            nn.BatchNorm2d(output_channels, affine=True),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p),
            SeparableConv2d(output_channels, output_channels), 
            nn.BatchNorm2d(output_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=1, 
            stride=2, 
            padding=0
        )
        
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


# --- Hybrid Bayesian Version (Recommended Starting Point) ---
@variational_estimator
class XceptionHybrid(nn.Module):
    """Xception with only classifier being Bayesian - good starting point"""
    def __init__(self, input_channels=3, filter_num=[8, 16, 32, 64, 128, 256, 512], prior_config=None):
        super().__init__()
        self.filter_num = filter_num
        
        # Use default prior if none provided
        if prior_config is None:
            prior_config = create_prior_dict(prior_sigma=0.1, prior_pi=1.0)
        
        # Deterministic backbone (same as frequentist)
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8, 
                     kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU()
        )
        
        blocks = []
        input_channels = 8
        for out_filters in self.filter_num:
            blocks.append(XceptionResBlock(input_channels, out_filters))
            input_channels = out_filters
        self.blocks = nn.ModuleList(blocks)

        self.final_sepconv = nn.Sequential(
            SeparableConv2d(self.filter_num[-1], self.filter_num[-1]*2),
            nn.BatchNorm2d(self.filter_num[-1]*2, affine=True),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Only classifier is Bayesian - using helper function for robustness
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            create_bayesian_layer('linear', self.filter_num[-1]*2, 2, bias=False, prior_sigma=0.1)
        )

    def forward(self, x: torch.Tensor):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_sepconv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- Improved Separable Conv with Consistent Bayesian Design ---
class SeparableConv2d_Bayesian(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 bias=False, fully_bayesian=False, prior_config=None):
        super().__init__()
        
        if prior_config is None:
            prior_config = create_prior_dict()
        
        if fully_bayesian:
            # Both components are Bayesian
            self.depthwise = create_bayesian_layer('conv2d', in_channels, in_channels, 
                                                 kernel_size=kernel_size, stride=stride, 
                                                 padding=padding, groups=in_channels, 
                                                 bias=bias, prior_sigma=0.1)
            self.pointwise = create_bayesian_layer('conv2d', in_channels, out_channels, 
                                                 kernel_size=1, stride=1, padding=0, 
                                                 bias=bias, prior_sigma=0.1)
        else:
            # Only pointwise is Bayesian (more stable)
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, groups=in_channels, bias=bias
            )
            self.pointwise = create_bayesian_layer('conv2d', in_channels, out_channels, 
                                                 kernel_size=1, stride=1, padding=0, 
                                                 bias=bias, prior_sigma=0.1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# --- Improved Bayesian Residual Block ---
class XceptionResBlock_Bayesian(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, 
                 fully_bayesian=False, bayesian_shortcut=False, prior_config=None):
        super().__init__()
        
        if prior_config is None:
            prior_config = create_prior_dict()

        self.residual = nn.Sequential(
            SeparableConv2d_Bayesian(input_channels, output_channels, 
                                   fully_bayesian=fully_bayesian, prior_config=prior_config),
            nn.BatchNorm2d(output_channels, affine=True),
            nn.ReLU(),
            SeparableConv2d_Bayesian(output_channels, output_channels, 
                                   fully_bayesian=fully_bayesian, prior_config=prior_config),
            nn.BatchNorm2d(output_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Consistent shortcut design
        if bayesian_shortcut:
            self.shortcut = BayesianConv2d(
                input_channels, output_channels, kernel_size=1, 
                stride=2, padding=0, **prior_config
            )
        else:
            self.shortcut = nn.Conv2d(
                in_channels=input_channels, out_channels=output_channels, 
                kernel_size=1, stride=2, padding=0
            )
    
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


# --- Full Bayesian Xception (Advanced Version) ---
@variational_estimator
class XceptionFullBayesian(nn.Module):
    """Fully Bayesian Xception - use after hybrid version works well"""
    def __init__(self, input_channels=3, filter_num=[8, 16, 32, 64, 128, 256, 512], 
                 fully_bayesian=True, prior_config=None):
        super().__init__()
        self.filter_num = filter_num
        
        if prior_config is None:
            prior_config = create_prior_dict(prior_sigma=0.05, prior_pi=1.0)  # Tighter prior for full Bayesian
        
        # Entry block - can be Bayesian or not
        if fully_bayesian:
            self.entry = nn.Sequential(
                BayesianConv2d(input_channels, 8, kernel_size=3, stride=2, padding=1, 
                              bias=False, **prior_config),
                nn.BatchNorm2d(8, affine=True),
                nn.ReLU()
            )
        else:
            self.entry = nn.Sequential(
                nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(8, affine=True),
                nn.ReLU()
            )
        
        # Bayesian residual blocks
        blocks = []
        input_channels = 8
        for out_filters in self.filter_num:
            blocks.append(XceptionResBlock_Bayesian(
                input_channels, out_filters, 
                fully_bayesian=fully_bayesian,
                bayesian_shortcut=fully_bayesian,
                prior_config=prior_config
            ))
            input_channels = out_filters
        self.blocks = nn.ModuleList(blocks)

        # Final layers
        self.final_sepconv = nn.Sequential(
            SeparableConv2d_Bayesian(self.filter_num[-1], self.filter_num[-1]*2,
                                   fully_bayesian=fully_bayesian, prior_config=prior_config),
            nn.BatchNorm2d(self.filter_num[-1]*2, affine=True),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bayesian classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            BayesianLinear(self.filter_num[-1]*2, 2, bias=False, **prior_config)
        )

    def forward(self, x: torch.Tensor):
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_sepconv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


##########################################################################################
#####                                                                                #####
#####                            TRAINING UTILITIES                                  #####
#####                                                                                #####
##########################################################################################

class BayesianTrainer:
    """Helper class for training Bayesian models with proper ELBO weighting"""
    
    def __init__(self, model, num_batches, kl_weight_start=0.01, kl_weight_end=1.0, warmup_epochs=10):
        self.model = model
        self.num_batches = num_batches
        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end
        self.warmup_epochs = warmup_epochs
        
    def get_kl_weight(self, epoch):
        """Gradually increase KL weight during warmup"""
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            kl_weight = self.kl_weight_start + alpha * (self.kl_weight_end - self.kl_weight_start)
        else:
            kl_weight = self.kl_weight_end
        
        return kl_weight / self.num_batches
    
    def elbo_loss(self, outputs, targets, epoch):
        """Compute ELBO loss with proper weighting"""
        likelihood_loss = F.cross_entropy(outputs, targets)
        kl_loss = self.model.nn_kl_divergence() 
        kl_weight = self.get_kl_weight(epoch)
        
        return likelihood_loss + kl_weight * kl_loss, likelihood_loss, kl_loss


# --- Example Usage ---
if __name__ == "__main__":
    # Create models
    frequentist_model = XceptionCustom()
    
    # Start with hybrid model (recommended)
    prior_config = create_prior_dict(prior_sigma=0.1, prior_pi=1.0, posterior_rho_init=-3.0)
    hybrid_model = XceptionHybrid(prior_config=prior_config)
    
    # MC Dropout baseline
    mc_dropout_model = XceptionMCDropout(dropout_p=0.1)
    
    # Full Bayesian (use after hybrid works)
    # full_bayesian_model = XceptionFullBayesian(fully_bayesian=True, prior_config=prior_config)
    
    print("Models created successfully!")
    print(f"Hybrid model parameters: {sum(p.numel() for p in hybrid_model.parameters())}")