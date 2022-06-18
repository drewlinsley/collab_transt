import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Function


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding_mode='zeros', batchnorm=True, use_attention=True, timesteps=64):  # Timesteps is funky here... but go ahead and try this until you figure out the exact training length
        " Referenced from https://github.com/happyjin/ConvGRU-pytorch"
        super(ConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        hidden_size = hidden_dim
        self.batchnorm = batchnorm
        self.timesteps = timesteps
        self.use_attention = use_attention

        if self.use_attention:
            self.a_wu_gate = nn.Conv2d(hidden_size + input_dim, hidden_size, 1, padding=1 // 2)
            init.orthogonal_(self.a_wu_gate.weight)
            init.constant_(self.a_wu_gate.bias, 1.)
        self.i_w_gate = nn.Conv2d(hidden_size + input_dim, hidden_size, 1)
        self.e_w_gate = nn.Conv2d(hidden_size * 2, hidden_size, 1)
        self.inh_init = nn.Conv2d(1, hidden_size, 1, padding=1 // 2)
        self.exc_init = nn.Conv2d(1, hidden_size, 1, padding=1 // 2)

        spatial_h_size = kernel_size
        self.h_padding = spatial_h_size // 2
        self.w_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        self.w_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList([nn.GroupNorm(1, hidden_size, eps=1e-03, affine=True) for i in range(2)])

        init.orthogonal_(self.w_inh)
        init.orthogonal_(self.w_exc)

        init.orthogonal_(self.i_w_gate.weight)
        init.orthogonal_(self.e_w_gate.weight)

        for bn in self.bn:
            init.constant_(bn.weight, 0.1)

        init.uniform_(self.alpha, a=0., b=0.1)
        init.uniform_(self.mu, a=0., b=0.1)
        init.uniform_(self.gamma, a=0., b=0.1)
        init.uniform_(self.kappa, a=0., b=0.1)

        # Init gate biases
        init.uniform_(self.i_w_gate.bias.data, 1, self.timesteps - 1)
        self.i_w_gate.bias.data.log()
        self.e_w_gate.bias.data = -self.i_w_gate.bias.data

    def forward(self, input, excitation, inhibition, label=None, activ=F.softplus, testmode=False, reset_hidden=False):
        "Run the dales law circuit."""

        if inhibition is None:  #  or reset_hidden:
            inhibition = activ(self.inh_init(label))
        if excitation is None:  #  or reset_hidden:
            excitation = activ(self.exc_init(label))

        if self.use_attention:
            input_state_cur = torch.cat([input, excitation], dim=1)
            att_gate = self.a_wu_gate(input_state_cur)  # Attention Spotlight -- MOST RECENT WORKING
            att_gate = torch.sigmoid(att_gate)

        # Gate E/I with attention immediately
        if self.use_attention:
            gated_input = input  # * att_gate  # In activ range
            gated_excitation = att_gate * excitation
            gated_inhibition = inhibition  # att_gate * inhibition
        else:
            gated_input = input

        # Compute inhibition
        inh_intx = activ(F.conv2d(self.bn[0](gated_excitation), self.w_inh, padding=self.h_padding))  # in activ range
        inhibition_hat = activ(input - inh_intx * (self.alpha * gated_inhibition + self.mu))

        # Integrate inhibition
        inh_gate = torch.sigmoid(self.i_w_gate(torch.cat([gated_input, gated_inhibition], dim=1)))
        inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range

        # Pass to excitatory neurons
        exc_gate = torch.sigmoid(self.e_w_gate(torch.cat([gated_excitation, inhibition * att_gate], dim=1)))  # used to be gated_inhibition
        exc_intx = activ(F.conv2d(self.bn[1](inhibition), self.w_exc, padding=self.h_padding))  # In activ range
        excitation_hat = activ(exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim
        excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat
        if testmode:
            return excitation, inhibition, att_gate
        else:
            return excitation, inhibition
