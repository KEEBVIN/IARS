from torch import nn

import convolution
import augments

# input_dims: dimension of input time series
# output_dims: dimension of output time series
# hidden_dims: parameter of the model
# depth: number of layers for the model
# input: B x T x C
# output: B x T x Cout


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims #input dimension of the model  (dimension of a time series)
        self.output_dims = output_dims #output dimension of the model (per time stamp, how many dimensions)
        self.hidden_dims = hidden_dims #hidden dimension in the model (per time stamp, how many dimensions)
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = convolution.DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = augments.generate_binomial_mask(x.size(0), x.size(1)).to(x.device)

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x