import torch
from darts.models import RNNModel
from darts.models.forecasting.rnn_model import _RNNModule
from darts.utils.likelihood_models import GaussianLikelihood


class ResRNNModule(_RNNModule):
    def forward(self, x_in, h=None):
        predictions, last_hidden_state = super().forward(x_in, h)
        assert predictions.shape[2] == 1, "multivariate model not supported"

        x, _ = x_in

        if self.likelihood:
            assert isinstance(self.likelihood, GaussianLikelihood)
            mu = x[..., 0:1] + torch.nn.functional.softplus(predictions[:, :, 0:1, 0])
            sigma = predictions[:, :, 0:1, 1]
            predictions = torch.stack([mu, sigma], dim=-1)

        else:
            predictions = x[..., 0:1, None] + torch.nn.functional.softplus(predictions)

        return predictions, last_hidden_state


class ResRNNModel(RNNModel):
    def _create_model(self, train_sample):
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        model = ResRNNModule(
            name=self.rnn_type_or_module,
            input_size=input_dim,
            target_size=output_dim,
            nr_params=nr_params,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.n_rnn_layers,
            **self.pl_module_params,
        )
        return model
