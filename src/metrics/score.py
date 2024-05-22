import torch
from torch import Tensor
from typing import Iterable
from itertools import chain
from .benchmark import Benchmark
from .metric import Metric
from core import (
    FunctionClass,
    ContextModel,
)
from .function_error import FunctionClassError

class RegressionScore(Benchmark):
    # zero_err is of shape (1)
    # base_err is of shape (seq_length,)
    def __init__(self, metric: Metric, function_class: FunctionClass, zero_err, baseline: ContextModel):
        self.funct_err = FunctionClassError(metric, function_class)
        self.zero_err = zero_err
        self.baseline = baseline

    def evaluate(self, models: Iterable[ContextModel], num_batches: int = 1) -> Iterable[Tensor]:
        # generating model errors
        model_err = torch.tensor(
            self.funct_err.evaluate(
                chain([self.baseline], models), 
                num_batches=num_batches
            )
        )
        model_err = model_err.squeeze(-1) # remove sq_err

        norm_model_err = torch.sub(model_err[1:], self.zero_err)
        norm_base_err = torch.sub(model_err[0], self.zero_err)

        scores = torch.div(torch.mean(norm_model_err, dim=-1), torch.mean(norm_base_err, dim=-1))
        scores_mean = torch.mean(scores, dim=1)
        scores_std = torch.std(scores, dim=1) / (scores.shape[1] ** 0.5)

        return scores_mean, scores_std
