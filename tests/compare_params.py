from typing import Tuple, List

from torch.nn import Module


def capture_params(model: Module) -> List[Tuple[str, List[float]]]:
    """
    Take a copy of the model parameters at the time of invocation.
    :param model: Pytorch module.
    :return: List of named parameters with their value snapshot (list of floats).
    """
    return [(n, p.data.tolist()) for n, p in model.named_parameters()]


def compare_parameters(before_params: List[Tuple[str, List[float]]],
                       after_params: List[Tuple[str, List[float]]]):
    """
    Check that all model parameters have changed. An assertion will be thrown if any
    paramater remains unchanged after the optimization step.
    :param before_params: result of capture_params used before optimization step
    :param after_params: result of capture_params used after optimization step
    :return: None
    """
    before_param_values = {}
    after_param_values = {}

    for name, p in before_params:
        before_param_values[name] = p
    for name, p in after_params:
        after_param_values[name] = p

    assert list(before_param_values.keys()) == list(after_param_values.keys())
    param_names_did_not_change = []
    param_names_changed = []
    for name in before_param_values.keys():
        after_values = after_param_values[name]
        before_values = before_param_values[name]
        if after_values == before_values:
            param_names_did_not_change.append(name)
        else:
            param_names_changed.append(name)
    assert len(
        param_names_did_not_change) == 0, f"some parameters did not change and shoud have: {param_names_did_not_change}," \
                                          f" these parameters did change: {param_names_changed}"
