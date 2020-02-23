import torch
from functools import reduce, partial


__all__ = ["multi_apply", "reduce_loss"]


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_loss(losses, reduction="mean"):
    losses_sum = reduce(lambda x, y: x + y, losses)

    if reduction == "sum":
        return losses_sum
    elif reduction == "mean":
        return losses_sum / len(losses)
    else:
        raise NotImplementedError
