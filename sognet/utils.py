import torch


__all__ = ["multi_apply", "split"]


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, list) and len(split_size_or_sections) == 1:
        assert split_size_or_sections[0] == tensor.size(dim)
        return (tensor, )

    return torch.split(tensor, split_size_or_sections, dim=dim)
