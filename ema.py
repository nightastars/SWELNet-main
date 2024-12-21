import torch
import torch.distributed as dist

# def update_ema_variables(model, ema_model, alpha, global_step):
#     # Use the true average until the exponential average is more correct
#     alpha = min(1 - 1 / (global_step + 1), alpha)
#     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#         ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
#         # ema_param.data.mul_(alpha).add_(param.data, 1 - alpha)


def update_ema_variables(model, ema_model, alpha, global_step):
    world_size = get_world_size()
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    iCount = 0
    for ema_param, param in zip(ema_model.parameters(recurse=True), model.parameters(recurse=True)):
        if world_size > 1:
            update_param = param.data.clone()
            torch.distributed.all_reduce(update_param, op=torch.distributed.ReduceOp.SUM)
            update_param /= world_size
        else:
            update_param = param.data
        ema_param.data.mul_(alpha).add_(1 - alpha, update_param)


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
