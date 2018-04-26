def ensure_global_grads(local_model, global_model, global_device):
    for local_param, global_param in zip(local_model.parameters(),
                                         global_model.parameters()):
        if global_param.grad is not None:
            return
        global_param._grad = local_param.grad.to(global_device)
