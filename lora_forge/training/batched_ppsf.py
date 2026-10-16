"""A batched step() for prodigyplus.ProdigyPlusScheduleFree.

ProdigyPlusScheduleFree updates one tensor at a time with ~35 small GPU operations each; a LoRA has
hundreds of tensors (an Anima rank-32 LoRA has 900), so its step is bound by Python and kernel-launch
overhead, not by the GPU. This step stacks the tensors that share a parameter group and a shape and
updates each stack with the same math in one go. The optimizer's state keeps its per-tensor layout, so
saved states resume either way, and Prodigy's d bookkeeping is the optimizer's own (on_end_step).
Verified against the original on real Anima gradients: d and the weights agree to ~1e-7 per step.
Only stochastic rounding (bf16 weights) draws different random numbers.

The update math follows prodigy-plus-schedule-free 2.0 by Logan Booker (Apache-2.0,
https://github.com/LoganBooker/prodigy-plus-schedule-free). Options outside the supported set
(UNSUPPORTED below) and other versions keep the original step.

Kept free of lora_forge imports: run_entry.py loads it in the training process.
"""

import torch

CHECKED_VERSION = (2, 0, 0)
# group options this step does not implement: the optimizer's own step runs instead
UNSUPPORTED = ("use_bias_correction", "use_adopt", "use_cautious", "use_grams", "use_orthograd", "use_focus",
               "use_speed", "fused_back_pass")


def unsupported_options(optimizer) -> list:
    found = []
    for group in optimizer.param_groups:
        if not group.get("use_schedulefree", True):
            found.append("use_schedulefree=False")
        found += [key for key in UNSUPPORTED if group.get(key)]
    return sorted(set(found))


def stochastic_round(source: torch.Tensor) -> torch.Tensor:
    """The optimizer's copy_stochastic_ for a whole fp32 tensor: fp32 values whose low 16 bits are randomly
    rounded away, so the copy to bf16 is exact."""
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=1 << 16)
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    return result.view(dtype=torch.float32)


def write_back(targets: list, source: torch.Tensor, stochastic: bool):
    if stochastic and targets[0].dtype == torch.bfloat16:
        source = stochastic_round(source)
    torch._foreach_copy_(targets, list(source.unbind(0)))


@torch.no_grad()
def step_stack(optimizer, group, params: list, states: list, factored_dims, prodigy_active: bool, running):
    """step_param_schedulefree for tensors of one shape and group, stacked along a new first dimension."""
    beta1, beta2, beta3 = optimizer.get_betas(group)
    d, eps = group["d"], group["eps"]
    dlr = optimizer.get_dlr(group)
    stochastic = group["stochastic_rounding"]
    shape = params[0].shape

    grads = torch.stack([p.grad for p in params])
    # Prodigy's statistics use the raw gradient; G below may be the same tensor and changes in place
    sliced_grad = grads.flatten(1)[:, ::11].clone() if prodigy_active else None
    G = grads.float()
    z_stored = torch.stack([s["z"] for s in states])
    Z = z_stored.float()
    Y = torch.stack(params).float()

    # second moment (update_second_moment, get_denom)
    if factored_dims is not None:
        dr, dc = factored_dims
        R = torch.stack([s["exp_avg_sq_row"] for s in states])
        C = torch.stack([s["exp_avg_sq_col"] for s in states])
        R.mul_(beta2).add_(G.norm(dim=dr + 1, keepdim=True).square_().mul_(1 / shape[dr]), alpha=(1 - beta2) * d ** 2)
        C.mul_(beta2).add_(G.norm(dim=dc + 1, keepdim=True).square_().mul_(1 / shape[dc]), alpha=(1 - beta2) * d ** 2)
        reduce_dc = dc - 1 if dc > dr else dc
        row_col_mean = R.mean(dim=reduce_dc + 1, keepdim=True).add_(1e-30)
        denom = (R.div(row_col_mean) * C).sqrt_()
    else:
        E = torch.stack([s["exp_avg_sq"] for s in states])
        E.mul_(beta2).addcmul_(G, G, value=(1 - beta2) * d ** 2)
        denom = E.sqrt()

    G.mul_(d)
    if eps is None:  # Adam-atan2
        update = G.atan2_(denom).mul_(1.2732395)
    else:
        update = G.div_(denom.add_(d * eps))
    del denom
    if group["use_stableadamw"]:
        rms = update.flatten(1).norm(dim=1).div(update[0].numel() ** 0.5).clamp_min(1.0)
        update.mul_(rms.rsqrt().view((-1,) + (1,) * len(shape)))

    # update_prodigy, on z as stored before this step
    if prodigy_active:
        d_update = d ** 2 / group["d0"] ** 0.5
        sliced_z = z_stored.flatten(1)[:, ::11]
        if states[0]["p0"].dim() == 0:  # the initial weights were all zero
            x0_minus = states[0]["p0"] - sliced_z
        else:
            x0_minus = torch.stack([s["p0"] for s in states]) - sliced_z
        dot_dtype = torch.promote_types(sliced_grad.dtype, x0_minus.dtype)
        x0_dot = (sliced_grad.float() * x0_minus.float()).sum(dim=1).to(dot_dtype)
        S = torch.stack([s["s"] for s in states])
        S.mul_(beta3).add_(sliced_grad, alpha=d_update)
        numerator, denominator = running
        denominator.add_(S.abs().sum(dim=1).float().sum())
        numerator.add_(x0_dot.float().sum(), alpha=d_update)
    else:
        # past prodigy_steps the optimizer frees Prodigy's state
        for state in states:
            state.pop("s", None)
            state.pop("p0", None)

    # update_params
    decay = optimizer.get_weight_decay(group)
    weight = dlr ** 2
    weight_sum = group["running_weight_sum"] = group.get("weight_sum", 0) + weight
    ckp1 = weight / weight_sum if weight_sum else 0
    schedulefree_c = group.get("schedulefree_c", 0)
    if schedulefree_c > 0:
        ckp1 = min(1.0, ckp1 * (1 - beta1) * schedulefree_c)
    xy_step = 1 - beta1 * (1 - ckp1)
    group["effective_lr"] = group["lr"] * xy_step
    y_wd = None
    if decay != 0:
        if group["weight_decay_by_lr"]:
            update.add_(Y, alpha=decay)
        else:
            y_wd = Y.clone()
    Y.lerp_(end=Z, weight=ckp1)
    Z.sub_(update, alpha=dlr)
    Y.sub_(update, alpha=dlr * xy_step)
    if y_wd is not None:
        Z.sub_(y_wd, alpha=decay)
        Y.sub_(y_wd, alpha=decay * xy_step)

    write_back(params, Y, stochastic)
    write_back([s["z"] for s in states], Z, stochastic)
    if factored_dims is not None:
        torch._foreach_copy_([s["exp_avg_sq_row"] for s in states], list(R.unbind(0)))
        torch._foreach_copy_([s["exp_avg_sq_col"] for s in states], list(C.unbind(0)))
    else:
        torch._foreach_copy_([s["exp_avg_sq"] for s in states], list(E.unbind(0)))
    if prodigy_active:
        torch._foreach_copy_([s["s"] for s in states], list(S.unbind(0)))


@torch.no_grad()
def batched_step(optimizer, closure=None):
    optimizer.try_unhook_kohya_fbp()
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    optimizer.on_start_step()
    for group in optimizer.param_groups:
        if not group["train_mode"]:
            raise Exception("Not in train mode!")
        prodigy_steps = group["prodigy_steps"]
        prodigy_active = prodigy_steps <= 0 or group["k"] < prodigy_steps
        stacks = {}
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if not state:
                state = optimizer.initialise_state(p, group)
            factored_dims = state.get("exp_avg_sq_metadata")
            zero_p0 = prodigy_active and state["p0"].dim() == 0
            key = (p.shape, p.dtype, p.grad.dtype, state["z"].dtype, factored_dims, zero_p0)
            params, states = stacks.setdefault(key, ([], []))
            params.append(p)
            states.append(state)
        running = optimizer.get_running_values_for_group(group)
        for key, (params, states) in stacks.items():
            step_stack(optimizer, group, params, states, key[4], prodigy_active, running)
    # the optimizer updates d once its last parameter is done
    optimizer.parameters_to_process = 1
    optimizer.on_end_step()
    return loss


def install(optimizer_class, core_class) -> bool:
    """Replaces optimizer_class.step; returns False (nothing changed) for an unchecked version."""
    if getattr(core_class, "VERSION", None) != CHECKED_VERSION:
        return False
    original = optimizer_class.step
    reported = []

    def step(self, closure=None):
        unsupported = unsupported_options(self)
        if not reported:
            reported.append(True)
            if unsupported:
                print(f"LoRA-Forge: ProdigyPlusScheduleFree uses its own step ({', '.join(unsupported)} "
                      "is not batched)", flush=True)
            else:
                print("LoRA-Forge: ProdigyPlusScheduleFree updates same-shaped tensors together (batched step)", flush=True)
        if unsupported:
            return original(self, closure)
        return batched_step(self, closure)

    optimizer_class.step = step
    return True
