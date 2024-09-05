def take_step(substep_ind, in_tup):
    root, dir, t, step_size, is_hit, hit_id, step_count = in_tup
    # print(root.shape, dir.shape, t.shape, step_size.shape, is_hit.shape, hit_id.shape, step_count.shape)
    can_step = not is_hit  # this ensure that if we hit on a previous substep, we don't keep stepping
    step_count += not is_hit  # count another step for this ray (unless concvered)
    # loop over all function in the list
    func_id = 1
    for func, params in zip(funcs_tuple, params_tuple):
        pos_start = root + t * dir
        half_vec = 0.5 * step_size * dir
        pos_mid = pos_start + half_vec
        print("pos mid and half vec shapes: ", pos_mid.shape, half_vec.shape)
        if isinstance(func, CrownImplicitFunction):
            box_type = func.classify_box(params, pos_mid, half_vec)
        else:
            box_type = func.classify_general_box(params, pos_mid, half_vec[None, :])

        # test if the step is safe
        can_step = torch.logical_and(
            torch.tensor(can_step),
            torch.logical_or(box_type == SIGN_POSITIVE, box_type == SIGN_NEGATIVE)
        )

        # For convergence testing, sample the function value at the start the interval and start + eps
        pos_start = root + t * dir
        pos_eps = root + (t + opts['hit_eps']) * dir
        val_start = func(params, pos_start)
        val_eps = func(params, pos_eps)
        # Check if we converged for this func
        this_is_hit = torch.sign(val_start) != torch.sign(val_eps)
        # hit_id = torch.where(this_is_hit, func_id, hit_id)
        # hit_id = torch.full_like(this_is_hit, func_id).where(this_is_hit, torch.full_like(this_is_hit, hit_id))
        hit_id = torch.full_like(this_is_hit, func_id).where(this_is_hit, torch.full_like(this_is_hit, hit_id))
        is_hit = torch.logical_or(torch.tensor(is_hit), this_is_hit)
        func_id += 1

    # take a full step of step_size if it was safe, but even if not we still inch forward
    # (this matches our convergence/progress guarantee)
    this_step_size = torch.where(can_step, step_size, opts['hit_eps'])

    # take the actual step (unless a previous substep hit, in which case we do nothing)
    t = torch.where(is_hit, t, t + this_step_size * opts['safety_factor'])
    # update the step size
    step_size = torch.where(can_step,
                            step_size * opts['interval_grow_fac'],
                            step_size * opts['interval_shrink_fac'])
    step_size = torch.clip(step_size, min=opts['hit_eps'])
    return (root, dir, t, step_size, is_hit, hit_id, step_count)


    def take_several_steps(root, dir, t, step_size):
        # print(root.shape, dir.shape, t.shape, step_size.shape)
        # Perform some substeps
        # N = root.shape[0]
        # is_hit = torch.full((N,), False)
        # hit_id = torch.zeros((N,))
        # step_count = torch.zeros((N,))
        is_hit = False
        hit_id = 0
        step_count = 0
        in_tup = (root, dir, t, step_size, is_hit, hit_id, step_count)


        def fori_loop(lower, upper, body_fun, init_val):
            val = init_val
            for i in range(lower, upper):
                val = body_fun(i, val)
            return val

        out_tup = fori_loop(0, n_substeps, take_step, in_tup)
        _, _, t, step_size, is_hit, hit_id, step_count = out_tup

        step_count = torch.as_tensor(step_count)

        return t, step_size, is_hit, hit_id, step_count


def outward_normal(funcs_tuple, params_tuple, hit_pos, hit_id, eps, method='finite_differences'):
    grad_out = torch.zeros(3)
    i_func = 1
    for func, params in zip(funcs_tuple, params_tuple):
        f = partial(func, params)

        if method == 'autodiff':
            grad_f = functorch.jacfwd(f)
            grad = grad_f(hit_pos)

        elif method == 'finite_differences':
            # 'tetrahedron' central differences approximation
            # see e.g. https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            offsets = torch.tensor((
                (+eps, -eps, -eps),
                (-eps, -eps, +eps),
                (-eps, +eps, -eps),
                (+eps, +eps, +eps),
            ))
            x_pts = hit_pos[None, :] + offsets
            if isinstance(func, CrownImplicitFunction):
                print(hit_pos, hit_id)
                print(x_pts)
                print(torch.cuda.memory_summary())
                samples = func(params, x_pts)

            else:
                samples = vmap(f)(x_pts)
            grad = torch.sum(offsets * samples[:, None], dim=0)

        else:
            raise ValueError("unrecognized method")

        grad = geometry.normalize(grad)
        grad_out = torch.where(hit_id == i_func, grad, grad_out)
        i_func += 1

    return grad_out


def outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, eps, method='finite_differences'):
    this_normal_one = lambda p, id: outward_normal(funcs_tuple, params_tuple, p, id, eps, method=method)
    return vmap(this_normal_one)(hit_pos, hit_ids)



# def outward_normals(funcs_tuple, params_tuple, hit_pos, hit_id, eps, method='finite_differences'):
#     N = hit_pos.shape[0]
#     grad_out = torch.zeros((N, 3))
#     batch_size_per_iteration = 1250
#     for start_idx in range(0, N, batch_size_per_iteration):
#         torch.cuda.empty_cache()
#         end_idx = min(start_idx + batch_size_per_iteration, N)
#         i_func = 1
#         for func, params in zip(funcs_tuple, params_tuple):
#             f = partial(func, params)
#
#             if method == 'autodiff':
#                 grad_f = functorch.jacfwd(f)
#                 grad = grad_f(hit_pos[start_idx:end_idx])
#
#             elif method == 'finite_differences':
#                 # 'tetrahedron' central differences approximation
#                 # see e.g. https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
#                 offsets = torch.tensor((
#                     (+eps, -eps, -eps),
#                     (-eps, -eps, +eps),
#                     (-eps, +eps, -eps),
#                     (+eps, +eps, +eps),
#                     ))
#
#                 x_pts = hit_pos[start_idx:end_idx].unsqueeze(1) + offsets
#                 if isinstance(func, CrownImplicitFunction):
#                     # print(x_pts.shape)
#                     # print(torch.cuda.memory_summary())
#
#                     x_pts = x_pts.view(-1, 1, 3)
#                     print(x_pts.shape)
#
#                     samples = vmap(f)(x_pts)
#                     samples = samples.view(-1, 4, 1)
#                     grad = torch.sum(offsets * samples, dim=1)
#                 else:
#                     x_pts = x_pts.view(-1, 3)
#                     samples = vmap(f)(x_pts)
#                     samples = samples.view(-1, 4, 1)
#                     grad = torch.sum(offsets * samples, dim=1)
#             else:
#                 raise ValueError("unrecognized method")
#
#             grad = torch.nn.functional.normalize(grad)
#             grad_out[start_idx:end_idx] = grad.where(hit_id[start_idx:end_idx].unsqueeze(1).expand(-1, 3) == i_func, grad_out[start_idx:end_idx])
#             i_func += 1
#
#     return grad_out

def cast_rays_cw(funcs_tuple, params_tuple, roots, dirs, c=1e-3, delta=0.001, num_iter=4000, lr=0.5):
    func_id = 1
    # hit_id = torch.zeros(dirs.shape[0], dtype=torch.int)
    hit_id = 0
    t = torch.empty((dirs.shape[0], 1), requires_grad=True)
    optimizer = optim.Adam([t], lr=lr, weight_decay=0.)
    # optimizer = optim.Adam([t], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = optim.SGD([t], lr=lr, weight_decay=c)
    for func, params in zip(funcs_tuple, params_tuple):
        model = func_as_torch(params)
        model.eval()
        mask = torch.ones((dirs.shape[0], 1))
        for i in range(num_iter):
            perts = t * dirs
            inputs = roots + perts
            outputs = model(inputs)
            # log_square_diff = torch.log(outputs.abs()) * mask
            # loss = log_square_diff.sum()
            loss = outputs.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # this_is_hit = (model(roots + (t.detach() + delta) * dirs) < 0.) & (model(roots + (t.detach() - delta) * dirs) > 0.)
            this_is_hit = (model(roots + (t.detach() + delta) * dirs) < 0.) & (model(roots + t.detach() * dirs) > 0.)
            hit_id = torch.where(this_is_hit, func_id, hit_id)
            mask = torch.logical_not(this_is_hit)
            t = t + delta * mask
            # if i % 300 == 0:
                # print(this_is_hit.reshape((1024, 1024))[509:515, 509:515])

        func_id += 1


    t_raycast = t.detach()

    counts, n_eval = torch.zeros_like(t_raycast), 0
    return t_raycast.flatten(), hit_id.detach().flatten(), counts, n_eval