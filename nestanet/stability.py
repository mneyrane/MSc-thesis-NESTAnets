import math
import torch

def adv_perturbation(x, opB, opR, c_B, eta, lr, num_iters, device):
    """
    Computes an adversarial perturbation e = [e1;e2] (assuming e1 = e2) for 
    a (subdifferentiable) reconstruction method for recovering a vector x given
    measurements y1 = B @ x + e1, y2 = B @ x + e2. Expressed in stacking
    form, with y = [y1;y2] and A = [B;B], then y = A @ x + e.

    Here B is assumed to be a linear operator where B @ B^* = c*I for some 
    constant c > 0. This means the block matrix B has full row rank and its 
    Moore-Penrose pseudoinverse is B^*/c.

    Inspired by [1] and [2], this function implements projected gradient 
    ascent to solve the nonconvex optimization problem

        max ||R(y + e) - R(y)||_2   s.t.   ||e1||_2 <= eta, e1 = e2.

    Due to the nonconvexity, gradient ascent should be run several times with
    different randomly initialized perturbations e. The e that yielding
    largest objective value is selected and is returned.

    NOTE: To avoid confusion, if the solver R uses a noise level of NOISE,
    so that it expects ||e||_2 <= NOISE, then if we want the perturbation
    noise level to be X times NOISE, then eta = X * NOISE / sqrt(2).

    Args:
        x (torch.Tensor) : ground truth vector
        opB (function) : measurement operator B
        opR (function) : reconstruction map
        c_B (float) : constant c for which B @ B^* = c*I
        eta (float) : constraint for e1 (not e!)
        lr (float) : learning rate for gradient ascent
        num_iters (int) : number of iterations of gradient ascent
        device : CPU or GPU as returned by torch.device

    Returns:
        best_e (torch.Tensor) : worst-case noise perturbation

    References:
        [1] Ch. 19.4. "Compressive Imaging: Structure, Sampling, Learning"
            Adcock, et al. ISBN:9781108421614.
        [2] Sec. 3.4. "Solving Inverse Problems With Deep Neural Networks --
            Robustness Included?" Genzel, et al. arXiv:2011.04268.
    """
    y = opB(x,1)
    x_rec = opR(y)
    
    N = x.shape[0]
    m = y.shape[0]

    # squared-norm function for complex tensors
    sq_norm = lambda z : torch.vdot(z,z)
    obj_fn = lambda e : -0.5*sq_norm(opR(y+e)-x_rec)
    
    best_obj_val = -float('Inf')
    obj_val = None
    best_e = None
    
    noise = torch.randn(2*m, dtype=torch.float64, device=device)

    noise = (eta/math.sqrt(m))*noise/torch.linalg.norm(noise,ord=2)

    e = noise[:m] + 1j*noise[m:]
    e.requires_grad_()

    optimizer = torch.optim.SGD([e], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-5)
    
    for i in range(num_iters):
        # descent step
        optimizer.zero_grad()
        obj = obj_fn(e)
        obj.backward()
        optimizer.step()

        with torch.no_grad():
            # projection
            e_len = torch.linalg.norm(e,ord=2)
            if e_len > eta:
                e.multiply_(eta/e_len)
            
            obj_val = -torch.real(obj_fn(e))
            scheduler.step(obj_val)

            obj_val = obj_val.cpu()

            print(
                'Step %-3d -- norm(e): %.3e -- obj val: %.5e -- lr: %.2e' %
                (
                    i+1, 
                    min(eta, float(e_len)), 
                    float(obj_val), 
                    float(optimizer.param_groups[0]['lr']),
                )
            )

            if obj_val > best_obj_val:
                best_obj_val = obj_val
                best_e = e.detach().clone()

    return best_e
