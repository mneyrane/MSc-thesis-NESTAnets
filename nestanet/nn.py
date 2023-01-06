import nestanet.operators as _op
import torch
from torch.nn.functional import relu as _relu
from torch.linalg import norm as _norm

def nesta_stacked(y1, y2, z0, opB, opW, c_B, L_W, num_iters, eta, mu, eval_fns=None):

    z = z0
    q_v = z0
    
    if eval_fns is not None:
        ev_values = {key : [] for key in eval_fns}
    else:
        ev_values = None

    # these quantities only need to be calculated once
    # ------------------------------------------------
    y_sum = y1+y2
    d_noise = 2*eta*eta-torch.real(torch.vdot(y1-y2,y1-y2))
    # ------------------------------------------------

    for n in range(num_iters):
        # -----------
        # compute x_n
        # -----------
        grad = opW(z,1)
        grad = _op.huber_fn_gradient(grad, mu)
        grad = mu/(L_W*L_W)*opW(grad,0)
        q = z-grad

        dy = y_sum-2*opB(q,1)
        lam = _relu(0.5*(1-torch.sqrt(d_noise/torch.real(torch.vdot(dy,dy)))))

        x = (lam/c_B)*opB(dy,0) + q

        if eval_fns is not None:
            for key in eval_fns:
                ev_values[key].append(eval_fns[key](x))

        # -----------
        # compute v_n
        # -----------
        alpha = (n+1)/2
        q_v = q_v-alpha*grad
        q = q_v

        dy = y_sum-2*opB(q,1)
        lam = _relu(0.5*(1-torch.sqrt(d_noise/torch.real(torch.vdot(dy,dy)))))

        v = (lam/c_B)*opB(dy,0) + q

        # ---------------
        # compute z_{n+1}
        # ---------------
        tau = 2/(n+3)
        z = tau*v+(1-tau)*x
        
    return x, ev_values

def restarted_nesta_stacked(y1, y2, z0, opB, opW, c_B, L_W, in_iters, re_iters, eta, mu_seq, eval_fns=None):

    z = z0

    if eval_fns is not None:
        re_ev_values = {key : [] for key in eval_fns}
    else:
        re_ev_values = None

    assert len(mu_seq) == re_iters

    for k in range(re_iters):
        mu = mu_seq[k]
        z, inner_ev_values = nesta_stacked(y1, y2, z, opB, opW, c_B, L_W, in_iters, eta, mu, eval_fns)

        if eval_fns is not None:
            for key in inner_ev_values:
                re_ev_values[key].append(inner_ev_values[key])

    return z, re_ev_values
