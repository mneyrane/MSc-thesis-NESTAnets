import math
import numpy as np
import torch
import torch.fft as _fft

def huber_fn_gradient(x, mu):
    """ Huber function gradient """
    y = torch.zeros_like(x)
    
    with torch.no_grad():
        mask = torch.abs(x) <= mu
    
    y[mask] = x[mask]/mu
    y[~mask] = x[~mask] / torch.abs(x[~mask])
    return y

def discrete_gradient_2d(x,mode,N1,N2):
    """2-D anisotropic discrete gradient (TV) operator 

    Periodic boundary conditions are used. The parameters N1 and N2
    are the height and width of the image x.

    For the forward operator, x is expected to a vectorized matrix X
    in row-major order (C indexing). The output is a stacking of 
    vectorized vertical and horizontal gradients.

    From the definition of the adjoint, if you have two N1xN2 matrices 
    X1, X2, then the adjoint computes the vectorization of

        grad_vert_adjoint(X1) + grad_hori_adjoint(X2).

    Thus, the input x for the adjoint should be a stacking of vectorizations
    of X1, X2, in row-major order. That is:

        x = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=2)
    """
   
    assert len(x.shape) == 1, 'Input x is not a vector'

    if mode == True:
        
        x = x.reshape((N1,N2))
        
        y1, y2 = torch.zeros_like(x), torch.zeros_like(x)
        
        # vertical (axis) gradient
        y1 = torch.roll(x,-1,0) - x

        # horizontal (axis) gradient
        y2 = torch.roll(x,-1,1) - x

        y = torch.stack([y1,y2], dim=2)
        
        return y.reshape(-1)

    else:
       
        x = x.reshape((N1,N2,2))
        
        # vertical (axis) gradient transpose
        y = torch.roll(x[:,:,0],1,0) - x[:,:,0] 
        
        # horizontal (axis) gradient transpose
        y = y + torch.roll(x[:,:,1],1,1) - x[:,:,1]
        
        return y.reshape(-1)

def fourier_2d(x, mode, N, mask, device):
    """ 2-D discrete Fourier transform 

    Normalized so that the forward and inverse mappings are unitary.

    The parameter 'mask' is assumed to be a NxN boolean matrix, used as a 
    sampling mask for Fourier frequencies.
    """

    assert len(x.shape) == 1, 'Input x is not a vector'
    
    if mode == True:
        z = _fft.fftshift(_fft.fft2(x.reshape((N,N)), norm='ortho'))
        y = z[mask]
        return y.reshape(-1)
    
    else: # adjoint
        z = torch.zeros(N*N,dtype=x.dtype)
        z = z.to(device)
        
        mask = mask.reshape(-1)

        z[mask] = x
        z = z.reshape((N,N))
        y = _fft.ifft2(_fft.fftshift(z), norm='ortho')
        return y.reshape(-1)
