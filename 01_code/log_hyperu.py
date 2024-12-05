import torch
import numpy as np
import scipy.special as sp

"""Unfortunately, it appears that not all approximations of the hyperu function as x -> 0 
are implemented in the scipy library. As such, this is an implementation of those 
appearing in the DLMF (https://dlmf.nist.gov/13.2#iii).
a, b, and z are all tensors of the same shape. The function returns a tensor of the same shape.
"""
def robust_hyperu(a, b, z):
    res = torch.zeros_like(z)
    res.fill_(np.nan)
    small_z = torch.abs(z) < 1e-9

    # Approximation for small z and b > 2
    res = torch.where(torch.logical_and(small_z, b > 2), 
                        torch.exp(torch.lgamma(b - 1) - torch.lgamma(a) + (1 - b) * torch.log(z)),
                        res)

    # Approximation for small z and b = 2
    res = torch.where(torch.logical_and(small_z, b == 2),
                        torch.exp(-torch.lgamma(a) - torch.log(z)),
                        res)

    # Approximation for small z and 1 < b < 2
    res = torch.where(torch.logical_and(torch.logical_and(small_z, b < 2), b > 1),
                        torch.exp(torch.lgamma(b - 1) - torch.lgamma(a) + (1 - b) * torch.log(z)) +
                        torch.exp(-torch.lgamma(1 - b) - torch.lgamma(a - b +1)),
                        res)

    # Approximation for small z and b = 1
    res = torch.where(torch.logical_and(small_z, b == 1),
                        -torch.exp(-torch.lgamma(a) - torch.log(z + torch.digamma(a) + 0.57721566490153286060651209008240243)),
                        res)

    # Approximation for small z and 0 < b < 1
    res = torch.where(torch.logical_and(torch.logical_and(small_z, b < 1), b > 0),
                        torch.exp(torch.lgamma(1 - b) - torch.lgamma(a - b + 1)),
                        res)

    # Approximation for small z and b = 0
    res = torch.where(torch.logical_and(small_z, b == 0),
                        torch.exp(-torch.lgamma(a + 1)),
                        res)

    # Approximation for small z and b < 0
    res = torch.where(torch.logical_and(small_z, b < 0),
                        torch.exp(torch.lgamma(1 - b) - torch.lgamma(a - b + 1)),
                        res)

    # Fill up all where no special case applies
    res = torch.where(torch.isnan(res), sp.hyperu(a, b, z), res)
    
    return(res)

"""Then, define the log_hyperu function that can be used in autograd
a, b, and x are all tensors of the same shape. The function returns a tensor of the same shape.
Currently, the function is not differentiable with respect to a and b. I'm not entirely sure this is possible anyways 
"""
class log_hyperu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, x):

        u_res = robust_hyperu(a, b, x)
        result = torch.log(u_res)

        ctx.mark_non_differentiable(a)
        ctx.mark_non_differentiable(b)

        ctx.save_for_backward(a, b, x, u_res)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        
        a, b, x, u_res = ctx.saved_tensors
        grad_x = grad_output * (-a * torch.div(robust_hyperu(a + 1, b + 1, x), u_res))

        return None, None, grad_x
    
# Alias the apply method:
log_hyperu = log_hyperu.apply