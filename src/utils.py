import time
import os
import inspect

from functools import partial
import torch
import polyscope as ps
import datetime
import polyscope.imgui as psim

# torch.set_default_tensor_type(torch.cuda.DoubleTensor)

class Timer(object):
    def __init__(self, name=None, filename=None):
        self.name = name 
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.3f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename, 'a') as file:
                print(str(datetime.datetime.now()) + ": ", message, file=file)

# Extends dict{} to allow access via dot member like d.elem instead of d['elem']
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


# === Polyscope

def combo_string_picker(name, curr_val, vals_list):

    changed = psim.BeginCombo(name, curr_val)
    clicked = False
    if changed:
        for val in vals_list:
            _, selected = psim.Selectable(val, curr_val==val)
            if selected:
                curr_val=val
                clicked = True
        psim.EndCombo()

    return clicked, curr_val

# === JAX helpers

# quick printing
def printarr(*arrs, data=True, short=True, max_width=200):

    # helper for below
    def compress_str(s):
        return s.replace('\n', '')
    name_align = ">" if short else "<"

    # get the name of the tensor as a string
    frame = inspect.currentframe().f_back
    try:
        # first compute some length stats
        name_len = -1
        dtype_len = -1
        shape_len = -1
        default_name = "[unnamed]"
        for a in arrs:
            name = default_name
            for k,v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            name_len = max(name_len, len(name))
            dtype_len = max(dtype_len, len(str(a.dtype)))
            shape_len = max(shape_len, len(str(a.shape)))
        len_left = max_width - name_len - dtype_len - shape_len - 5

        # now print the acual arrays
        for a in arrs:
            name = default_name
            for k,v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            print(f"{name:{name_align}{name_len}} {str(a.dtype):<{dtype_len}} {str(a.shape):>{shape_len}}", end='') 
            if data:
                # print the contents of the array
                print(": ", end='')
                flat_str = compress_str(str(a))
                if len(flat_str) < len_left:
                    # short arrays are easy to print
                    print(flat_str)
                else:
                    # long arrays
                    if short:
                        # print a shortented version that fits on one line
                        if len(flat_str) > len_left - 4:
                            flat_str = flat_str[:(len_left-4)] + " ..."
                        print(flat_str)
                    else:
                        # print the full array on a new line
                        print("")
                        print(a)
            else:
                print("") # newline
    finally:
        del frame



def logical_and_all(vals):
    out = vals[0]
    for i in range(1,len(vals)):
        out = torch.logical_and(out, vals[i])
    return out

def logical_or_all(vals):
    out = vals[0]
    for i in range(1,len(vals)):
        out = torch.logical_or(out, vals[i])
    return out

def minimum_all(vals):
    '''
    Take elementwise minimum of a list of arrays
    '''
    combined = torch.stack(vals, dim=0)
    return torch.min(combined, dim=0)

def maximum_all(vals):
    '''
    Take elementwise maximum of a list of arrays
    '''
    combined = torch.stack(vals, dim=0)
    return torch.max(combined, dim=0)

def all_same_sign(vals):
    '''
    Test if all values in an array have (strictly) the same sign
    '''
    return torch.logical_or(torch.all(vals < 0), torch.all(vals > 0))

# Given a 1d array mask, enumerate the nonero entries 
# example:
# in:  [0 1 1 0 1 0]
# out: [X 0 1 X 2 X]
# where X = fill_val
# if fill_val is None, the array lenght + 1 is used
def enumerate_mask(mask, fill_value=None):
    if fill_value is None:
        fill_value = mask.shape[-1]+1
    out = torch.cumsum(mask, dim=-1)-1
    out = torch.where(mask, out, fill_value)
    return out


# Returns the first index past the last True value in a mask
def empty_start_ind(mask):
    return torch.max(torch.arange(mask.shape[-1]) * mask)+1
    
# Given a list of arrays all of the same shape, interleaves
# them along the first dimension and returns an array such that
# out.shape[0] = len(arrs) * arrs[0].shape[0]
def interleave_arrays(arrs):
    s = list(arrs[0].shape)
    s[0] *= len(arrs)
    return torch.stack(arrs, dim=1).reshape(s)

def resize_array_axis(A, new_size, axis=0):
    first_N = min(new_size, A.shape[0])
    shape = list(A.shape)
    shape[axis] = new_size
    new_A = torch.zeros(shape, dtype=A.dtype)
    # new_A = new_A.at[:first_N,...].set(A.at[:first_N,...].get())
    new_A[:first_N, ...] = A[:first_N, ...]
    return new_A

def smoothstep(x):
    out = 3.*x*x - 2.*x*x*x
    out = torch.where(x < 0, 0., out)
    out = torch.where(x > 1, 1., out)
    return out

def binary_cross_entropy_loss(logit_in, target):
    # same as the pytorch impl, allegedly numerically stable
    neg_abs = -torch.abs(logit_in)
    loss = torch.clip(logit_in, min=0) - logit_in * target + torch.log(1 + torch.exp(neg_abs))
    return loss

# interval routines
def smallest_magnitude(interval_lower, interval_upper):
    min_mag = torch.maximum(torch.abs(interval_lower), torch.abs(interval_upper))
    min_mag = torch.where(torch.logical_and(interval_upper > 0, interval_lower < 0), 0., min_mag)
    return min_mag
    
def biggest_magnitude(interval_lower, interval_upper):
    return torch.maximum(interval_upper, -interval_lower)


def sin_bound(lower, upper):
    '''
    Bound sin([lower,upper])
    '''
    f_lower = torch.sin(lower)
    f_upper = torch.sin(upper)

    # test if there is an interior peak in the range
    lower /= 2. * torch.pi
    upper /= 2. * torch.pi
    contains_min = torch.ceil(lower - .75) < (upper - .75)
    contains_max = torch.ceil(lower - .25) < (upper - .25)

    # result is either at enpoints or maybe an interior peak
    out_lower = torch.minimum(f_lower, f_upper)
    out_lower = torch.where(contains_min, -1., out_lower)
    out_upper = torch.maximum(f_lower, f_upper)
    out_upper = torch.where(contains_max, 1., out_upper)

    return out_lower, out_upper

def cos_bound(lower, upper):
    return sin_bound(lower + torch.pi/2, upper + torch.pi/2)
