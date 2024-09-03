from functools import partial

import torch
import numpy as np
import functorch
# Imports from this project
from utils import *
import affine
import slope_interval
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def split_generator(generator, num_splits=2):
    return [torch.Generator(device=device).manual_seed(generator.initial_seed() + i) for i in range(num_splits)]


# ===== High-level flow

def build_spec(mlp_op_list):
    out_params = {}
    
    # prepend an opcount to the list of operations
    # (entries now look like "0007.dense.A", "0003.relu", etc)
    for i_op, op in enumerate(mlp_op_list):
        for key, val in op.items():
            key = f"{i_op:04d}." + key
            out_params[key] = val

    return out_params

def initialize_params(params, rngkey):

    N_op = n_ops(params)
    out_params = {}

    # perform initialization
    for i_op in range(N_op):
        name, orig_args = get_op_data(params, i_op)
        if name in initialize_func:

            # apply the init function
            subkey, rngkey = split_generator(rngkey)
            init_args = initialize_func[name](rngkey=subkey, **orig_args)

            # replace the updated data in the array
            for a in init_args:
                a_op = f"{i_op:04d}.{name}.{a}"
                out_params[a_op] = init_args[a]
            
        # functions which require no initialization
        # (just copy the params content)
        else:
            for a in orig_args:
                a_op = f"{i_op:04d}.{name}.{a}"
                out_params[a_op] = orig_args[a]


    return out_params

def opt_param_keys(params):

    N_op = n_ops(params)
    keys = []
    
    for i_op in range(N_op):
        name, orig_args = get_op_data(params, i_op)

        if name in opt_params:
            for a in orig_args:
                if a in opt_params[name]:
                    fullname = f"{i_op:04d}.{name}.{a}"
                    keys.append(fullname)

    return set(keys)


# Easy helper for defining MLP from layers and activations
def quick_mlp_spec(layer_sizes, activation):

    spec_list = []

    for i in range(len(layer_sizes)-1):
        d_in = layer_sizes[i]
        d_out = layer_sizes[i+1]

        spec_list.append(dense(d_in, d_out))
    
        # apply activation
        is_last = (i+2 == len(layer_sizes))
        if not is_last:
            if activation == 'relu':
                spec_list.append(relu())
            elif activation == 'elu':
                spec_list.append(elu())
            else: raise ValueError("unrecognized activation")

    spec_list.append(squeeze_last())

    return spec_list

def func_from_spec(mode='default'):
    # be careful of mutable default arg ^^^

    def eval_spec(params, x, mode_dict=None):
        N_op = n_ops(params)

        # walk the list of operations, evaluating each
        # TODO generalize w/ data tape to not assume linear dataflow
        for i_op in range(N_op):
            name, args = get_op_data(params, i_op)
            if mode_dict is not None:
                args.update(mode_dict)
            if "_" in args:
                del args["_"] 
            x = apply_func[mode][name](x, **args)
        return x

    return eval_spec


def bounded_func_from_spec(mode='affine'):
    # be careful of mutable default arg ^^^

    def eval_spec(params, x, mode_dict=None):
        N_op = n_ops(params)
        bound_dict = {}
        # walk the list of operations, evaluating each
        # TODO generalize w/ data tape to not assume linear dataflow
        for i_op in range(N_op):
            name, args = get_op_data(params, i_op)
            if mode_dict is not None:
                args.update(mode_dict)
            if "_" in args:
                del args["_"]
            if mode_dict['ctx'].mode == 'affine+backward':
                if name == 'dense':
                    x = apply_func[mode][name](x, **args)
                    bound_dict[i_op] = {}
                    bound_dict[i_op]['name'] = 'dense'
                    bound_dict[i_op]['A_l'] = torch.as_tensor(args['A']).squeeze().T
                    bound_dict[i_op]['A_u'] = torch.as_tensor(args['A']).squeeze().T
                    if i_op == N_op - 2:
                        bound_dict[i_op]['b_l'] = torch.as_tensor(args['b'])
                        bound_dict[i_op]['b_u'] = torch.as_tensor(args['b'])
                    else:
                        bound_dict[i_op]['b_l'] = torch.as_tensor(args['b']).unsqueeze(-1)
                        bound_dict[i_op]['b_u'] = torch.as_tensor(args['b']).unsqueeze(-1)
                elif name == 'relu':
                    x, A, b_l, b_u = apply_func[mode][name](x, **args)
                    bound_dict[i_op] = {}
                    bound_dict[i_op]['name'] = 'relu'
                    bound_dict[i_op]['A_l'] = torch.diag(A)
                    bound_dict[i_op]['A_u'] = torch.diag(A).clone()
                    bound_dict[i_op]['b_l'] = b_l.unsqueeze(-1)
                    bound_dict[i_op]['b_u'] = b_u.unsqueeze(-1)
            else:
                x = apply_func[mode][name](x, **args)

        return bound_dict

    return eval_spec


def func_as_torch(params):
    op_list = []
    N_op = n_ops(params)
    for i_op in range(N_op):
        name, args = get_op_data(params, i_op)
        # print(name)
        # self.op_list.append((name, args))
        if name == 'dense':
            A = torch.tensor(args['A']).T#.to(device)
            b = torch.tensor(args['b'])#.to(device)
            linear = torch.nn.Linear(A.shape[1], A.shape[0])
            linear.weight = torch.nn.Parameter(A)
            linear.bias = torch.nn.Parameter(b)
            op_list.append(linear)
        elif name == 'relu':
            op_list.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*op_list)
    # model.to(device)
    # print("Torch Model: ")
    # print(model)
    return model


# ===== Utilities

def get_op_data(params, i_op):
    i_op_str = f"{i_op:04d}"
    name = ""
    args = {}
    for key in params:
        if key.startswith(i_op_str):
            tokens = key.split(".")
            name = tokens[1]
            if len(tokens) > 2:
                argname = tokens[2]
                args[argname] = params[key]
    
    if name == "": 
        print(params.keys())
        raise ValueError(f"didn't find op {i_op}")

    return name, args

def n_ops(params):
    n = 0
    for key in params:
        vals = key.split(".")
        try:
            i_op = int(vals[0])
        except ValueError:
            raise ValueError(f"Could not parse out key {key}. Is this a valid mlp spec? Did you make a mistake passing params dictionaries around?")
        n = max(n, i_op+1)
    return n

# helper to add an operation to an existing MLP
# call like:
#   params = prepend_op(params, spatial_transformation())
def prepend_op(params, op):
    new_params = {}

    # increment the op ind in the key of all existing ops
    for key in params:
        vals = key.split(".")
        i_op = int(vals[0])
        i_op += 1
        vals[0] = f"{i_op:04d}"
        new_key = ".".join(vals)
        new_params[new_key] = params[key]

    # add the new op
    N = n_ops(params)
    for key, val in op.items():
        key = f"{0:04d}." + key
        new_params[key] = val

    return new_params

def check_rng_key(key):
    if key is None:
        raise ValueError("to initialize model weights, must pass an RNG key")

def load(filename):
    out_params = {}
    param_count = 0
    with np.load(filename) as data:
        for key,val in data.items():
            # print(f"mlp layer key: {key}")
            # convert numpy to jax arrays
            if isinstance(val, np.ndarray):
                param_count += val.size
                val = np.array(val)
            out_params[key] = val
    print(f"Loaded MLP with {param_count} params")
    return out_params

def save(filename, params):

    np_params = {} # copy to a new dict, we will modify
    for key, val in params.items():
        # convert jax to numpy arrays
        if isinstance(val, np.ndarray):
            val = np.array(val)
        np_params[key] = val

    np.savez(filename, **np_params)    


# ===== Listing of layer types and associated functions
# These are populated below, along with the creation functions themselves

# Initializes array buffers for the functions
initialize_func = {}

# A list of the keys which need to be optimized during training
opt_params = {}

# These are populated in 'affine_layers' and 'slope_interval_layers', respectively.
# TODO bad software design: need to import affine_layers, etc later for these to get populated
apply_func = {
        'default' : {},
        'affine' : {},
        'slope_interval' : {}
    }


# == Dense linear layer

def dense(in_dim, out_dim, with_bias=True, A=None, b=None):
    if(not with_bias and b is not None):
        raise ValueError("cannot specifify 'b' and 'with_bias=False'")

    # initialize A
    if A is None:
        # random initialize later
        A = (in_dim, out_dim)
    else:
        # use the input
        A = np.array(A)
        if A.shape != (in_dim,out_dim):
            raise ValueError(f"A should have shape ({in_dim},{out_dim}). Has shape {A.shape}.")
    
    # initialize b
    if b is None and with_bias:
        # random initialize later
        b = (out_dim,)
    else:
        # use the input
        b = np.array(b)
        if b.shape != (out_dim,):
            raise ValueError(f"b should have shape ({out_dim}). Has shape {b.shape}.")

    subdict = {
      "dense.A" : A,
    }
   
    if with_bias:
        subdict["dense.b"] = b

    return subdict

opt_params['dense'] = ['A', 'b']

def default_dense(input, A, b):
    A = torch.tensor(A, dtype=input[0].dtype, device=input[0].device)
    out = torch.matmul(input, A)
    if b is not None:
        b = torch.tensor(b, dtype=input[0].dtype, device=input[0].device)
        out += b
    return out
apply_func['default']['dense'] = default_dense

def initialize_dense(rngkey=None, A=None, b=None):
    if isinstance(A, tuple): # if A needs initialization, it is a tuple giving the size
        check_rng_key(rngkey)
        subkey, rngkey = split_generator(rngkey)
        initF = torch.nn.functional.initializers.glorot_normal()
        A = initF(subkey, A)
    if isinstance(b, tuple): # if b needs initialization, it is a tuple giving the size
        check_rng_key(rngkey)
        subkey, rngkey = split_generator(rngkey)
        initF = torch.nn.functional.initializers.normal()
        b = initF(subkey, b)

    out_dict = { 'A' : A }
    if b is not None:
        out_dict['b'] = b

    return out_dict
initialize_func['dense'] = initialize_dense



# == Common activations

def relu():
    return {"relu._" : np.array([])}
def default_relu(input):
    return torch.nn.functional.relu(input)
apply_func['default']['relu'] = default_relu

def elu():
    return {"elu._" : np.array([])}
def default_elu(input):
    return torch.nn.functional.elu(input)
apply_func['default']['elu'] = default_elu


def sin():
    return {"sin._" : np.array([])}
def default_sin(input):
    return np.sin(input)
apply_func['default']['sin'] = default_sin

# == Positional encoding

def pow2_frequency_encode(count_pow2, start_pow=0, with_shift=True):
    pows = np.power(2., np.arange(start=start_pow, stop=start_pow+count_pow2, dtype=float))
    coefs = pows * np.pi
    
    if with_shift:
        coefs = np.repeat(coefs, 2)
        shift = np.zeros_like(coefs)
        shift = shift.at[1::2].set(np.pi)
        return {"pow2_frequency_encode.coefs" : coefs, "pow2_frequency_encode.shift" : shift}
    else:
        return {"pow2_frequency_encode.coefs" : coefs}

def default_pow2_frequency_encode(input, coefs, shift=None):
    x = input[:,None] * coefs[None,:]
    if shift is not None:
        x += shift
    x = x.flatten()
    return x
apply_func['default']['pow2_frequency_encode'] = default_pow2_frequency_encode


# == Utility


def squeeze_last():
    return {"squeeze_last._" : np.array([])}
def default_squeeze_last(input):
    return np.squeeze(input, axis=0)
apply_func['default']['squeeze_last'] = default_squeeze_last

# R,t are a transformation for the SHAPE, input points will get the opposite transform
def spatial_transformation():
    return {
            "spatial_transformation.R" : np.eye(3),
            "spatial_transformation.t" : np.zeros(3),
            }


def default_spatial_transformation(input, R, t):
    # if the shape transforms by R,t, input points need the opposite transform
    R_inv = np.linalg.inv(R)
    t_inv = np.dot(R_inv, -t)
    return default_dense(input, A=R_inv, b=t_inv)
apply_func['default']['spatial_transformation'] = default_spatial_transformation

# TODO bad software design, see note above
import affine_layers
import slope_interval_layers
