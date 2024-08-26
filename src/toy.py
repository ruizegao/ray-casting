"""
A toy example for bounding neural network outputs under input perturbations.
"""
import torch
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class simple_model(torch.nn.Module):
    """
    A very simple 2-layer neural network for demonstration.
    """
    def __init__(self):
        super().__init__()
        # Weights of linear layers.
        self.w1 = torch.tensor([[1., -1.], [2., -1.]])
        self.w2 = torch.tensor([[1., -1.]])

    def forward(self, x):
        # Linear layer.
        z1 = x.matmul(self.w1.t())
        # Relu layer.
        hz1 = torch.nn.functional.relu(z1)
        # Linear layer.
        z2 = hz1.matmul(self.w2.t())
        return z2


model = simple_model()

# Input x.
x = torch.tensor([[1., 1.], [-1., -2.], [2., 1.]])
# Lowe and upper bounds of x.
lower = torch.tensor([[-1., -2.], [-1., -2.], [-1., -2.]])
upper = torch.tensor([[2., 1.], [2., 1.], [2., 1.]])

# Wrap model with auto_LiRPA for bound computation.
# The second parameter is for constructing the trace of the computational graph,
# and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(x))
pred = lirpa_model(x)
print(f'Model prediction: {pred}')

# Compute bounds using LiRPA using the given lower and upper bounds.
# norm = float("inf")
norm = float(2)
ptb = PerturbationLpNorm(x_L=lower, x_U=upper)
# ptb = PerturbationLpNorm(eps=torch.tensor([[0.3], [0.3], [0.3]]), norm=norm)
bounded_x = BoundedTensor(x, ptb)
print(f'Bounded perturbation: {ptb}')
# Compute bounds.
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
print(f'IBP bounds: lower={lb}, upper={ub}')
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
print(lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN'))
print(f'CROWN bounds: lower={lb.flatten()}, upper={ub.flatten()}')

# Getting the linear bound coefficients (A matrix).
required_A = defaultdict(set)
required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
lb, ub, A = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN', return_A=True, needed_A_dict=required_A)
print('CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias, where')
print(A[lirpa_model.output_name[0]][lirpa_model.input_name[0]])

# Opimized bounds, which is tighter.
lb, ub, A = lirpa_model.compute_bounds(x=(bounded_x,), method='alpha-CROWN', return_A=True, needed_A_dict=required_A)
print(f'alpha-CROWN bounds: lower={lb}, upper={ub}')
print('alpha-CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias, where')
print(A[lirpa_model.output_name[0]][lirpa_model.input_name[0]])

# forward + backward.
lb, ub, A = lirpa_model.compute_bounds(x=(bounded_x,), method='forward+backward', return_A=True, needed_A_dict=required_A)
print(f'forward+backward bounds: lower={lb}, upper={ub}')
print('forward+backward linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias, where')
print(A[lirpa_model.output_name[0]][lirpa_model.input_name[0]])

# forward optimized.
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='forward-optimized')
print(f'forward optimized bounds: lower={lb}, upper={ub}')

# layer = torch.nn.Linear(in_features=1, out_features=2)
# layer.weight = torch.nn.Parameter(torch.tensor([[1.], [1.]]))
# layer.bias = torch.nn.Parameter(torch.tensor([-1., -2.]))
#
# new_model = torch.nn.Sequential(layer, model)
# new_lirpa_model = BoundedModule(new_model, torch.empty((1, 1)))
# t_l = torch.tensor([[0.]])
# t_u = torch.tensor([[3.]])
# ptb = PerturbationLpNorm(x_L=torch.tensor([[0.]]), x_U=torch.tensor([[3.]]))
# t = torch.tensor([[0.]])
# bounded_t = BoundedTensor(t, ptb)
# print(layer(t_l), layer(t_u))
# print(new_lirpa_model.compute_bounds(x=(bounded_t,), method='CROWN'))

bound_dict = {
    0: {'name': 'dense', 'A_l': torch.tensor([[2., 1.], [-3., 4.]]), 'A_u': torch.tensor([[2., 1.], [-3., 4.]]), 'b_l': torch.zeros((2, 1)), 'b_u': torch.zeros((2, 1))},
    1: {'name': 'relu', 'A_l': torch.zeros((2, 2)), 'A_u': torch.diag(torch.tensor([0.58, 0.64])), 'b_l': torch.zeros((2, 1)), 'b_u': torch.tensor([[2.92], [6.43]])},
    2: {'name': 'dense', 'A_l': torch.tensor([[4., -2.], [2., 1.]]), 'A_u': torch.tensor([[4., -2.], [2., 1.]]), 'b_l': torch.zeros((2, 1)), 'b_u': torch.zeros((2, 1))},
    3: {'name': 'relu', 'A_l': torch.diag(torch.tensor([0., 1])), 'A_u': torch.diag(torch.tensor([0.4375, 1])), 'b_l': torch.zeros((2, 1)), 'b_u': torch.tensor([[15.75], [0]])},
    4: {'name': 'dense', 'A_l': torch.tensor([-2., 1.]), 'A_u': torch.tensor([-2., 1.]), 'b_l': torch.zeros(1), 'b_u': torch.zeros(1)},
}

lower = torch.tensor([[-2.], [-1.]])
upper = torch.tensor([[2.], [3.]])
X_0 = torch.tensor([[0.], [1.]])
epsilon = 2
print("-----------")
def pseudo_crown(bounds, x_l, x_u):
    N_ops = len(bounds)
    A_l = bounds[N_ops-1]['A_l']
    A_u = bounds[N_ops-1]['A_u']
    d_l = bounds[N_ops-1]['b_l']
    d_u = bounds[N_ops-1]['b_u']
    print(A_l.shape, A_u.shape, d_l.shape, d_u.shape)
    for i in range(len(bounds)-2, -1, -1):
        W_l = bounds[i]['A_l']
        W_u = bounds[i]['A_u']
        b_l = bounds[i]['b_l']
        b_u = bounds[i]['b_u']
        print(W_l.shape, W_u.shape, b_l.shape, b_u.shape)
        if bounds[i]['name'] == 'relu':
            d_l = d_l + torch.where(A_l > 0, A_l, 0.) @ b_l + torch.where(A_l <= 0, A_l, 0.) @ b_u
            d_u = d_u + torch.where(A_u > 0, A_u, 0.) @ b_u + torch.where(A_u <= 0, A_u, 0.) @ b_l
            A_l = torch.where(A_l > 0, A_l, 0.) @ W_l + torch.where(A_l <= 0, A_l, 0.) @ W_u
            A_u = torch.where(A_u > 0, A_u, 0.) @ W_u + torch.where(A_u <= 0, A_u, 0.) @ W_l
        elif bounds[i]['name'] == 'dense':
            # print(d_l.shape, A_l.shape, b_l.shape)
            d_l = d_l + A_l @ b_l
            d_u = d_u + A_u @ b_u
            A_l = A_l @ W_l
            A_u = A_u @ W_u
        print(A_l, A_u, d_l, d_u)
    # print(A_l.shape, lower.shape, d_l.shape)
    # lower_bound = A_l @ x_0 - torch.norm(A_l, p=1) * eps + d_l
    # upper_bound = A_u @ x_0 + torch.norm(A_u, p=1) * eps + d_u
    center = (x_l + x_u) / 2.0
    diff = (x_u - x_l) / 2.0
    lower_bound = A_l @ center - A_l.abs() @ diff + d_l
    upper_bound = A_u @ center + A_u.abs() @ diff + d_u
    # print(lower_bound.shape)
    return lower_bound, upper_bound

results = pseudo_crown(bound_dict, x_l=lower, x_u=upper)
print(results)

# print(bound_dict[0]['A_l'] @ lower)
# print(lower.flatten() @ bound_dict[0]['A_u'].T)
#
# diagonal = torch.diag(torch.tensor([3., 2., 2.5]))
# print(diagonal)
# diagonal = torch.diag(diagonal)
# print(diagonal)
