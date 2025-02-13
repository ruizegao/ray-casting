import matplotlib.pyplot as plt
import triangle
from bouncing_letters import carve, load_net_object
import shapely
import torch

import execjs
import json

js_code = """
const decomp = require('/home/ruize/PycharmProjects/ray-casting/node_modules/poly-decomp-es');

function decomposePolygon(vertices) {
    let poly = vertices.map(v => [v[0], v[1]]);
    return decomp.decomp(poly);
}
"""

ctx = execjs.compile(js_code)

# Example polygon

c_net = load_net_object('/home/ruize/PycharmProjects/ray-casting/models/C_MLP.pth', 'mlp')
c_net = c_net.to(device=torch.device('cuda'))
c_components = carve(c_net, deep=False)
C_COMP = [shapely.geometry.Polygon(vertices) for vertices in c_components]
merged_polygon = shapely.ops.unary_union(C_COMP)

polygon = [[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]]

polygon = list(merged_polygon.exterior.coords)
print(len(polygon))
import time
t0 = time.time()
decomposed = ctx.call("decomposePolygon", polygon)
t1 = time.time()
print("time:", t1 - t0)
print(decomposed)
