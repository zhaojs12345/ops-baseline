# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native H800 baseline for the exact original FlagGems ac9 nanquantile core recipe."""
import torch
OP_NAME = 'nanquantile_scalar_out'
DTYPES = [torch.float32]
IS_INPLACE = True
_SOURCE = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
_SHAPES = [(1,), (17,), (1024,), (1025,), (128,128), (512,512), (1024,1024)]
def native():
    return torch.ops.aten.nanquantile.scalar_out

def grid():
    return [{"shape":list(shape),"variant":0} for shape in _SHAPES]

def key_shape(binding):
    return list(binding["shape"])

def build_inputs(binding,dtype,device):
    shape=tuple(binding["shape"])
    torch.manual_seed(0)
    inp=torch.randn(shape,dtype=dtype,device=device)
    inp.reshape(-1)[::5]=float("nan")
    out=torch.empty(shape[:-1],dtype=dtype,device=device)
    return (inp,0.73,-1,False),{"interpolation":"linear","out":out}

def _describe(shape,dtype):
    strides=[];size=1
    for dim in reversed(shape):strides.append(size);size*=dim
    return dict(shape=list(shape),dtype=str(dtype),stride=list(reversed(strides)),requires_grad=False)

def config(binding,dtype):
    shape=tuple(binding["shape"])
    args=[_describe(shape,dtype),0.73,-1,False];kwargs={"interpolation":"linear","out":_describe(shape[:-1],dtype)}
    return dict(inputs=dict(args=args,kwargs=kwargs),source_commit=_SOURCE,benchmark="benchmark/test_nanquantile.py",descriptor=list(shape),scope="core",variant=0,input_recipe=dict(seed=0,input="torch.randn",nan_start=0,nan_stride=5,q=0.73,dim=-1,keepdim=False,interpolation="linear"))
