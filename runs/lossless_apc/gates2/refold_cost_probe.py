#!/usr/bin/env python3
"""Gate-5 isolated cost: ms for ONE canonical 1024-token refold (the W2 publish
op that fires at each MAMBA_BLOCK_SIZE crossing) at the deployed GDN geometry,
per layer and x24 layers. Reuses the refold-parity probe kernel setup."""
import importlib.util, os, sys, time
import torch, torch.nn.functional as F
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
dev="cuda"; torch.manual_seed(0)
H,K,V=32,128,128; DT=torch.bfloat16; NLAYERS=24
def make(T):
    q=torch.randn(1,T,H,K,device=dev,dtype=DT)
    k=F.normalize(torch.randn(1,T,H,K,device=dev,dtype=DT).float(),p=2,dim=-1).to(DT)
    v=torch.randn(1,T,H,V,device=dev,dtype=DT)
    beta=torch.rand(1,T,H,device=dev,dtype=DT).sigmoid()
    g=F.logsigmoid(torch.rand(1,T,H,device=dev,dtype=torch.float32)).to(DT)
    return q,k,v,g,beta
def fold(q,k,v,g,beta,seed):
    T=q.shape[1]; cu=torch.tensor([0,T],dtype=torch.int32,device=q.device)
    _o,hs=chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,initial_state=seed.to(v.dtype),
        output_final_state=True,cu_seqlens=cu,use_qk_l2norm_in_kernel=False)
    return hs
for T in (1024, 512, 256):
    q,k,v,g,beta=make(T); seed=torch.zeros(1,H,K,V,device=dev,dtype=torch.float32)
    for _ in range(5): fold(q,k,v,g,beta,seed)   # warmup
    torch.cuda.synchronize(); t0=time.perf_counter(); N=50
    for _ in range(N): fold(q,k,v,g,beta,seed)
    torch.cuda.synchronize(); per=(time.perf_counter()-t0)/N*1000
    print(f"T={T:4d}: {per:.3f} ms/layer-refold  x{NLAYERS} layers = {per*NLAYERS:.2f} ms/crossing")
