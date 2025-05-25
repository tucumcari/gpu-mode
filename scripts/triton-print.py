# https://discord.com/channels/1189498204333543425/1189607595451895918/1375926203067797698

import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

# Dummy wrapper
class Dummyhook:
    def module_init(self, module):
        print(f"module init called with module: {module}")

dummyhook = Dummyhook()

# Hook to run after compilation
def my_compiled_hook(*args, **kwargs):
    fn = kwargs["fn"]
    compile_info = kwargs["compile"]
    device = compile_info["device"]
    key = compile_info["key"]

    # Get compiled kernel object
    # compiled_kernel = fn.jit_function.cache[device][key]
    compiled_kernel = fn.jit_function.device_caches[device][0][key]
    
    # Extract and initialize 
    cu_module = compiled_kernel.module
    print(f"[compiled_hook] Initializing with CUmodule: {cu_module}")
    dummyhook.module_init(cu_module)

# Register global hook
JITFunction.compiled_hook = my_compiled_hook

# Example Triton kernel
@triton.jit
def my_kernel(X_ptr):
    pid = tl.program_id(0)
    x = tl.load(X_ptr + pid)
    tl.store(X_ptr + pid, x + 1)

# Allocate tensor
X = torch.arange(16, dtype=torch.float32, device="cuda")

# Trigger compilation and kernel launch
my_kernel[(1,)](X)
