"""Make mmdetection v1.2.0 build and run without CUDA.

Upstream setup.py refuses to build without CUDA, and every op module imports
its compiled CUDA extension at import time. For CPU inference with the GloSAT
Cascade R-CNN only two compiled pieces are needed (compiling_info, nms_cpu);
RoIAlign switches to torchvision's CPU kernel, which matches mmdetection's
legacy aligned=False behaviour that the GloSAT config uses.

    python patch_mmdet_cpu.py /opt/mmdetection
"""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])


def edit(rel, old, new, count=1):
    f = root / rel
    src = f.read_text()
    if src.count(old) < count:
        sys.exit(f"{rel}: expected text not found: {old[:60]!r}")
    f.write_text(src.replace(old, new))
    print("patched", rel)


# 1. setup.py: build only the CPU extensions, as plain C++ extensions
setup = (root / "setup.py").read_text()
start = setup.index("        ext_modules=[")
end = setup.index("        cmdclass=")
setup = (setup[:start]
         + "        ext_modules=[\n"
         + "            CppExtension('mmdet.ops.utils.compiling_info', ['mmdet/ops/utils/src/compiling_info.cpp']),\n"
         + "            CppExtension('mmdet.ops.nms.nms_cpu', ['mmdet/ops/nms/src/nms_cpu.cpp']),\n"
         + "        ],\n"
         + setup[end:])
setup = setup.replace("from torch.utils.cpp_extension import BuildExtension, CUDAExtension",
                      "from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension")
(root / "setup.py").write_text(setup)
print("patched setup.py (CPU-only extensions)")

# 2. compiling_info.cpp includes the CUDA runtime header unconditionally
edit("mmdet/ops/utils/src/compiling_info.cpp",
     "#include <cuda_runtime_api.h>\n",
     "#ifdef WITH_CUDA\n#include <cuda_runtime_api.h>\n#endif\n")

# 3. CUDA-only op modules become optional imports (unused by this model)
for f in (root / "mmdet" / "ops").rglob("*.py"):
    src = f.read_text()
    new = re.sub(r"^from \. import ((?:\w+_cuda(?:, )?)+)$",
                 lambda m: "\n".join(f"try:\n    from . import {n}\nexcept ImportError:\n    {n} = None"
                                     for n in m.group(1).split(", ")),
                 src, flags=re.M)
    new = new.replace("from . import nms_cpu, nms_cuda",
                      "from . import nms_cpu\ntry:\n    from . import nms_cuda\nexcept ImportError:\n    nms_cuda = None")
    if new != src:
        f.write_text(new)
        print("optional CUDA import:", f.relative_to(root))

# 4. RoIAlign: use torchvision's kernel by default (CPU-capable)
edit("mmdet/ops/roi_align/roi_align.py",
     "                 use_torchvision=False,",
     "                 use_torchvision=True,")
