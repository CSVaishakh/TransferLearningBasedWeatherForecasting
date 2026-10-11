# CPU-only reproduction of the GloSAT / CascadeTabNet environment
# (Singh & Middleton; Ziomek & Middleton HIP-2021).
#
# The upstream setup pins Python 3.7, torch 1.4, mmcv 0.4.3 and mmdetection
# v1.2.0 compiled from source. Debian buster's gcc 8 can still compile those
# extensions; the host's gcc 15 cannot. No CUDA: the host GPU is AMD.
#
# Build context = the glosat_table_dataset clone (its dla/ folder is copied in):
#   docker build -t glosat-cpu -f digitization/glosat/Dockerfile ../glosat_table_dataset
FROM python:3.7-slim-buster

# buster is end-of-life; its packages now live on archive.debian.org
RUN sed -i -e 's|deb.debian.org|archive.debian.org|g' \
           -e 's|security.debian.org|archive.debian.org|g' \
           -e '/buster-updates/d' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential git libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        torch==1.4.0+cpu torchvision==0.5.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html \
 && pip install --no-cache-dir \
        "numpy<1.20" cython "pillow==6.2.1" "opencv-python-headless==4.5.5.64" \
        terminaltables matplotlib "scikit-learn<1.1" pycocotools \
 && pip install --no-cache-dir "mmcv==0.4.3"

# mmdetection v1.2.0, built without CUDA (CPU kernels only)
WORKDIR /opt
RUN git clone --depth 1 --branch v1.2.0 https://github.com/open-mmlab/mmdetection.git
WORKDIR /opt/mmdetection
# v1.2.0 refuses to build without CUDA: build CPU ops only, RoIAlign via torchvision
COPY --from=patches patch_mmdet_cpu.py /opt/patch_mmdet_cpu.py
RUN python /opt/patch_mmdet_cpu.py /opt/mmdetection \
 && pip install --no-cache-dir -e . --no-build-isolation

# GloSAT code + its mmdetection overrides (dla/src/install.py)
COPY dla /opt/mmdetection/dla
RUN python dla/src/install.py \
 && python -c "import re,pathlib; f=pathlib.Path('mmdet/datasets/voc.py'); \
f.write_text(re.sub(r'CLASSES = \([^)]*\)', \"CLASSES = ('table_body', 'cell', 'full_table', 'header', 'heading')\", f.read_text(), count=1))" \
 && grep -n "CLASSES =" mmdet/datasets/voc.py

# CPU patches: device strings and the GPU-only scatter step
COPY --from=patches patch_cpu.py /opt/patch_cpu.py
RUN python /opt/patch_cpu.py /opt/mmdetection

# models/ and data are mounted at run time (see run.sh)
VOLUME ["/models", "/data"]
CMD ["bash"]
