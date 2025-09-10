FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

ENV PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    KERAS_BACKEND=tensorflow \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && pip install \
      "keras==3.3.3" \
      "keras-tuner==1.4.7" \
      "matplotlib==3.8.4" \
      "biopython>=1.83" \
      "pandas>=2.0" \
      "scikit-learn>=1.4"

# Code lives in /opt (baked into image)
WORKDIR /opt
RUN git clone https://github.com/chiras/MetAnoDe.git

# Runtime working dir is /data (host-mounted)
WORKDIR /data

# Wrapper: ensure dirs, then ALWAYS copy pretrained into /data
# Default: no-overwrite (keeps any user-trained files); set PRETRAINED_OVERWRITE=1 to force overwrite.
RUN printf '#!/usr/bin/env bash\nset -e\n\
mkdir -p /data/models /data/predictions /data/plots /data/tuner\n\
copy_mode=${PRETRAINED_OVERWRITE:-0}\n\
if [ -d /opt/MetAnoDe/models ]; then\n\
  if [ \"$copy_mode\" = \"1\" ]; then\n\
    echo \"[metanode] Sync pretrained models (overwrite)\" >&2\n\
    cp -r /opt/MetAnoDe/models/. /data/models/\n\
  else\n\
    echo \"[metanode] Sync pretrained models (no-overwrite)\" >&2\n\
    cp -rn /opt/MetAnoDe/models/. /data/models/ || true\n\
  fi\n\
fi\n\
if [ -d /opt/MetAnoDe/tuner ]; then\n\
  if [ \"$copy_mode\" = \"1\" ]; then\n\
    echo \"[metanode] Sync pretrained tuner (overwrite)\" >&2\n\
    cp -r /opt/MetAnoDe/tuner/. /data/tuner/\n\
  else\n\
    echo \"[metanode] Sync pretrained tuner (no-overwrite)\" >&2\n\
    cp -rn /opt/MetAnoDe/tuner/. /data/tuner/ || true\n\
  fi\n\
fi\n\
exec python /opt/MetAnoDe/metanode.py \"$@\"\n' \
  > /usr/local/bin/run_metanode.sh && chmod +x /usr/local/bin/run_metanode.sh

ENTRYPOINT ["/usr/local/bin/run_metanode.sh"]
