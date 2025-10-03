# Pic2Product

# 1) 新建并进入环境（Python 3.10）
conda create -n pic2product python=3.11
conda activate pic2product

# 2) 用 conda-forge 把“底座”装好（版本彼此兼容）
conda install -y -c conda-forge \
    numpy=1.26.4 pandas=2.2.2 pillow tqdm \
    opencv=4.10.0 \
    protobuf=4.25.* abseil-cpp=20240116.*

# 3) 升级 pip 工具
python -m pip install -U pip setuptools wheel

# 4) 装 PyTorch（macOS 上直接 pip）
python -m pip install "torch==2.3.*" "torchvision==0.18.*"

# 5) 业务库（带上 ultralytics 依赖）
python -m pip install \
  "ultralytics==8.3.30" "open_clip_torch==2.24.0" \
  ftfy regex sentencepiece huggingface_hub \
  matplotlib psutil py-cpuinfo "scipy<1.13" seaborn ultralytics-thop