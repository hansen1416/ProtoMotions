cd /workspace && git clone -b feature/hhi https://github.com/hansen1416/ProtoMotions.git && cd ProtoMotions

<!-- humos_128_offset.pt -->
pip install gdown && gdown 1iO5sTFrY41Lj2UQwVR_g5yU_t6Wgw5-K

<!-- download data from R2 -->
<!-- todo need to update rclone first -->
rclone copy r2:proto-data/merged4/ /workspace/merged4/ \
  --transfers=4 \
  --multi-thread-streams=16 \
  --multi-thread-chunk-size=128M \
  --progress

pip install -e .

wandb login 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


apt update && apt install -y tmux
tmux new -s hhi
tmux new -t hhi
tmux kill-session -t hhi


```bash
python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1_motion_128_shape \
    --motion-file ./humos_128_offset.pt \
    --num-envs 1024 \
    --batch-size 4096 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_1_motion_128_shape
```

```bash
python -u protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_1024_motion \
    --motion-file /workspace/merged4/humos_slurmrank.pt \
    --num-envs 8192 \
    --batch-size 32768 \
    --ngpu 4 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group hhi_1024_motion
```

```bash
docker run --gpus all --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --ipc=host --shm-size=16g hansen1416/hhi-protomotions-isaacgym:v1 /bin/bash
```
----


  docker pull hansen1416/hhi-protomotions-isaacgym:v1

  docker run -d \
  --name proto \
  --network=host \
  --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt:/mnt \
  hansen1416/hhi-protomotions-isaacgym:v1 \
  tail -f /dev/null

docker exec -it proto /bin/bash

----------

apt-get update
apt-get install -y ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update

apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

--

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

---------


docker run -it --name hhi-protomotions \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  hansen1416/hhi-protomotions-isaacgym:v1 \
  /bin/bash


  ------------

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo systemctl enable --now docker




curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


