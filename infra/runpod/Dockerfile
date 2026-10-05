FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    openssh-server \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/^#*PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/^#*PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub

COPY infra/runpod/start.sh /usr/local/bin/start-runpod
RUN chmod +x /usr/local/bin/start-runpod

WORKDIR /workspace
EXPOSE 22
CMD ["/usr/local/bin/start-runpod"]
