FROM manimcommunity/manim:v0.19.0

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-core fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

USER manimuser
WORKDIR /workspace

# Runtime được khởi động với network=none, rootfs read-only và chỉ /output có quyền ghi.
ENTRYPOINT []
