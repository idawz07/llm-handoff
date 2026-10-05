# The facade itself needs no GPU. The *engines* it launches must exist inside the
# container (or be reachable over the network), which is why the image is slim and
# mounts engines/models from the host.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="handoff-router" \
      org.opencontainers.image.description="One GPU, many engines: a VRAM-gated backend router" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY handoff_router ./handoff_router
RUN pip install --no-cache-dir .

# nvidia-smi is injected by the NVIDIA container runtime; psutil-style process
# control requires the container to allow signals to the child process groups.
ENV HANDOFF_ROUTER_CONFIG=/config/router-backends.yaml
EXPOSE 8081

ENTRYPOINT ["handoff-router"]
CMD ["--host", "0.0.0.0", "--port", "8081"]
