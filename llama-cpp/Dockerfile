# Pinned, not :server-cuda. The floating tag drifted onto a Debian with a
# PEP 668 externally-managed Python and broke this build outright; a base image
# that changes under you is not a base image. The digest is what actually
# freezes it — the tag is here so a human can see which build that is. Bump
# both together, and re-check the two assumptions below when you do.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b10362@sha256:182a26fbd68d1774860bd2a0fb5581ba3047974307eaeee64930d8bf889e0c0c

# No layers on purpose, so the built image is exactly the pinned digest:
#
#   1. curl (8.5.0) already ships in this base, which is all the compose
#      healthcheck needs. Installing it again would mean apt-get update, and
#      pulling whatever apt serves today is the drift the pin just removed.
#   2. The model is mounted from ./models and never fetched at boot, so there
#      is no Python toolchain to install — see entrypoint.sh.
#
# entrypoint.sh is bind-mounted by docker-compose rather than copied in, so
# editing it does not require a rebuild.
