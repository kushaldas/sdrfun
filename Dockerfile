# sdrfun in a container: builds the binary against librtlsdr, libhackrf, ALSA and
# libopus, then keeps a small runtime image with just those libraries.
#
#   docker build -t sdrfun .
#   docker run --rm -p 8010:8010 --device /dev/bus/usb sdrfun web --no-play
#
# docker compose (recommended, see docker-compose.yml):
#
#   docker compose up --build -d

FROM rust:1-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      pkg-config librtlsdr-dev libhackrf-dev libasound2-dev libopus-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY assets ./assets
RUN cargo build --release

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      rtl-sdr libhackrf0 libasound2t64 libopus0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/sdrfun /usr/local/bin/sdrfun
WORKDIR /data
EXPOSE 8010
ENTRYPOINT ["sdrfun"]
CMD ["web", "--no-play"]
