# Specifically use Debian 12 due to the runtime image running also running on Debian 12
FROM rust:1.80.1-bookworm as builder
# Compile without any optimizations by default
ARG rust_flags=""

WORKDIR /ModelRunner
COPY . /ModelRunner

ENV RUSTFLAGS=${rust_flags}
RUN cargo build --release --bin model_runner
RUN cargo build --release --bin model_runner_health

FROM gcr.io/distroless/cc-debian12 as runtime
LABEL org.opencontainers.image.source="https://github.com/frkri/ModelRunner"
LABEL org.opencontainers.image.title="ModelRunner"
LABEL org.opencontainers.image.license="MIT"

WORKDIR /ModelRunner
COPY --from=builder /ModelRunner/target/release/model_runner model_runner
COPY --from=builder /ModelRunner/target/release/model_runner_health model_runner_health

# Required for the sqlite database
COPY ./migrations migrations
# Required for the whisper model
COPY ./melfilters.bytes melfilters.bytes

# Set the RUST_LOG environment variable to 'debug' to see more verbose logs
ENV RUST_LOG=info
# By default, modelrunner will listen on port 25566
EXPOSE 25566

ENTRYPOINT ["./model_runner"]
