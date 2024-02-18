# Specifically use Debian 12 due to the runtime image running also running on Debian 12
FROM rust:1.76.0-bookworm as builder
WORKDIR /ModelRunner
COPY . /ModelRunner
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12 as runtime
WORKDIR /ModelRunner
COPY --from=builder /ModelRunner/target/release/model_runner /ModelRunner/model_runner
# Required for the whisper model
COPY ./melfilters.bytes /ModelRunner/melfilters.bytes

# Set the RUST_LOG environment variable to 'debug' to see more verbose logs
ENV RUST_LOG=info
# Change this to the port you want to expose
EXPOSE 25566

CMD ["./model_runner"]
