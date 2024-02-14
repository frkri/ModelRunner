FROM rust:1.76.0 as builder
WORKDIR /ModelRunner
COPY . /ModelRunner
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12 as runtime
WORKDIR /ModelRunner
COPY --from=builder /ModelRunner/target/release/model_runner /ModelRunner/model_runner
COPY --from=builder /ModelRunner/ModelRunner.toml.example /ModelRunner/ModelRunner.toml

# Set the environment variable to debug to see the logs
ENV RUST_LOG=info
# Change this to the port you want to expose
EXPOSE 25566

CMD ["./model_runner"]
