FROM ubuntu:24.04 AS build

ARG ONNXRUNTIME_VERSION=1.31.1
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        cmake \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH}" in \
        amd64) ONNXRUNTIME_ARCH=x64 ;; \
        arm64) ONNXRUNTIME_ARCH=aarch64 ;; \
        *) echo "Unsupported Docker target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && mkdir -p /opt/onnxruntime-sdk \
    && curl --fail --location --silent --show-error \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz" \
        | tar -xz --strip-components=1 -C /opt/onnxruntime-sdk

WORKDIR /src
COPY cpp/ cpp/
RUN cmake -S cpp -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION}" \
        -DONNXRUNTIME_ROOT=/opt/onnxruntime-sdk \
        -DENABLE_CUDA=OFF \
        -DENABLE_OPENMP=ON \
        -DBUILD_TESTING=OFF \
    && cmake --build /build --parallel 2

FROM ubuntu:24.04 AS runtime

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/onnxruntime-sdk/lib/ /opt/onnxruntime-sdk/lib/
COPY --from=build /build/vision_engine_cli /usr/local/bin/vision_engine_cli
COPY --from=build /build/preprocess_benchmark /usr/local/bin/preprocess_benchmark

ENV LD_LIBRARY_PATH=/opt/onnxruntime-sdk/lib
ENTRYPOINT ["/usr/local/bin/vision_engine_cli"]
