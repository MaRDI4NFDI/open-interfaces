ARG REGISTRY=docker.io
FROM ${REGISTRY}/julia:1.7-bullseye

RUN apt update && apt install -y gcc g++ libjulia-dev cmake

COPY . /src

RUN mkdir /build \
    && cd /build \
    && cmake /src \
    && cmake --build . \
    && cmake --install .

CMD [ "/usr/local/bin/oif_main" ]
