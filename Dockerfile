FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:b4e1651cf114649b04d51d0edd4c9de9b5e2b277

ENV R_HOME=/usr/lib/R \
    R_LIBOIF_CONNECTOR=/usr/local/lib/liboif_connector.so

COPY . /src

ARG M2_CXX=g++
ARG M2_CC=gcc

ENV CXX=${M2_CXX} CC=${M2_CC}

RUN mkdir /build \
    && cd /build \
    && cmake -GNinja /src \
    && cmake --build . \
    && cmake --install . \
    && ldconfig

CMD [ "ninja", "-C", "/build", "test" ]
