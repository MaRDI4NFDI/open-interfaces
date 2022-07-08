FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:042fcab2da31cfd2a1508d8d1c0d72dae74df80f

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
