FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:ce30e51b8114a92cd8b93c51614b46fadf563671

ENV R_HOME=/usr/lib/R \
    R_LIBOIF_CONNECTOR=/usr/local/lib/liboif_connector.so

COPY . /src

ARG M2_CXX=g++
ARG M2_CC=gcc

ENV CXX=${M2_CXX} CC=${M2_CC}

RUN mkdir /build \
    && cd /build \
    && cmake /src \
    && cmake --build . \
    && cmake --install . \
    && ldconfig

CMD [ "pytest", "-rA", "/src" ]
