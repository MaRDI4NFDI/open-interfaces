FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:e0ddebc994e54351661b6710e74fd989e92ea021

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
