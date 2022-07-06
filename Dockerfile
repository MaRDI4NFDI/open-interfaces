FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:090996bc494b9cc17d7248f821dc494b49be2181

ENV R_HOME=/usr/lib/R \
    R_LIBOIF_CONNECTOR=/usr/local/lib/liboif_connector.so

COPY . /src
RUN pip install --no-cache -r /src/requirements.txt

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
