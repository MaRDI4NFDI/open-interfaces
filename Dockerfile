FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:894bbca696659263a79e11cafcf858ebb4723a06

ENV R_HOME=/usr/lib/R \
    R_LIBOIF_CONNECTOR=/usr/local/lib/liboif_connector.so

COPY . /src

RUN mkdir /build \
    && cd /build \
    && cmake /src \
    && cmake --build . \
    && cmake --install . \
    && ldconfig

CMD [ "pytest", "-rA", "/src" ]
