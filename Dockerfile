FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:1f1286003e5ac4a466df0bcc8ce429a8284aac92

COPY . /src

RUN mkdir /build \
    && cd /build \
    && cmake /src \
    && cmake --build . \
    && cmake --install .

CMD [ "/usr/local/bin/oif_main" ]
