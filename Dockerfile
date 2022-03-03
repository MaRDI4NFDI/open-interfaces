FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:f14ad416bb3a16d48c69206e335ed09fea83307d

COPY . /src

RUN mkdir /build \
    && cd /build \
    && cmake /src \
    && cmake --build . \
    && cmake --install . \
    && ldconfig

CMD [ "/usr/local/bin/oif_c" ]
