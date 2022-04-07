FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:4e626b3e1ad850f9b2285c14b5bc7b691b51a549

COPY . /src

RUN mkdir /build \
    && cd /build \
    && cmake /src \
    && cmake --build . \
    && cmake --install . \
    && ldconfig

ENV R_HOME /usr/lib/R

CMD [ "/usr/local/bin/oif_c" ]
