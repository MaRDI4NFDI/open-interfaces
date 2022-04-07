FROM zivgitlab.wwu.io/ag-ohlberger/mardi/container/m2-dev:1f829f4b013d00e6cbe05c38636f1de738daadcf

COPY . /src

RUN mkdir /build \
    && cd /build \
    && cmake /src \
    && cmake --build . \
    && cmake --install . \
    && ldconfig

ENV R_HOME /usr/lib/R

CMD [ "/usr/local/bin/main_c" ]
