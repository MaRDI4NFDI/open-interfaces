FROM docker.io/julia:1.7-bullseye

RUN apt update && apt install -y gcc g++ libjulia-dev cmake 

COPY . /src

RUN cd /src && cmake . && make

CMD [ "/src/julia_embed" ]