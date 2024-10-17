VERSION 0.8

alma-base:
    FROM almalinux:9
    RUN dnf update -y \
        && dnf upgrade -y

    RUN dnf install -y \
        --enablerepo crb \
        cmake gcc clang git \
        openblas openblas-devel

faiss:
    FROM +alma-base
    WORKDIR /faiss
    COPY ./faiss-sys/ci/install_faiss_c.sh .
    RUN chmod +x install_faiss_c.sh
    RUN ./install_faiss_c.sh
    SAVE ARTIFACT $HOME/.faiss_c faiss_c

rust-base:
    FROM +alma-base

    RUN curl \
        --proto '=https' \
        --tlsv1.2 \
        -sSf \
        https://sh.rustup.rs \
        | sh -s -- -y

    ENV PATH=$HOME/.cargo/bin/:$PATH

rust-with-faiss:
    FROM +rust-base
    COPY +faiss/faiss_c /faiss
    RUN cp /faiss/lib*.so /lib64/

faiss-rs:
    FROM +rust-with-faiss
    WORKDIR /faiss-rs
    COPY Cargo.toml .
    COPY Cargo.lock .
    COPY src/ src/
    COPY faiss-sys/ faiss-sys/

test:
    FROM +faiss-rs
    RUN cargo test --verbose
