FROM ghcr.io/astral-sh/uv:python3.12-bookworm

USER root
RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml README.md ./
COPY src src
COPY --chmod=755 run.sh run.sh

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --no-dev

ENTRYPOINT ["./run.sh"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
