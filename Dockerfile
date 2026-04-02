FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml README.md ./
COPY src src
COPY run.sh run.sh

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --no-dev

RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
