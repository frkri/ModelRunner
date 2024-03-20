# Deployments with Docker Compose

This directory provides various ways to deploy modelrunner using Docker Compose
This is made possible using the [include](https://docs.docker.com/compose/multiple-compose-files/include/) element

## Getting started

The base config [`compose.yml`](compose.yml) contains only modelrunner itself

```bash
# Startup
docker compose up -d
# Cleanup
docker compose rm -sf
```

The full opinionated compose file [`compose.full.yml`](compose.full.yml) contains multiple external containers that integrate / co-exist with modelrunner.

```bash
# Startup
docker compose -f compose.full.yml up -d
# Cleanup
docker compose -f compose.full.yml rm -sf
```

## Custom compose file

In this example both the [`watchtower`](watchtower/compose.yml) and [`cf-tunnel`](cf-tunnel/compose.yml) compose files depend on the base [`compose.yml`](compose.yml) file meaning that modelrunner is always included.

```bash
# Startup
docker compose -f watchtower/compose.yml -f cf-tunnel/compose.yml up -d
# Cleanup
docker compose  -f watchtower/compose.yml -f cf-tunnel/compose.yml rm -sf
```
