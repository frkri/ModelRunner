# Deployments with Docker Compose

This directory provides various ways to deploy modelrunner using Docker Compose

## Getting started

The base config [`compose.yml`](compose.yml) contains only modelrunner itself

```bash
# Startup
docker compose up -d
# Cleanup
docker compose rm -sf
```

## Custom combinations of compose files

In this example both the [`traefik`](traefik/compose.yml) and [`watchtower`](watchtower/compose.yml) compose files
depend on the base [`compose.yml`](compose.yml) file meaning that modelrunner is always included.
> [!NOTE]
> The order of compose files matter when using the `-f` flag! In this case traefik must come first in the sequence
> followed by others due to [`compose.override.yml`](traefik/compose.override.yml)

```bash
# Startup
docker compose -f traefik/compose.yml -f watchtower/compose.yml up -d
# Cleanup
docker compose -f traefik/compose.yml -f watchtower/compose.yml rm -sf
```
