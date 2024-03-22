# Deployments with Docker Compose

This directory provides various ways to deploy modelrunner using Docker Compose.

## Getting started

The base config [`compose.yml`](compose.yml) contains only modelrunner itself and no other services.

```bash
# Startup
docker compose up -d
# Cleanup -> stop and remove container
docker compose rm -sf
```

## Custom deployment

You can pick and choose which services you need using the `-f` flag.

In this example below both the [`traefik`](traefik/compose.yml) and [`watchtower`](watchtower/compose.yml) compose files
depend on the base [`compose.yml`](compose.yml) file meaning that modelrunner is always included.

> [!NOTE]
> The order of compose files matters when using the `-f` flag. In this case traefik must come first in the sequence
> followed by others due to the [`compose.override.yml`](traefik/compose.override.yml) overriding the base compose file

```bash
# Startup
docker compose -f traefik/compose.yml -f watchtower/compose.yml up -d
# Cleanup
docker compose -f traefik/compose.yml -f watchtower/compose.yml rm -sf
```

## Setting up cloudflare tunnel

The tunnel can be [`remotely managed`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/) through the dashboard or with a config file.
Here the first method is used as an example here.

[`cloudflare tunnel`](cf-tunnel/compose.yml) can be setup to forward request to [`traefik`](traefik/compose.yml) by specifying the target service to `http://traefik:80`,
where traefik is the name of the service defined in the [`docker compose file`](traefik/compose.yml) and the port on which traefik listens to for requests.
> [!NOTE]
> Make sure to change the environment variable to match your tunnel token
