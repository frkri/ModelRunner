# Deployments with Docker Compose

This directory provides various ways to deploy modelrunner using Docker Compose.
In most cases, minimal configuration is required, but some services may require additional setup.

## Getting started

The base config [`compose.yml`](compose.yml) contains only modelrunner itself and no other services.
Use this to get started with the most basic setup.

```bash
# Start -> start container in detached mode
docker compose up -d

# Logs -> check logs
docker compose logs

# Stop -> stop containers
docker compose stop

# Cleanup -> force stop and remove containers
docker compose rm -sf
```

## Custom deployment

You can pick and choose which services you need using the `-f` flag followed by the path to the compose file.

In this example below both [`traefik/compose.yml`](traefik/compose.yml)
and [`watchtower/compose.yml`](watchtower/compose.yml) compose files
depend on the base [`compose.yml`](compose.yml) file meaning that modelrunner is always included.

```bash
# Start
docker compose -f traefik/compose.yml -f watchtower/compose.yml up -d

# Logs
docker compose -f traefik/compose.yml -f watchtower/compose.yml logs

# Stop
docker compose -f traefik/compose.yml -f watchtower/compose.yml stop

# Cleanup
docker compose -f traefik/compose.yml -f watchtower/compose.yml rm -sf
```

## Setup up cloudflare tunnel

The tunnel can be [`remotely managed`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/) through the dashboard or with a config file.
Here the first method is used as an example.

[`cloudflare tunnel`](cf-tunnel/compose.yml) can be setup to forward request to [`traefik`](traefik/compose.yml) by specifying the target service to `http://traefik:80`,
where traefik is the name of the service defined in the [`docker compose file`](traefik/compose.yml) and `80` the port
on which traefik listens to for requests.
> [!NOTE]
> Make sure to change the environment variable to match your tunnel token
