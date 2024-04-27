# Telemetry

Includes Grafana, Tempo and Prometheus

- Grafana is setup with dashboards and datasources out of the box
- Traefik can be setup to export metric and traces

## Pipeline

```mermaid
  graph TD;
    ModelRunner-->Otel-Collector;
    
    Otel-Collector<-->Tempo<-->Grafana;
    Otel-Collector<-->Prometheus<-->Grafana;
```
