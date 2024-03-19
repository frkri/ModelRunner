PRAGMA foreign_keys = OFF;

CREATE TABLE api_clients_new
(
    id         text primary key not null,
    name       text,
    key        text             not null,
    created_at integer          not null,
    updated_at integer          not null,
    created_by text
);

INSERT INTO api_clients_new (id, name, key, created_at, updated_at)
SELECT id, name, key, created_at, updated_at
FROM api_clients;

DROP TABLE api_clients;
ALTER TABLE api_clients_new
    RENAME TO api_clients;

PRAGMA foreign_keys = ON;