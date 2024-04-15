-- Hack for sqlx to disable foreign_keys
COMMIT TRANSACTION;
PRAGMA foreign_keys= OFF;
BEGIN TRANSACTION;

CREATE TABLE client
(
    id          text primary key not null,
    name        text,
    key         text             not null,
    permissions integer          not null,
    created_at  integer          not null,
    updated_at  integer          not null,
    created_by  text
);

INSERT INTO client (id, name, key, permissions, created_at, updated_at, created_by)
SELECT id, name, key, 0, created_at, updated_at, created_by
FROM api_clients;
-- todo transform permissions to integer?

-- Cleanup
DROP TABLE api_clients;
DROP TABLE api_client_permission_scopes;
DROP TABLE permission_scopes;

COMMIT TRANSACTION;
PRAGMA foreign_keys= ON;
BEGIN TRANSACTION;