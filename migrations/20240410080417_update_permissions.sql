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

CREATE TEMPORARY TABLE temp_permissions
(
    permission text primary key not null,
    value      integer          not null
);

INSERT INTO temp_permissions (permission, value)
VALUES ('use', 1 << 0),    -- into USE_SELF
       ('status', 1 << 2), -- into STATUS_SELF
       ('create', 1 << 4), -- into CREATE_SELF
       ('delete', 1 << 6), -- into DELETE_SELF
       ('update', 1 << 8);
-- into UPDATE_SELF


-- Transfer over clients
INSERT INTO client (id, name, key, permissions, created_at, updated_at, created_by)
SELECT id, name, key, 0, created_at, updated_at, created_by
FROM api_clients;

-- Update permissions to new bit flags format
UPDATE client
SET permissions = (SELECT SUM(temp_permissions.value)
                   FROM api_client_permission_scopes
                            INNER JOIN permission_scopes ON api_client_permission_scopes.scope_id = permission_scopes.id
                            INNER JOIN temp_permissions ON permission_scopes.permission = temp_permissions.permission
                   WHERE api_client_permission_scopes.api_client_id = client.id);

-- Cleanup old tables
DROP TABLE api_clients;
DROP TABLE api_client_permission_scopes;
DROP TABLE permission_scopes;

COMMIT TRANSACTION;
PRAGMA foreign_keys= ON;
BEGIN TRANSACTION;