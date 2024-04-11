-- Hack for sqlx to disable foreign_keys
COMMIT TRANSACTION;
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;

-- Update permissions and rename tables
PRAGMA foreign_keys = OFF;

ALTER TABLE api_clients
    RENAME TO client;

-- Update permissions, deleting old permissions table
create table permission
(
    id          integer primary key,
    description text not null
);

-- Insert new permission scopes
insert into permission
values (0, 'UseSelf');
insert into permission
values (1, 'UseOther');
insert into permission
values (2, 'StatusSelf');
insert into permission
values (3, 'StatusOther');
insert into permission
values (4, 'CreateSelf');
insert into permission
values (5, 'CreateOther');
insert into permission
values (6, 'DeleteSelf');
insert into permission
values (7, 'DeleteOther');
insert into permission
values (8, 'UpdateSelf');
insert into permission
values (9, 'UpdateOther');

-- Update client permissions, deleting old table
create table client_permission
(
    client_id     text    not null,
    permission_id integer not null,
    foreign key (client_id) references client (id),
    foreign key (permission_id) references permission (id)
);

--- Updating old permission to new ones
-- Use -> UseSelf
UPDATE api_client_permission_scopes
SET scope_id = 0
where scope_id = 1;

-- Status -> StatusSelf
UPDATE api_client_permission_scopes
SET scope_id = 2
where scope_id = 2;

-- Create -> CreateSelf
UPDATE api_client_permission_scopes
SET scope_id = 4
where scope_id = 3;

-- Delete -> CreateSelf
UPDATE api_client_permission_scopes
SET scope_id = 6
where scope_id = 4;

-- Update -> UpdateSelf
UPDATE api_client_permission_scopes
SET scope_id = 8
where scope_id = 5;

-- Rename columns on old table, allows easier inserts
ALTER TABLE api_client_permission_scopes
    RENAME api_client_id TO client_id;
ALTER TABLE api_client_permission_scopes
    RENAME scope_id TO permission_id;

-- Finally transfer items
INSERT INTO client_permission
SELECT *
from api_client_permission_scopes;

-- Cleanup
DROP TABLE api_client_permission_scopes;
DROP TABLE permission_scopes;

COMMIT TRANSACTION;
PRAGMA foreign_keys=ON;
BEGIN TRANSACTION;