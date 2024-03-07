create table api_clients
(
    id   text primary key not null,
    name text,
    key  text             not null
);

create table permission_scopes
(
    id         integer primary key autoincrement,
    permission text not null
);

create table api_client_permission_scopes
(
    api_client_id text primary key not null,
    scope_id      integer          not null,
    foreign key (api_client_id) references api_clients (id),
    foreign key (scope_id) references permission_scopes (id)
);

insert into permission_scopes (permission)
values ('status');
insert into permission_scopes (permission)
values ('create');
insert into permission_scopes (permission)
values ('delete');
