create table api_clients
(
    id   integer primary key autoincrement not null,
    name text,
    key  text                              not null
);
