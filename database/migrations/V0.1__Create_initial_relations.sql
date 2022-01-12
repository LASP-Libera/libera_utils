-- L0 file table
CREATE TABLE level0 (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    ingest_complete BOOL NOT NULL DEFAULT false,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')
);

CREATE TABLE packet_type (
    id SERIAL PRIMARY KEY,
    apid INT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE packet (
    id SERIAL PRIMARY KEY
);
