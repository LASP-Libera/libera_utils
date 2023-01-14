CREATE TABLE sdp.l1b_cam (
    id SERIAL PRIMARY KEY,
    file_name TEXT UNIQUE NOT NULL,
    revision INTEGER NOT NULL,
    quality_flag INTEGER NOT NULL
);


CREATE TABLE sdp.l1b_rad (
    id SERIAL PRIMARY KEY,
    file_name TEXT UNIQUE NOT NULL,
    revision INTEGER NOT NULL,
    quality_flag INTEGER NOT NULL
);
