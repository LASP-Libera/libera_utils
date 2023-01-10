

CREATE TABLE l1b_cam (
    id SERIAL PRIMARY KEY,
    file_name TEXT UNIQUE NOT NULL,
    revision INTEGER NOT NULL
);


CREATE TABLE l1b_cam_pds_file_jt (
    lib_cam_file_id INTEGER NOT NULL,
    FOREIGN KEY (lib_cam_file_id) REFERENCES l1b_cam(id),
    pds_file_id INTEGER NOT NULL,
    FOREIGN KEY (pds_file_id) REFERENCES pds_file(id),
    PRIMARY KEY (lib_cam_file_id, pds_file_id)
);


CREATE TABLE l1b_rad (
    id SERIAL PRIMARY KEY,
    file_name TEXT UNIQUE NOT NULL,
    revision INTEGER NOT NULL
);
