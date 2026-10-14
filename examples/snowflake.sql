-- Run in your chosen disposable example schema before loading the generated CSVs.
-- Unquoted names are uppercase in Snowflake; Nori-Rel resolves them unambiguously.
CREATE TABLE users (
    user_id BIGINT NOT NULL,
    created_at TIMESTAMP_TZ NOT NULL,
    country VARCHAR NOT NULL,
    PRIMARY KEY (user_id)
);
CREATE TABLE orders (
    order_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    event_time TIMESTAMP_TZ NOT NULL,
    amount NUMBER(12, 2) NOT NULL,
    PRIMARY KEY (order_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
