-- Run in your chosen disposable Unity Catalog schema, with the session in UTC.
SET TIME ZONE 'UTC';
CREATE TABLE users (
    user_id BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    country STRING NOT NULL
) USING DELTA;
CREATE TABLE orders (
    order_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    event_time TIMESTAMP NOT NULL,
    amount DECIMAL(12, 2) NOT NULL
) USING DELTA;
-- Supply primary_keys and foreign_keys to client.connect(), as in the guide.
