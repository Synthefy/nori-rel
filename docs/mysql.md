# MySQL sources

Nori-Rel reads MySQL 8.0 or later, including Amazon RDS for MySQL and Aurora
MySQL 3. MariaDB and MySQL 5.7 are refused when the connector first connects:
the SQL it generates needs MySQL 8's common table expressions and window
functions.

```bash
python -m pip install "nori-rel[mysql,local]"
# Hosted inference only:
python -m pip install "nori-rel[mysql]"
```

## Credentials and permissions

Use a user that can read the selected tables and nothing more:

```sql
CREATE USER 'nori_reader'@'%' IDENTIFIED BY '...';
GRANT SELECT ON shop.* TO 'nori_reader'@'%';
```

Keep the connection in an environment variable or in AWS Secrets Manager,
never in code. A URL names the user, host, port and, optionally, a default
database; percent-encode special characters in the password:

```bash
export MYSQL_URL='mysql://nori_reader:...@shop.abc123.us-east-1.rds.amazonaws.com:3306/shop?ssl_mode=VERIFY_IDENTITY&ssl_ca=/etc/ssl/rds/global-bundle.pem'
```

A Secrets Manager secret may hold such a URL, or the JSON fields of an RDS or
Aurora MySQL secret: `engine` (`mysql` or `aurora-mysql`), `host`, `port`
(default 3306), `username`, `password` and an optional `dbname`, plus the
optional `ssl_mode` and `ssl_ca` options below. A secret that RDS manages for
the master user holds only the user name and password; store a complete one.

A user without a password can log in to RDS or Aurora MySQL with
[IAM database authentication](#rds-iam-authentication).

## Connect and predict

```python
from datetime import UTC, datetime

from nori_rel import EnvironmentCredential, NoriRelClient

with NoriRelClient() as client:
    shop = client.connect(
        engine="mysql",
        credential=EnvironmentCredential(variable="MYSQL_URL"),
        schema_name="shop",
        tables=["users", "orders"],
        time_columns={"orders": "created_at"},
    )
    predictions = client.predict(
        source=shop,
        query="PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id",
        anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
        anchor_time=datetime(2026, 3, 1, tzinfo=UTC),
        observed_through=datetime(2026, 3, 1, tzinfo=UTC),
        entity_created_at="signed_up_at",
    )
```

The [README example](../README.md#predict-customer-spend) explains each
argument; they mean the same on every engine.

`schema_name` is required: it is the MySQL database to read, whatever
default database the URL names. Relationships come from declared `FOREIGN KEY`
constraints, which InnoDB keeps. Explicit `primary_keys` and `foreign_keys`
mappings are available for warehouse engines only, so a MySQL schema without
declared keys needs them added before Nori-Rel can follow its links. Views are
not read; only base tables are.

## TLS

The `ssl_mode` URL option takes the MySQL client's names:

| `ssl_mode` | Behavior |
| --- | --- |
| `DISABLED` | No TLS. Local hosts only. |
| `PREFERRED` | TLS when the server offers it, without certificate checks. Local hosts only; the default for them. |
| `REQUIRED` | TLS without certificate checks. The default for a remote host. |
| `VERIFY_CA` | TLS and a certificate signed by `ssl_ca`, or by a system CA without it. |
| `VERIFY_IDENTITY` | `VERIFY_CA`, and the certificate names the host. |

A local host is `localhost`, `127.0.0.1` or `::1`; every other host needs
`REQUIRED`, `VERIFY_CA` or `VERIFY_IDENTITY`. `ssl_ca` is the path to a PEM
bundle: for RDS and Aurora, download the
[global bundle](https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem).
Other URL options are refused rather than passed to the driver, so a URL
cannot enable `LOAD DATA LOCAL` or run SQL at connect time.

## Through an SSH tunnel

When the database is reachable only through a bastion host, open a tunnel
and point the URL at its local end:

```bash
ssh -N -L 127.0.0.1:3307:shop.abc123.us-east-1.rds.amazonaws.com:3306 you@bastion.example.com
export MYSQL_URL='mysql://nori_reader:...@127.0.0.1:3307/shop'
```

- Nori-Rel does not open tunnels. Keep the `ssh` process running while you
  connect and predict; it can use your SSH config, keys and agent as usual.
- SSH encrypts the hop to the bastion. The URL's host is local, so `ssl_mode`
  defaults to `PREFERRED`: RDS offers TLS, so the connection uses it and a
  server with `require_secure_transport=ON` accepts it.
- To check the server certificate through the tunnel, use
  `ssl_mode=VERIFY_CA&ssl_ca=...`. `VERIFY_IDENTITY` fails, because the
  certificate names the RDS endpoint, not 127.0.0.1.
- The tunnel carries the connection; the database user still logs in, with
  its MySQL password or with an [RDS IAM token](#rds-iam-authentication).

## RDS IAM authentication

With
[IAM database authentication](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.html),
the MySQL user has no password. Each new connection logs in with a token
that your AWS identity signs, valid for 15 minutes; Nori-Rel signs one for
every connection it opens, so long predictions keep working.

1. Turn on IAM database authentication for the instance or cluster.
2. Create the user for AWS's plugin and let it read the tables:

   ```sql
   CREATE USER 'nori_reader'@'%' IDENTIFIED WITH AWSAuthenticationPlugin AS 'RDS';
   GRANT SELECT ON shop.* TO 'nori_reader'@'%';
   ```

3. Allow your AWS identity to connect as that user. The resource names the
   instance's resource ID (`db-...`) or the cluster's (`cluster-...`):

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": "rds-db:connect",
         "Resource": "arn:aws:rds-db:us-east-1:123456789012:dbuser:db-ABCDEFGHIJKL01234/nori_reader"
       }
     ]
   }
   ```

Connect with `RdsIamToken` and a URL without a password:

```python
from nori_rel import RdsIamToken

credential = RdsIamToken(
    url="mysql://nori_reader@shop.abc123.us-east-1.rds.amazonaws.com:3306/shop"
    "?ssl_mode=VERIFY_IDENTITY&ssl_ca=/etc/ssl/rds/global-bundle.pem",
)
shop = client.connect(engine="mysql", credential=credential, schema_name="shop")
```

Through an SSH tunnel, the URL names the tunnel's local end and `endpoint`
names the endpoint the token is signed for:

```python
credential = RdsIamToken(
    url="mysql://nori_reader@127.0.0.1:3307/shop"
    "?ssl_mode=VERIFY_CA&ssl_ca=/etc/ssl/rds/global-bundle.pem",
    endpoint="shop.abc123.us-east-1.rds.amazonaws.com:3306",
)
```

- The token is signed for an instance, cluster or proxy endpoint and its
  port, by default the URL's host and port. AWS refuses a token signed for a
  custom DNS name, such as a Route 53 record, so name the RDS endpoint.
  A local URL without `endpoint` is refused.
- The token travels as a clear-text password, so TLS is required even
  through a tunnel: `ssl_mode` defaults to `REQUIRED`, and `PREFERRED` and
  `DISABLED` are refused. Check the certificate with `VERIFY_IDENTITY` on a
  direct connection and `VERIFY_CA` through a tunnel.
- Signing happens in your process with boto3's default credentials, such as
  `AWS_PROFILE` or an instance role, and calls no AWS API. Temporary
  credentials must still be valid whenever a new connection opens.
- `region_name` defaults to the Region in the endpoint name, `us-east-1`
  above, and otherwise to boto3's default Region.
- AWS refusing a token, for example for a policy that does not name the user,
  reports `MySQL connection failed (code=1045)`.

## Snapshot and time behavior

- Each operation runs in one `START TRANSACTION WITH CONSISTENT SNAPSHOT, READ
  ONLY` transaction, in a `REPEATABLE READ` session with
  `transaction_read_only=ON`. Labels and feature tables come from the same
  InnoDB read view; tables in another storage engine, such as MyISAM, have no
  snapshot isolation.
- The session time zone is UTC. `TIMESTAMP` values are converted to UTC by
  the server; `DATETIME` and `DATE` values are read as UTC as stored, so store
  them in UTC.
- `RelationalLimits.statement_timeout_ms` becomes the session's
  `max_execution_time`, which bounds each `SELECT`. The driver's socket
  timeouts are 30 seconds longer.
- An open read transaction holds metadata locks on the tables it read, so DDL
  on them waits until the read finishes.
- Errors report the MySQL error number, e.g. `MySQL connection failed
  (code=1045)` for a refused login or 3159 for a server that requires TLS,
  but no SQL text, parameters or server message.

## Types

| MySQL type | Nori-Rel reads it as |
| --- | --- |
| `TINYINT(1)`, `BOOLEAN` | Boolean |
| `TINYINT`, `SMALLINT`, `MEDIUMINT`, `INT`, `BIGINT`, with or without `UNSIGNED` | Integer; an `UNSIGNED BIGINT` key above 2^63 - 1 stays exact |
| `DECIMAL`, `FLOAT`, `DOUBLE` | Number |
| `CHAR`, `VARCHAR`, `TEXT` types, `ENUM` | Text |
| `DATE`, `DATETIME`, `TIMESTAMP` | Time, in UTC |

Other types, such as `JSON`, `SET`, `TIME`, `YEAR`, `BIT` and binary types,
are read into feature tables but cannot be PQL targets, filters or keys. A
date pandas cannot represent, such as a `9999-12-31` sentinel, stays as read in
a feature table and fails in a registered time column.

## Tests

See [testing](testing.md#mysql) for the live MySQL suite.
