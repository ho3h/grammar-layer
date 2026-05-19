"""Apply schema constraints + indexes idempotently."""

from __future__ import annotations

from neograph.cypher import NeographClient
from neograph.util import exit_marker, get_logger

log = get_logger("neograph.migrate")


def main() -> None:
    with NeographClient() as c:
        c.apply_schema()
        rows = c.run("SHOW INDEXES YIELD name, type, state RETURN name, type, state ORDER BY name")
        log.info("Indexes after migration:")
        for r in rows:
            log.info("  %s [%s] %s", r["name"], r["type"], r["state"])
        n_vec = sum(1 for r in rows if r["type"] == "VECTOR")
        n_constraints = c.run("SHOW CONSTRAINTS YIELD name RETURN count(*) AS n")[0]["n"]
        exit_marker(
            "schema-applied",
            ok=(n_vec >= 4 and n_constraints >= 9),
            vector_indexes=n_vec,
            constraints=n_constraints,
        )


if __name__ == "__main__":
    main()
