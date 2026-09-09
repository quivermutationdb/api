import { defineConfig } from "drizzle-kit";

// SQLITE, and deliberately so: this generates drizzle/*.sql, the schema of the
// intermediate SQLite the offline pipeline writes (qmd/d1_export.py renders
// SQL, scripts/pg-load-local.sh replays it into a SQLite, scripts/pg-export.py
// turns that into Postgres COPY text). It is NOT the schema the Worker serves
// -- that is src/db/schema.ts over Postgres, whose DDL lives in drizzle-pg/ and
// is hand-written. Never point this config at src/db/schema.ts.
export default defineConfig({
  schema: "./src/db/schema.sqlite.ts",
  out: "./drizzle",
  dialect: "sqlite",
});
