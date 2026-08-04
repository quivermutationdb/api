import { defineConfig } from "drizzle-kit";

// Migrations are generated with `npm run db:generate` and applied with
// `wrangler d1 migrations apply qmd --local|--remote` (see package.json),
// so no D1 HTTP credentials are needed here.
export default defineConfig({
  schema: "./src/db/schema.ts",
  out: "./drizzle",
  dialect: "sqlite",
});
