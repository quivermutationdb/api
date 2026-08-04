/**
 * Quiver Mutation Database — Worker entry point.
 *
 * One Worker serves both the API (mounted at /api/*) and the static frontend
 * (Workers Static Assets). Requests that match no API route fall through to
 * the assets directory (`run_worker_first: ["/api/*"]` in wrangler.jsonc, so
 * asset requests never invoke the Worker at all).
 */

import { Hono } from "hono";
import { api } from "./api";

const app = new Hono<{ Bindings: Env }>();

app.route("/api", api);

// /api/* is routed to the Worker ahead of assets; anything unmatched here is
// an unknown API path, not a page. Pages are served directly from assets.
app.notFound((c) => c.json({ error: "Not found" }, 404));

app.onError((err, c) => {
  console.error(err);
  return c.json({ error: "Internal server error" }, 500);
});

export default app;
