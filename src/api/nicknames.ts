/** GET /nicknames — the curated class nicknames (data/nicknames.json, via class_nicknames). */

import { asc, eq } from "drizzle-orm";
import { Hono } from "hono";
import { classNicknames as nick, mutationClasses as mc } from "../db/schema";
import { dbFor } from "../db/shard";

export const nicknamesRoutes = new Hono<{ Bindings: Env }>();

nicknamesRoutes.get("/", async (c) => {
  const rows = await dbFor(c.env, 0)
    .select({ mcId: nick.mcId, nickname: nick.nickname, slug: nick.slug, note: nick.note,
              addedBy: nick.addedBy, addedAt: nick.addedAt, n: mc.n, dynkin: mc.dynkinType })
    .from(nick).leftJoin(mc, eq(mc.id, nick.mcId)).orderBy(asc(nick.slug));
  c.header("Cache-Control", "public, max-age=300");
  return c.json({
    items: rows.map((r) => ({
      mc_id: r.mcId, nickname: r.nickname, slug: r.slug, note: r.note,
      added_by: r.addedBy, added_at: r.addedAt, num_vertices: r.n, dynkin_type: r.dynkin,
    })),
    total: rows.length,
  });
});
