/** GET /nicknames — the curated class nicknames (data/nicknames.json, via class_nicknames in the main database). */

import { asc } from "drizzle-orm";
import { Hono } from "hono";
import { classNicknames as nick } from "../db/schema";
import { dbForId, mainDb } from "../db/shard";
import { mutationClasses as mc } from "../db/schema";
import { eq } from "drizzle-orm";

export const nicknamesRoutes = new Hono<{ Bindings: Env }>();

nicknamesRoutes.get("/", async (c) => {
  const rows = await mainDb(c.env).select().from(nick).orderBy(asc(nick.slug));
  const items = await Promise.all(rows.map(async (r) => {
    const db = dbForId(c.env, r.mcId);
    const cls = db ? (await db.select({ n: mc.n, dynkin: mc.dynkinType }).from(mc).where(eq(mc.id, r.mcId)))[0] : undefined;
    return { mc_id: r.mcId, nickname: r.nickname, slug: r.slug, note: r.note, added_by: r.addedBy,
             added_at: r.addedAt, num_vertices: cls?.n ?? null, dynkin_type: cls?.dynkin ?? null };
  }));
  c.header("Cache-Control", "public, max-age=300");
  return c.json({ items, total: items.length });
});
