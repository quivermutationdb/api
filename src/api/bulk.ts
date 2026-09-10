/**
 * /api/bulk/* — the full census as files, served from R2.
 *
 * Why this exists at all: a complete pull through /api/export.ndjson is 50.8 M
 * rows of streamed database reads, which is slow for the user and pointless
 * load for a PS-40. The same bytes sit in R2 as ~one file per rank, and R2
 * egress is free where PlanetScale's is metered past 100 GB/month.
 *
 *   GET /api/bulk                 JSON index: files, sizes, checksums, licence
 *   GET /api/bulk/manifest.json   the manifest as generated
 *   GET /api/bulk/{file}          a corpus file (gzipped NDJSON, or aux JSON)
 *
 * Range requests are honoured so a 2 GB download can resume; this is the one
 * place on the site where a user is expected to be moving gigabytes, and a
 * transfer that cannot resume is a transfer that fails.
 *
 * Content is built by scripts/r2-build-corpus.py, whose rows are byte-for-byte
 * what /api/export.ndjson serves.
 */

import { Hono } from "hono";
import { Unavailable } from "./errors";

export const bulkRoutes = new Hono<{ Bindings: Env }>();

/**
 * Corpus filenames only; anything else 404s before R2 is touched. A rank is one
 * file when it fits, or numbered parts when it does not (rank 6 is ~1.3 GB) --
 * each part is an independent gzip member, so `cat qmd-n6.part*.ndjson.gz |
 * gunzip` reads the rank as one stream.
 */
const FILE = /^(?:qmd-n\d+(?:\.part\d+)?\.ndjson\.gz|manifest\.json|rank_stats\.json|nicknames\.json)$/;

function bucket(env: Env): R2Bucket {
  const b = env.BULK;
  if (!b) throw new Unavailable("bulk corpus is not configured");
  return b;
}

/** Index: what is here, how big, how to verify it, how to cite it. */
bulkRoutes.get("/bulk", async (c) => {
  const obj = await bucket(c.env).get("manifest.json");
  if (!obj) {
    return c.json({
      detail: "The bulk corpus has not been published yet. Use /api/export.ndjson "
        + "for now (resumable via the X-Next-Cursor header).",
    }, 503);
  }
  const manifest = await obj.json<Record<string, unknown>>();
  c.header("Cache-Control", "public, max-age=3600");
  return c.json({
    ...manifest,
    base_url: new URL("/api/bulk/", c.req.url).toString(),
    usage: {
      one_rank: "curl -O https://quivermutationdb.org/api/bulk/qmd-n4.ndjson.gz",
      split_rank: "for f in $(curl -s .../api/bulk | jq -r '.ranks[]|select(.rank==6).parts[].file'); "
        + "do curl -O https://quivermutationdb.org/api/bulk/$f; done",
      verify: "cat qmd-n6.part*.ndjson.gz | gunzip | shasum -a 256   # compare with the rank's sha256_ndjson",
      read: "gunzip -c qmd-n4.ndjson.gz | jq -c 'select(.mutation_finite == true)'",
      resumable: "curl -C - -O <url>   # Range requests are supported",
    },
  });
});

bulkRoutes.get("/bulk/:file", async (c) => {
  const file = c.req.param("file");
  if (!FILE.test(file)) return c.json({ detail: "No such bulk file" }, 404);

  const range = c.req.header("range");
  const obj = await bucket(c.env).get(file, range ? { range: c.req.raw.headers } : undefined);
  if (!obj) return c.json({ detail: "No such bulk file" }, 404);

  const h = new Headers();
  obj.writeHttpMetadata(h);
  h.set("etag", obj.httpEtag);
  // A day, and deliberately NOT `immutable`. Corpus filenames are STABLE across
  // releases -- qmd-n4.ndjson.gz is regenerated with new contents when the
  // census grows -- so `immutable` would tell every cache to serve the old
  // bytes for a year without ever revalidating, and a reader would silently get
  // last release's data with this release's checksums. The ETag makes
  // revalidation cheap; a day of staleness after a release is the cost, and
  // releases are rare. Only version-in-the-name files may be immutable.
  h.set("Cache-Control", "public, max-age=86400");
  h.set("Accept-Ranges", "bytes");
  h.set("Content-Disposition", `attachment; filename="${file}"`);
  if (file.endsWith(".gz")) {
    // application/gzip, NOT Content-Encoding: gzip -- the user is downloading a
    // gzip FILE, and declaring it as a transfer encoding makes browsers
    // silently decompress it, so the sha256 in the manifest would never match
    // what landed on disk.
    h.set("Content-Type", "application/gzip");
  } else {
    h.set("Content-Type", "application/json; charset=utf-8");
  }

  // Gate on the REQUEST carrying a Range, not on obj.range being populated:
  // R2 fills obj.range in for a plain full get too, so keying off it returned
  // 206 Partial Content for ordinary downloads -- with no Content-Length, and
  // a status some clients treat as a broken transfer.
  if (range && obj.range && "offset" in obj.range) {
    const start = obj.range.offset ?? 0;
    const length = obj.range.length ?? (obj.size - start);
    h.set("Content-Range", `bytes ${start}-${start + length - 1}/${obj.size}`);
    return new Response(obj.body, { status: 206, headers: h });
  }
  // Set, but the runtime streams the R2 body and answers with
  // Transfer-Encoding: chunked instead, so browsers show no progress bar on a
  // multi-hundred-megabyte file. That is why the manifest and the download
  // dialog both publish per-file sizes up front.
  h.set("Content-Length", String(obj.size));
  return new Response(obj.body, { headers: h });
});
