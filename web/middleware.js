// Login gate for the whole site: every request needs the shared username and password
// (HTTP Basic Auth). Set them in Vercel, not here:
//   vercel env add SITE_USER production
//   vercel env add SITE_PASSWORD production
// With either one missing, nobody gets in.
import { createHash, timingSafeEqual } from "node:crypto";

export const config = { runtime: "nodejs" };

const digest = (s) => createHash("sha256").update(s).digest();
const same = (a, b) => timingSafeEqual(digest(a), digest(b));

export default function middleware(request) {
  const user = process.env.SITE_USER, pass = process.env.SITE_PASSWORD;
  const [scheme, encoded] = (request.headers.get("authorization") || "").split(" ");
  if (user && pass && scheme === "Basic" && encoded) {
    const given = Buffer.from(encoded, "base64").toString("utf8");
    const at = given.indexOf(":");
    if (at > 0 && same(given.slice(0, at), user) & same(given.slice(at + 1), pass)) {
      return new Response(null, { headers: { "x-middleware-next": "1" } }); // let the request through
    }
  }
  return new Response("Sign in to see Signify.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Signify", charset="UTF-8"', "Cache-Control": "no-store" },
  });
}
