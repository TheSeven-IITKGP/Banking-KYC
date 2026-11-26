export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith("/api/")) {
      if (request.method === "OPTIONS") {
        return new Response(null, {
          status: 204,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
          },
        });
      }
      const origin = (env.API_ORIGIN || "").replace(/\/$/, "");
      if (!origin) {
        return new Response(
          JSON.stringify({
            error: "API_ORIGIN is not configured in wrangler.toml or Worker secrets.",
            hint: "Start the Python backend (`make serve`) behind cloudflared or a GPU origin and set API_ORIGIN.",
          }),
          { status: 503, headers: { "Content-Type": "application/json" } }
        );
      }
      const targetUrl = `${origin}${url.pathname}${url.search}`;
      const proxied = new Request(targetUrl, request);
      return fetch(proxied);
    }
    return env.ASSETS.fetch(request);
  },
};
