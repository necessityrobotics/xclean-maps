# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""`xclean-maps view <map> --browser`: serve one built map to the web browser of this PC.

The page (web/index.html) uses three.js and Spark (both MIT, vendored in web/vendor) and reads the map's own files:
splats.ply, cameras.json and report.json. The server binds 127.0.0.1 only and serves only:
  /_viewer/...  the packaged page and libraries
  /splats.ply, /cameras.json, /report.json  from the map folder
Everything else is 404. No network access is needed after installation.
"""
from __future__ import annotations

import functools
import http.server
import os
import socketserver
import sys
import webbrowser

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MAP_FILES = ("splats.ply", "cameras.json", "report.json")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".json": "application/json",
         ".ply": "application/octet-stream"}


class _Handler(http.server.BaseHTTPRequestHandler):
    map_dir = None

    def log_message(self, *a):            # quiet
        pass

    def _resolve(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/", "/index.html"):
            return os.path.join(WEB, "index.html")
        if path.startswith("/_viewer/"):
            rel = os.path.normpath(path[len("/_viewer/"):]).replace("\\", "/")
            if rel.startswith("..") or os.path.isabs(rel):
                return None
            return os.path.join(WEB, rel)
        name = path.lstrip("/")
        if name in MAP_FILES:
            return os.path.join(self.map_dir, name)
        return None

    def do_GET(self):
        p = self._resolve()
        if not p or not os.path.isfile(p):
            self.send_error(404)
            return
        size = os.path.getsize(p)
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(os.path.splitext(p)[1].lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                try:
                    self.wfile.write(b)
                except (BrokenPipeError, ConnectionResetError):
                    return


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(map_dir, port=0, open_browser=True):
    md = os.path.abspath(map_dir)
    missing = [f for f in MAP_FILES if not os.path.isfile(os.path.join(md, f))]
    if missing:
        print("xclean-maps: %s is not a finished map folder (missing %s). Build it first: xclean-maps build ..."
              % (md, ", ".join(missing)), file=sys.stderr)
        return 2
    handler = functools.partial(_Handler)
    _Handler.map_dir = md
    with _Server(("127.0.0.1", port), handler) as srv:
        url = "http://127.0.0.1:%d/" % srv.server_address[1]
        size_mb = os.path.getsize(os.path.join(md, "splats.ply")) / 1e6
        print("xclean-maps: map %s (%.0f MB) at %s" % (md, size_mb, url))
        print("xclean-maps: open that address in a browser (Chrome, Edge or Firefox). Ctrl+C stops the server.")
        if open_browser:
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nxclean-maps: stopped")
    return 0
