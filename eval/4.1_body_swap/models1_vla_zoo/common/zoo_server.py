"""vla_zoo policy server (2026-11-03). Same websocket + msgpack protocol as openpi's WebsocketPolicyServer
(first frame = metadata dict; then request obs dict -> reply dict with "actions"), so the oftpi-env client
(zoo_client.py, openpi_client.WebsocketClientPolicy) talks to it unchanged.

Usage: python zoo_server.py --adapter <module path .py> --port P [--adapter-args k=v ...]
The adapter module must define  `class Adapter: __init__(**kw); metadata -> dict; infer(raw_obs) -> np.ndarray (N, 7)`
with actions in LIBERO env convention (delta pos, delta axis-angle, gripper -1 open / +1 close).
Requires only numpy, msgpack, websockets (>= 13, asyncio API) in the model's own env.
"""
import argparse
import asyncio
import functools
import http
import importlib.util
import logging
import time
import traceback

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames


# --- copy of openpi_client/msgpack_numpy.py (Apache-2.0) ---
def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def load_adapter(path, kw):
    spec = importlib.util.spec_from_file_location("zoo_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Adapter(**kw)


class Server:
    def __init__(self, adapter, port):
        self.adapter, self.port = adapter, port
        self.n = 0

    async def handler(self, ws):
        logging.info(f"connection {ws.remote_address}")
        packer = Packer()
        await ws.send(packer.pack(dict(self.adapter.metadata)))
        while True:
            try:
                obs = unpackb(await ws.recv())
                t = time.monotonic()
                act = np.asarray(self.adapter.infer(obs), np.float64)
                assert act.ndim == 2 and act.shape[1] == 7, act.shape
                self.n += 1
                await ws.send(packer.pack({"actions": act, "server_timing": {"infer_ms": (time.monotonic() - t) * 1000}}))
            except websockets.ConnectionClosed:
                logging.info(f"closed {ws.remote_address}")
                break
            except Exception:
                tb = traceback.format_exc()
                logging.error(tb)
                await ws.send(tb)
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="server error")
                raise

    async def run(self):
        def health(conn, req):
            if req.path == "/healthz":
                return conn.respond(http.HTTPStatus.OK, "OK\n")
            return None
        async with _server.serve(self.handler, "0.0.0.0", self.port, compression=None, max_size=None,
                                 process_request=health, ping_interval=None) as srv:
            logging.info(f"listening on {self.port}")
            await srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--adapter-args", nargs="*", default=[])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kw = dict(x.split("=", 1) for x in a.adapter_args)
    ad = load_adapter(a.adapter, kw)
    logging.info(f"adapter metadata: {ad.metadata}")
    asyncio.run(Server(ad, a.port).run())
