"""Wrap a uvicorn server in cProfile and dump pstats on shutdown.

Useful for getting a coarse breakdown of per-request CPU after a bench run.

Usage:
    python -m bench.profile_server pxt --port 8000
    python -m bench.profile_server pg  --port 8001

Drive load against the port (e.g. python -m bench.drive ...), then Ctrl-C the
server. The pstats file is written to bench/profile-<label>.pstats.

Inspect with:
    python -c "import pstats; pstats.Stats('bench/profile-pxt.pstats').sort_stats('cumulative').print_stats(40)"
or open with snakeviz:
    snakeviz bench/profile-pxt.pstats
"""

import argparse
import cProfile
import signal
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('label', choices=['pxt', 'pg'])
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    if args.label == 'pg':
        from bench.pg_serve import app
    else:
        from pixeltable.serving import FastAPIRouter
        from bench.queries import q1

        app = FastAPIRouter()
        app.add_query_route(path='/q1', query=q1, inputs=['i'], one_row=True, method='get')

    out = Path(__file__).with_name(f'profile-{args.label}.pstats')
    profiler = cProfile.Profile()

    def stop(signum: int, frame: object) -> None:
        profiler.disable()
        profiler.dump_stats(str(out))
        print(f'wrote {out}', file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    profiler.enable()
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning', workers=1)


if __name__ == '__main__':
    main()
