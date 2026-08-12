#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Entry point for running Magpie MCP server as a module.

Usage:
    python -m Magpie.mcp                              # stdio (for MCP clients)
    python -m Magpie.mcp --transport streamable-http   # HTTP (for remote servers)
    python -m Magpie.mcp --transport sse               # SSE over HTTP

For HTTP/SSE transports, bind address can be set with --host / --port (applied
before the server module loads) or with environment variables:

    MAGPIE_HOST  - Bind host (default: 0.0.0.0)
    MAGPIE_PORT  - Bind port (default: 8000)
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Magpie MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Bind host for HTTP transports (overrides MAGPIE_HOST; default 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port for HTTP transports (overrides MAGPIE_PORT; default 8000)",
    )
    args = parser.parse_args()

    # FastMCP reads host/port at import time in server.py; set env before import.
    if args.host is not None:
        os.environ["MAGPIE_HOST"] = args.host
    if args.port is not None:
        os.environ["MAGPIE_PORT"] = str(args.port)

    from .server import mcp

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
