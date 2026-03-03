import numpy as np
import pandas as pd
import re
import xml.sax.saxutils as saxutils

def write_sumo_routes_xml(route_paths: dict[str, list[str]], route_groups: pd.DataFrame, out_path: str, dep_bin_minutes: float) -> None:
    # Bin label -> seconds (best-effort). If dep_bin is already numeric seconds, use it.
    def bin_to_sec(b):

        # Numeric seconds (including numpy scalars)
        if isinstance(b, (int, float, np.integer, np.floating)):
            return float(b)

        b = str(b).strip()

        # hh:mm:ss format
        if ":" in b:
            parts = b.split(":")
            if len(parts) != 3:
                raise ValueError(f"Invalid hh:mm:ss time: {b}")

            h, m, s = map(float, parts)
            return 3600*h + 60*m + s

        # "5min" format
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*min\s*$", b)
        if m:
            return float(m.group(1)) * 60.0

        raise ValueError(f"Cannot parse departure bin: {b}")

    with open(out_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<routes>\n")

        # Define routes
        for rid, edges in route_paths.items():
            edges_str = " ".join(saxutils.escape(str(e)) for e in edges)
            f.write(f'  <route id="{rid}" edges="{edges_str}"/>\n')

        # Emit flows grouped by dep_bin and vtype
        for row in route_groups.itertuples(index=False):
            rid = row.route_id
            count = int(row.count)
            vtype = saxutils.escape(str(row.vtype_key))
            depart = bin_to_sec(row.dep_bin)
            # spread uniformly over the bin window
            begin = max(0.0, depart)
            end = begin + float(dep_bin_minutes) * 60.0
            f.write(
                f'  <flow id="f_{rid}_{vtype}_{int(begin)}" route="{rid}" type="{vtype}" '
                f'begin="{begin:.1f}" end="{end:.1f}" number="{count}"/>\n'
            )

        f.write("</routes>\n")
