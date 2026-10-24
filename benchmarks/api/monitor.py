"""Sample GPU device totals and host contention; no changes to other workloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=1800)
    args=ap.parse_args()
    args.output.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    start=time.monotonic()
    with args.output.open('x') as f:
        while time.monotonic()-start<args.seconds:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks_event_reasons.active,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            stat=Path('/proc/stat').read_text().splitlines()[0].split()[1:]
            processes=subprocess.check_output(['ps','-eo','pid,pcpu,nlwp,rss,comm','--sort=-pcpu'],text=True).splitlines()[1:16]
            f.write(json.dumps(dict(monotonic=time.monotonic(),elapsed=time.monotonic()-start,utc=time.time(),
                gpu=dict(zip(['memory_mib','temperature_c','utilization_percent','power_w','sm_mhz','clock_reasons','sw_thermal_slowdown','hw_thermal_slowdown'],gpu)),
                load_average=os.getloadavg(),cpu_ticks=list(map(int,stat)),top=processes))+'\n')
            f.flush()
            time.sleep(1)


if __name__=='__main__':main()
