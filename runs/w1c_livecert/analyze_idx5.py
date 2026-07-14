#!/usr/bin/env python3
import json, re
DONE = re.compile(
    r"req=(chatcmpl-\S+) done: model_forwards=(\d+) forced_token_count=(\d+) "
    r"value_tokens=(\d+) projected_value_tokens_exact=(\d+) generated_tokens=(\d+) "
    r"stop_reason=(\S+)(?:.*?w1\[on=(\w+) spans=(\d+) toks=(\d+) vfwd=(\d+) rej=(\d+)\])?")
rows = []
for ln in open('runs/w1c_livecert/server_on.log', errors='replace'):
    m = DONE.search(ln)
    if not m or "_warmup" in m.group(1):
        continue
    d = dict(id=m.group(1), fwd=int(m.group(2)), gen=int(m.group(6)), proj=int(m.group(5)))
    if m.group(8):
        d.update(spans=int(m.group(9)), toks=int(m.group(10)),
                 vfwd=int(m.group(11)), rej=int(m.group(12)))
    rows.append(d)
target = 'chatcmpl-99ff89c639ec2830'
for i, d in enumerate(rows):
    if d['id'].startswith(target):
        prev = rows[i - 1] if i > 0 else dict(spans=0, toks=0, vfwd=0, rej=0)
        print('idx5 corrupted request:', d['id'])
        print('  spans_committed=%d toks_committed=%d verify_fwd=%d rejects=%d' % (
            d['spans'] - prev.get('spans', 0), d['toks'] - prev.get('toks', 0),
            d['vfwd'] - prev.get('vfwd', 0), d['rej'] - prev.get('rej', 0)))
        print('  model_forwards=%d generated=%d proj_exact_tripwire=%d' % (
            d['fwd'], d['gen'], d['proj']))
        break
