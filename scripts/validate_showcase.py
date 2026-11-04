"""Check demo partitions, feedback, and site wiring with the Python standard library."""
from pathlib import Path
from itertools import combinations
from html.parser import HTMLParser
import json
import re

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'docs/data/showcase.json').read_text())
words = data['puzzle']['words']
gold = [set(g['words']) for g in data['puzzle']['groups']]
assert len(words) == len(set(words)) == 16
assert len(gold) == 4 and all(len(g) == 4 for g in gold)
assert set().union(*gold) == set(words)

def pairs(groups):
    return {tuple(sorted(p)) for group in groups for p in combinations(group, 2)}

def quality(groups):
    predicted, actual = pairs(groups), pairs(gold)
    return 2 * len(predicted & actual) / (len(predicted) + len(actual))

for run in data['runs']:
    seen = set()
    for state in run['steps']:
        flattened = [w for g in state['partition'] for w in g]
        assert len(flattened) == len(set(flattened)), (run['id'], state['title'], 'duplicate assignment')
        assert set(flattened) <= set(words)
        assert 0 <= quality(state['partition']) <= 1
        if 'action' in state:
            action = frozenset(state['action'])
            assert len(action) == 4 and action <= set(words)
            overlap = max(len(action & g) for g in gold)
            expected = 'REPEATED' if action in seen else 'CORRECT' if overlap == 4 else 'ONE-AWAY' if overlap == 3 else 'INCORRECT'
            assert state['feedback'] == expected, (run['id'], state['title'])
            seen.add(action)
    scores = [quality(s['partition']) for s in run['steps']]
    assert max(scores) == 1
    assert scores[-1] == 1 if run['id'] == 'success' else scores[-1] < max(scores)

class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = set(); self.refs = []; self.assets = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids, 'duplicate ID'
            self.ids.add(a['id'])
        if a.get('href', '').startswith('#') and len(a['href']) > 1:
            self.refs.append(a['href'][1:])
        if tag in ('link', 'script'):
            src = a.get('src', a.get('href', ''))
            if src and not src.startswith(('http', '#')):
                self.assets.append(src)

page = Page(); page.feed((root / 'docs/index.html').read_text())
assert all(ref in page.ids for ref in page.refs)
assert all((root / 'docs' / path).is_file() for path in page.assets)
app = (root / 'docs/app.js').read_text()
assert set(re.findall(r"\$\('([^']+)'\)", app)) <= page.ids
print(f"Validated {len(data['runs'])} runs / {sum(len(r['steps']) for r in data['runs'])} states; partitions, feedback, assets, anchors, and DOM bindings pass.")
