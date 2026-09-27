# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Build the sequel's toolbox recap, offline, in both themes.

Run: python3 content/blog/assets/src/comprehension_visuals.py
Writes only comprehension-v-toolbox-{light,dark}.svg, at 420 CSS pixels.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_visuals import Figure, PALETTES, ASSETS


def toolbox(theme):
    f = Figure(theme, 'toolbox', 342,
        'ContextCite locates the source the answer rests on: yes. LOCOS head list '
        'locates heads needed to carry the fact: yes. Attention knockout tests '
        'whether the answer needs the read: yes. LOCOS write score judges '
        'answer quality: no, clean answers scored lower.')
    rows = [('ContextCite', 'Locate the source?', 'Yes', 'teal'),
            ('LOCOS head list', 'Locate the heads needed?', 'Yes', 'blue'),
            ('Attention knockout', 'Test whether the read is needed?', 'Yes', 'teal'),
            ('LOCOS write score', 'Judge answer quality?', 'No', 'amber')]
    for i, (tool, question, verdict, color) in enumerate(rows):
        y = 12 + i * 80
        f.box(16, y, 388, 68, 'panel', 'line')
        f.text(30, y + 24, tool, 15, color, bold=True)
        f.text(30, y + 49, question, 14)
        f.box(347, y + 15, 44, 37, color + '_bg', radius=5)
        f.text(369, y + 39, verdict, 17, color, 'middle', True)
    content = '\n'.join(f.parts + ['</svg>']) + '\n'
    assert '\u2013' not in content and '\u2014' not in content
    (ASSETS / f'comprehension-v-toolbox-{theme}.svg').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    for theme in PALETTES:
        toolbox(theme)
