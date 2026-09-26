# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Build the reasoning-trace figures with system Python, offline.

Run: /usr/bin/python3 content/blog/assets/src/trace_visuals.py
Writes only trace-v-{name}-{light,dark}.svg, each 420 CSS pixels wide.
Stories and filler spans are schematic. Verdict bars share a linear zero-to-one
scale and use only the author's supplied results over 12 filtered pairs.
"""

import sys

sys.dont_write_bytecode = True

from attention_visuals import Figure
from luna_v_visuals import ASSETS, PALETTES


class TraceFigure(Figure):
    def save(self):
        content = '\n'.join(self.parts + ['</svg>']) + '\n'
        assert '\u2013' not in content and '\u2014' not in content
        (ASSETS / f'trace-v-{self.name}-{self.theme}.svg').write_text(
            content, encoding='utf-8')


def stories(theme):
    f = TraceFigure(theme, 'stories', 336,
        'Two hypotheses, drawn schematically. Working memory: facts feed a '
        'written note, which is read back for the answer. Narration: extra '
        'processing steps lead to the answer and emit a written report; '
        'the report is not the necessary route to the answer.')
    for left, label, color in ((16, 'Working memory', 'blue'),
                                (218, 'Narration', 'teal')):
        center = left + 93
        f.box(left, 12, 186, 294, 'panel', 'line')
        f.text(center, 37, label, 16, color, 'middle', True)
        f.box(left + 23, 55, 140, 34, color + '_bg', color)
        f.text(center, 77, 'Facts', 14, color, 'middle')
        f.box(left + 43, 259, 100, 33, color + '_bg', color)
        f.text(center, 281, 'Answer', 14, color, 'middle', True)

    f.arrow(109, 95, 109, 137, 'blue', 2)
    f.box(45, 144, 128, 56, 'blue_bg', 'blue')
    f.text(109, 166, 'Written note', 14, 'blue', 'middle')
    f.line(67, 181, 151, 181, color='blue')
    f.line(67, 188, 134, 188, color='blue')
    f.arrow(109, 205, 109, 252, 'blue', 2)
    f.text(120, 233, 'read back', 12, 'blue')

    f.path('M244 89 L244 119', 'teal', 2)
    f.text(321, 111, 'Extra steps', 12, 'teal', 'middle')
    for x in (244, 310, 376):
        f.box(x - 10, 125, 20, 20, 'teal_bg', 'teal', radius=4)
    f.arrow(257, 135, 296, 135, 'teal', 2)
    f.arrow(323, 135, 362, 135, 'teal', 2)
    f.arrow(310, 151, 310, 173, 'muted')
    f.box(251, 181, 118, 46, 'bg', 'line')
    f.text(310, 200, 'Written', 13, 'muted', 'middle')
    f.text(310, 217, 'report', 13, 'muted', 'middle')
    f.path('M376 151 L376 243 L311 243', 'teal', 2)
    f.arrow(311, 243, 311, 253, 'teal', 2)
    f.text(210, 327, 'Competing stories · schematic paths', 12, 'muted', 'middle')
    f.save()


def filler(theme):
    f = TraceFigure(theme, 'filler', 320,
        'Schematic trace spans. The full trace has an earlier span, a derivation '
        'and a later span. Cutting the derivation removes content and shortens '
        'the trace. Replacing it with filler removes that content but preserves '
        'the token count and the later span’s position.')

    def span(x, y, width, label, color='muted'):
        fill = 'panel' if color == 'muted' else color + '_bg'
        f.box(x, y, width, 34, fill, 'line' if color == 'muted' else color, radius=4)
        f.text(x + width / 2, y + 22, label, 13, color, 'middle')

    for y, label in ((26, 'Full trace'), (125, 'Cut'), (224, 'Filler version')):
        f.text(18, y, label, 15, bold=True)
        span(18, y + 12, 100, 'Earlier')
    span(126, 38, 166, 'Derivation', 'blue')
    span(300, 38, 100, 'Later')
    span(126, 137, 100, 'Later')
    f.text(245, 159, 'shorter', 13, 'amber')
    f.text(18, 193, 'Content removed · fewer steps', 13, 'amber')
    span(126, 236, 166, 'Filler', 'amber')
    span(300, 236, 100, 'Later')
    f.text(18, 292, 'Content removed · same token count', 13, 'amber')
    f.text(210, 314, 'Schematic spans · widths are illustrative', 11, 'muted', 'middle')
    f.save()


def verdict(theme):
    f = TraceFigure(theme, 'verdict', 355,
        'Over the same 12 filtered pairs: masking every derivation restatement '
        'leaves correctness at 0.83; a random-sentence mask leaves 1.00. '
        'Corrupting every derivation restatement is followed on 0.92; corrupting '
        'an unrelated placeholder is followed on 0.00. All bars run from zero '
        'to one. The written derivation is usually not needed, but trusted.')
    for top, heading, metric, color, rows in (
        (12, 'Not needed on most items', 'Correct answer', 'teal',
         (('Derivation masked', .83), ('Random sentences masked', 1.00))),
        (176, 'But trusted when wrong', 'Wrong value followed', 'amber',
         (('Derivation corrupted', .92), ('Unrelated placeholder', .00))),
    ):
        f.box(16, top, 388, 154, 'panel', 'line')
        f.text(30, top + 25, heading, 16, color, bold=True)
        f.text(30, top + 45, metric, 12, 'muted')
        for i, (label, value) in enumerate(rows):
            y = top + 66 + i * 46
            f.text(30, y, label, 13)
            f.box(30, y + 9, 302, 14, 'faint', radius=2)
            if value:
                f.box(30, y + 9, 302 * value, 14, color, radius=2)
            else:
                f.line(30, y + 9, 30, y + 23, color=color, width=3)
            f.text(388, y + 22, f'{value:.2f}', 16, color, 'end', True)
    f.text(210, 348, '12 filtered pairs · all bars share a 0 to 1 scale', 12, 'muted', 'middle')
    f.save()


if __name__ == '__main__':
    for theme in PALETTES:
        for draw in (stories, filler, verdict):
            draw(theme)
