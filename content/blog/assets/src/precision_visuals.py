# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Build precision-v-{task,dose,length}-{light,dark}.svg offline.

Run: python3 content/blog/assets/src/precision_visuals.py
Uses the same SVG primitives and palettes as heads_visuals.py. All canvases
are 420 px wide. Only the task example is invented. Dose data: verified N17
fact sheet, thinking off. Length lists: copied exactly from the retired
length generator, with its incorrect whisker description corrected below.
"""
import sys

sys.dont_write_bytecode = True

from attention_visuals import Figure, sym
from luna_v_visuals import ASSETS, PALETTES

# N17, thinking off, 4,000 tokens. Q2_K fails the abstention validity gate.
DOSE = [('BF16', .322), ('Q8_0', .336), ('Q6_K', .322),
        ('Q5_K_M', .327), ('Q4_K_M', .292), ('Q3_K_M', .246),
        ('Q3_K_S', .108), ('Q2_K', .009)]

# N5, different authored one-fact items, one draw per cell.
ONE_FACT_TASK = [(1000, 1.000), (4000, 1.000), (8000, 1.000),
                 (16000, 0.900), (24000, 0.757)]
# N20, thinking on: material tokens, defined-only rate, strict, permissive.
# Bounds count undefined responses as misses/hits, not variation over draws.
TWO_FACT_JOIN = [(4096, 0.538, 0.523, 0.550),
                 (7558, 0.446, 0.436, 0.459),
                 (15360, 0.306, 0.298, 0.325),
                 (23161, 0.066, 0.064, 0.094)]
ONE_HOP_CONTROL = [(4096, 0.7207, 0.7018, 0.7281),
                   (7558, 0.5215, 0.4971, 0.5439),
                   (15360, 0.2692, 0.2661, 0.2778),
                   (23161, 0.0917, 0.0906, 0.1023)]
CHANCE_FLOOR = 0.1619


class PrecisionFigure(Figure):
    def save(self):
        content = '\n'.join(self.parts + ['</svg>']) + '\n'
        assert '\u2013' not in content and '\u2014' not in content
        (ASSETS / f'precision-v-{self.name}-{self.theme}.svg').write_text(
            content, encoding='utf-8')


def mark(f, x, y, color, shape='circle'):
    if shape == 'square':
        f.box(x - 3, y - 3, 6, 6, color, radius=0)
    elif shape == 'triangle':
        f.parts.append(f'<path d="M{x} {y-4} L{x+4} {y+3} L{x-4} {y+3} Z" '
                       f'fill="{f.col(color)}"/>')
    elif shape == 'cross':
        f.line(x-4, y-4, x+4, y+4, color=color, width=2)
        f.line(x-4, y+4, x+4, y-4, color=color, width=2)
    else:
        f.parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{f.col(color)}"/>')


def task(theme):
    f = PrecisionFigure(theme, 'task', 364,
        'Invented example. Fact one links Lira Venn to sketching Saturn. Fact '
        'two links Lira Venn to signing letters as Sorel Ash. Astronomy '
        'enthusiast identifies Lira; alias asks for Sorel. The question shares '
        'no words with the facts. The control stops after the first connection.')
    for y, label, lines, color in [
        (12, 'Fact one', ('Lira Venn keeps sketching Saturn.',), 'teal'),
        (83, 'Fact two', ('Lira Venn signs letters as Sorel Ash.',), 'blue')]:
        f.box(18, y, 384, 61, color + '_bg', color)
        f.text(32, y+20, label, 12, color)
        f.text(32, y+44, lines[0], 15, color)
    f.text(210, 174, 'Which alias belongs to an astronomy enthusiast?', 14,
           'ink', 'middle')
    for x, w, label, color in [(18, 114, 'Astronomy', 'muted'),
                               (153, 114, 'Lira Venn', 'teal'),
                               (288, 114, 'Sorel Ash', 'blue')]:
        f.box(x, 242, w, 40, color + '_bg' if color != 'muted' else 'panel',
              color if color != 'muted' else 'line')
        f.text(x+w/2, 267, label, 15, color, 'middle')
    f.arrow(133, 262, 151, 262, 'teal')
    f.arrow(268, 262, 286, 262, 'blue')
    f.text(140, 208, 'Saturn', 13, 'teal', 'middle')
    f.text(277, 230, 'signs letters as', 13, 'blue', 'middle')
    f.line(140, 213, 140, 248, color='teal', width=1)
    f.line(277, 235, 277, 248, color='blue', width=1)
    f.text(210, 306, 'Control stops here', 12, 'teal', 'middle')
    f.text(345, 326, 'Join reaches here', 12, 'blue', 'middle')
    f.text(210, 352, 'Invented example · no shared words with the question',
           12, 'muted', 'middle')
    f.save()


def dose(theme):
    f = PrecisionFigure(theme, 'dose', 352,
        'Thinking off, Qwen3-8B, 4,000-token context. Strict scores: ' +
        ', '.join(f'{rung} {score:.3f}' for rung, score in DOSE) +
        '. Q3_K_M at about 3.4 bits per weight has no detectable paired drop. '
        'First damage at Q3_K_S, about 3.2. Q2_K is invalid: abstention 0.766.')
    x = lambda i: 56 + i * 48
    y = lambda score: 238 - score / .4 * 190
    f.text(18, 23, 'Strict score', 13, 'muted')
    f.box(372, 40, 38, 204, 'amber_bg', radius=3)
    for value in (0, .1, .2, .3, .4):
        f.line(45, y(value), 405, y(value), color='line', width=1)
        f.text(36, y(value)+4, f'{value:.1f}', 11, 'muted', 'end')
    for i in range(6):
        f.line(x(i), y(DOSE[i][1]), x(i+1), y(DOSE[i+1][1]), color='blue', width=2)
    # The invalid point is deliberately disconnected from the valid series.
    for i, (rung, score) in enumerate(DOSE):
        color = 'amber' if i >= 6 else 'blue'
        mark(f, x(i), y(score), color, 'cross' if i == 7 else 'circle')
        f.text(x(i), y(score) + (18 if i == 6 else -12),
               f'{score:.3f}', 11, color, 'middle')
        f.text(x(i), 263, rung, 10.5, 'ink', 'middle')
    f.text(210, 289, 'Precision rungs · less precision →', 12, 'muted', 'middle')
    f.text(210, 315, '× Q2_K invalid: 0.766 abstention', 13, 'amber', 'middle')
    f.text(210, 340, 'Qwen3-8B · thinking off · 4,000 tokens', 12, 'muted', 'middle')
    f.save()


def length(theme):
    f = PrecisionFigure(theme, 'length', 430,
        'BF16, thinking on. Older one-fact series uses a different item set '
        'and one draw per cell. The join and control use nine draws per item. '
        'Whiskers are strict to permissive ranges, not variation across draws. '
        'Both join and control point rates cross the 0.1619 chance benchmark between '
        '15,360 and 23,161 material tokens. Data: ' + repr(
            (ONE_FACT_TASK, TWO_FACT_JOIN, ONE_HOP_CONTROL)))
    x = lambda tokens: 48 + tokens / 24000 * 345
    y = lambda score: 321 - score * 214
    for yy, label, color, shape, dashed in [
        (19, 'One fact · different items, one draw', 'blue', 'circle', False),
        (41, 'Two-fact join', 'amber', 'square', True),
        (63, 'One-connection control', 'teal', 'triangle', True)]:
        f.line(20, yy-4, 45, yy-4, color=color, width=2, dashed=dashed)
        mark(f, 32, yy-4, color, shape)
        f.text(55, yy, label, 13, 'ink')
    f.text(18, 90, 'Score', 12, 'muted')
    for value in (0, .25, .5, .75, 1):
        f.line(48, y(value), 397, y(value), color='line', width=1)
        f.text(39, y(value)+4, f'{value:g}', 11, 'muted', 'end')
    f.line(48, y(CHANCE_FLOOR), 397, y(CHANCE_FLOOR), color='muted', width=1, dashed=True)
    f.text(56, y(CHANCE_FLOOR)-9, 'Chance floor 0.1619', 12, 'muted')
    for points, color, shape, dashed in [
        (ONE_FACT_TASK, 'blue', 'circle', False),
        (TWO_FACT_JOIN, 'amber', 'square', True),
        (ONE_HOP_CONTROL, 'teal', 'triangle', True)]:
        f.path(' '.join(f'{"M" if i == 0 else "L"}{x(p[0])} {y(p[1])}'
                       for i, p in enumerate(points)), color, 2, dashed)
        for p in points:
            if len(p) == 4:
                xx, lo, hi = x(p[0]), y(p[2]), y(p[3])
                f.line(xx, lo, xx, hi, color=color, width=1.5)
                for yy in (lo, hi):
                    f.line(xx-5, yy, xx+5, yy, color=color, width=1.5)
            mark(f, x(p[0]), y(p[1]), color, shape)
    for tokens in (0, 8000, 16000, 24000):
        f.text(x(tokens), 341, f'{tokens//1000}K' if tokens else '0', 12, 'muted', 'middle')
    f.text(222, 365, 'Haystack material (tokens)', 13, 'muted', 'middle')
    # This is a legend key for the whiskers, separate from the regime caption.
    f.line(24, 386, 44, 386, color='muted', width=1.5)
    for xx in (24, 44):
        f.line(xx, 382, xx, 390, color='muted', width=1.5)
    f.text(55, 390, 'Strict to permissive range', 12, 'muted')
    f.text(210, 418, 'Qwen3-8B · BF16 · thinking on', 12, 'muted', 'middle')
    f.save()


if __name__ == '__main__':
    for theme in PALETTES:
        for draw in (task, dose, length):
            draw(theme)
