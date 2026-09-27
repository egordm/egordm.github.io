# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Build paired detector diagrams with system Python, offline.

Run: /usr/bin/python3 content/blog/assets/src/heads_visuals.py
Writes only heads-v-{name}-{light,dark}.svg in ASSETS. The example and measurements are supplied by the author.
The ablation, quality, direction and knockout functions serve the sequel.
Use --sequel to render only those figures. The current post introduces positions and heads, walks through
wu, qrhead, locos-write and locos-contrast, then closes with verdicts and real.
The default run only writes the current post's figures.
Text remains selectable. Every canvas is 420 CSS pixels wide.
"""

import sys

sys.dont_write_bytecode = True

from luna_v_visuals import PALETTES, ASSETS
from attention_visuals import Figure, sym
import json


# Sequel figures share the approved post's drawing primitives and palette.
def ablation(theme):
    f = DetectorFigure(theme, 'ablation', 405,
        'Qwen3-8B NoLiMa ROUGE-L. Published and reproduction: no heads off '
        '0.401 and 0.412; LOCOS top 50 off 0.000 and 0.000; Wu top 50 off '
        '0.292 and 0.342; random 50 off 0.391 in the reproduction only.')
    for x, color, label in [(18, 'blue', 'Published'), (175, 'teal', 'My reproduction')]:
        f.box(x, 14, 10, 10, color, radius=2)
        f.text(x + 17, 24, label, 13, 'muted')
    rows = [('No heads off', .401, .412), ('LOCOS: top 50 off', .000, .000),
            ('Wu: top 50 off', .292, .342), ('Random: 50 off', None, .391)]
    for i, (label, pub, repro) in enumerate(rows):
        y = 56 + i * 80
        f.text(18, y, label, 15, 'ink', bold=True)
        for j, (value, color) in enumerate([(pub, 'blue'), (repro, 'teal')]):
            yy = y + 10 + 23 * j
            if value is None:
                f.text(28, yy + 13, 'Not reported', 12, 'muted')
                continue
            w = value * 730
            if value:
                f.box(18, yy, w, 16, color, radius=2)
            else:
                f.line(18, yy, 18, yy + 16, color=color, width=3)
            f.text(28 + w, yy + 13, f'{value:.3f}', 14, color, bold=True)
    f.line(18, 366, 402, 366, color='line')
    f.text(210, 389, 'Qwen3-8B · NoLiMa · answer score (ROUGE-L)', 12, 'muted', 'middle')
    f.save()


def quality(theme):
    f = DetectorFigure(theme, 'quality', 330,
        'Clean minus flawed write score. All heads: -0.035, 95% interval '
        '-0.052 to -0.020. Top 50: -0.093, interval -0.165 to -0.041. '
        'Both intervals lie below zero, where flawed answers score higher.')
    f.text(210, 25, 'Write score: clean minus flawed', 13, 'muted', 'middle')
    def x(value):
        return 18 + (value + .18) / .24 * 384
    zero = x(0)
    f.box(18, 48, zero - 18, 212, 'amber_bg', radius=0)
    f.box(zero, 48, 402 - zero, 212, 'teal_bg', radius=0)
    f.text(30, 68, 'Flawed higher', 12, 'amber')
    f.text(354, 68, 'Clean higher', 12, 'teal', 'middle')
    f.line(zero, 79, zero, 260, color='muted', width=1, dashed=True)
    f.text(zero, 279, '0', 13, 'muted', 'middle')
    for y, label, mid, low, high, color in [(106, 'All heads', -.035, -.052, -.020, 'blue'),
                                         (198, 'Top 50 heads', -.093, -.165, -.041, 'amber')]:
        f.text(30, y, label, 14, color, bold=True)
        yy = y + 22
        f.line(x(low), yy, x(high), yy, color=color, width=3)
        for value in (low, high):
            f.line(x(value), yy - 7, x(value), yy + 7, color=color, width=2)
        f.parts.append(f'<circle cx="{x(mid)}" cy="{yy}" r="5" fill="{f.col(color)}"/>')
        f.text(30, y + 48, f'{mid:.3f}  [{low:.3f}, {high:.3f}]', 13, color)
    f.text(210, 313, 'Qwen3-8B · estimates and 95% intervals', 12, 'muted', 'middle')
    f.save()


def direction(theme):
    f = DetectorFigure(theme, 'direction', 340,
        'Schematic writes from the half-a-minute fact at the answer position. '
        'Each panel aligns its horizontal axis with the token being scored, '
        '30 or the mistaken 60. The projection measures support, not correctness. '
        'Arrow lengths are illustrative, not measured scores.')
    f.box(40, 12, 340, 36, 'teal_bg', 'teal')
    f.text(210, 36, 'The timeout is half a minute.', 16, 'teal', 'middle')
    f.text(210, 73, 'Write at the answer position', 14, 'muted', 'middle')
    for left, token, color in [(16, '30', 'blue'), (218, '60', 'amber')]:
        f.box(left, 91, 186, 211, 'panel', 'line')
        f.text(left + 93, 118, f'Scoring “{token}”', 16, color, 'middle', True)
        x0, y0 = left + 28, 226
        f.arrow(x0, y0, left + 166, y0, 'muted', 1)
        f.arrow(x0, y0, x0, 140, 'muted', 1)
        f.arrow(x0, y0, x0 + 93, y0 - 55, color, 3)
        f.text(x0 + 43, 158, 'write', 13, color)
        f.line(x0 + 93, y0 - 55, x0 + 93, y0, color=color, width=1, dashed=True)
        f.arrow(x0, y0, x0 + 93, y0, color, 4)
        f.math(left + 158, 251, sym('u', token), 18, color)
        f.text(left + 93, 283, f'Strong push for “{token}”', 13, color, 'middle')
    f.text(210, 327, 'Chosen token sets the axis · schematic writes', 12, 'muted', 'middle')
    f.save()


def knockout(theme):
    f = DetectorFigure(theme, 'knockout', 397,
        'Timeout example illustrates blocked attention from the answer position '
        'to fact tokens throughout a schematic layer stack. On Qwen2.5-Coder-7B, '
        'all-layer knockout gave 0.00 correctness for copy and inference versions. '
        'A separate five-layer sweep placed the loss in layers 19 to 27.')
    f.text(135, 26, 'Fact tokens', 14, 'teal', 'middle', True)
    f.text(337, 26, 'Answer position', 14, 'blue', 'middle', True)
    for x, word in [(73, 'half'), (135, 'a'), (197, 'minute')]:
        f.text(x, 264, word, 13, 'teal', 'middle')
        f.arrow(x, 239, x, 50, 'line', 1)
    f.arrow(337, 239, 337, 50, 'line', 1)
    for y in (72, 145, 218):
        f.box(50, y - 28, 170, 56, 'teal_bg', 'line')
        for x in (73, 135, 197):
            f.vector(x - 10, y - 22, 'teal', w=20)
        f.arrow(322, y, 229, y, 'amber', 1.5)
        f.vector(327, y - 22, 'blue', w=20)
        f.line(261, y - 11, 279, y + 11, color='amber', width=3)
        f.line(279, y - 11, 261, y + 11, color='amber', width=3)
    f.text(270, 112, '···', 18, 'muted', 'middle')
    f.text(270, 186, '···', 18, 'muted', 'middle')
    f.box(18, 283, 384, 67, 'amber_bg')
    f.text(30, 306, 'Blocked at every layer', 15, 'amber', bold=True)
    f.text(30, 331, 'Copy and inference: correctness', 13, 'muted')
    f.text(386, 334, '0.00', 24, 'amber', 'end', True)
    f.text(210, 372, 'Five-layer sweep: loss in layers 19 to 27', 13, 'muted', 'middle')
    f.text(210, 389, 'Qwen2.5-Coder-7B · schematic stack', 11, 'muted', 'middle')
    f.save()


# Current post: the same layout and palette as the attention post.
HEADS = {
    'A': dict(real=(.10, .05, .70, .05, .10), na=(.60, .10, .10, .10, .10), write=(.1, 1.0)),
    'B': dict(real=(.40, .10, .30, .10, .10), na=(.60, .10, .15, .05, .10), write=(3.0, .2)),
    'C': dict(real=(.80, .05, .05, .05, .05), na=(.80, .05, .05, .05, .05), write=(.2, .1)),
}


class DetectorFigure(Figure):
    def save(self):
        content = '\n'.join(self.parts + ['</svg>']) + '\n'
        assert '\u2013' not in content and '\u2014' not in content
        (ASSETS / f'heads-v-{self.name}-{self.theme}.svg').write_text(content, encoding='utf-8')


def positions(theme):
    f = DetectorFigure(theme, 'positions', 432,
        'The context and question form a wrapped sequence of schematic token boxes. '
        'QRHead measures attention from the question tokens to the fact. The sequence '
        'ends in an empty answer slot, filled with 30. Wu and LOCOS measure at the '
        'step that predicts this answer token. Token boundaries are illustrative.')

    def tokens(y, words, widths, color='muted'):
        x = 28
        for word, width in zip(words, widths):
            f.box(x, y, width, 31, color + '_bg' if color != 'muted' else 'panel',
                  color if color != 'muted' else 'line', radius=4)
            f.text(x + width / 2, y + 20, word, 13, color, 'middle')
            x += width + 5

    f.text(28, 24, 'Context', 13, 'muted')
    tokens(36, ('Logs', 'rotate', 'daily', '.'), (45, 54, 46, 20))
    tokens(79, ('The', 'timeout', 'is', 'half', 'a', 'minute', '.'),
           (38, 67, 24, 38, 22, 59, 20), 'teal')
    tokens(122, ('Retries', 'use', 'backoff', '.'), (63, 36, 65, 20))
    f.text(28, 189, 'Question tokens · QRHead reads from here', 14, 'teal', bold=True)
    tokens(205, ('How', 'many', 'seconds', 'is', 'the', 'timeout', '?'),
           (42, 46, 66, 24, 32, 67, 22), 'teal')
    f.text(28, 257, 'Average their attention to the fact tokens', 12, 'muted')
    # Sequence continuation, not an attention edge: it ends at the empty slot.
    f.path('M357 221 L400 221 L400 276 L61 276 L61 294', 'muted')
    f.arrow(61, 294, 61, 306, 'muted')
    f.box(28, 314, 66, 40, 'bg', 'blue', radius=4)
    f.text(199, 319, 'model fills it', 13, 'muted', 'middle')
    f.arrow(107, 337, 289, 337, 'blue')
    f.box(302, 314, 66, 40, 'blue_bg', 'blue', radius=4)
    f.text(335, 341, '30', 22, 'blue', 'middle', True)
    f.text(28, 379, 'Answer position · Wu and LOCOS', 15, 'blue', bold=True)
    f.text(28, 400, 'Measure at the step that fills this slot', 12, 'muted')
    f.text(210, 425, 'Sequence wraps · token boundaries are schematic', 11, 'muted', 'middle')
    f.save()


def heads(theme):
    f = DetectorFigure(theme, 'heads', 353,
        'Invented roles inspired by real heads. A matches the question to the fact: '
        'close attention, small push toward 30. B moves content toward the output: '
        'less fact attention, strong push toward 30. C rests on the first token '
        'regardless of the question. Arrows illustrate behaviour, not a numeric scale.')
    for i, (head, name, color, read, push, incoming, outgoing) in enumerate((
            ('A', 'matcher', 'teal', 'Close attention', 'Small push', 4, 1),
            ('B', 'mover', 'blue', 'Less attention', 'Strong push', 1.5, 4))):
        y = 12 + i * 108
        f.box(16, y, 388, 98, 'panel', 'line')
        f.text(30, y + 23, f'Head {head} · {name}', 16, color, bold=True)
        f.text(112, y + 47, read, 12, 'muted', 'middle')
        f.text(302, y + 47, push, 12, 'muted', 'middle')
        f.box(30, y + 58, 83, 27, color + '_bg', color, radius=4)
        f.text(71, y + 76, 'fact tokens', 12, color, 'middle')
        f.arrow(121, y + 71, 187, y + 71, color, incoming)
        f.node(210, y + 71, head, color)
        f.arrow(234, y + 71, 331, y + 71, color, outgoing)
        f.box(342, y + 58, 45, 27, 'bg', 'line', radius=4)
        f.text(365, y + 77, '30', 16, color, 'middle', True)
    f.box(16, 228, 388, 92, 'panel', 'line')
    f.text(30, 251, 'Head C · resting head', 16, 'amber', bold=True)
    f.box(30, 267, 83, 30, 'amber_bg', 'amber', radius=4)
    f.text(71, 287, 'first token', 12, 'amber', 'middle')
    f.arrow(123, 282, 187, 282, 'amber', 4)
    f.node(210, 282, 'C', 'amber')
    f.text(244, 278, 'Parks here,', 13, 'muted')
    f.text(244, 297, 'any question', 13, 'muted')
    f.text(210, 343, 'Illustrative roles · arrows show information flow', 11, 'muted', 'middle')
    f.save()


def verdicts(theme):
    f = DetectorFigure(theme, 'verdicts', 228,
        'Verdicts on half a minute: Wu selects none, QRHead selects A the matcher, '
        'LOCOS selects B the mover. In the copy version Wu detects A too.')
    f.text(28, 24, 'Same fact: “half a minute” · answer: “30”', 13, 'muted')
    for i, (detector, reason, verdict, color) in enumerate((
            ('Wu', 'No “30” to copy', 'None', 'amber'),
            ('QRHead', 'Question draws attention', 'A · matcher', 'teal'),
            ('LOCOS', 'Write pushes toward “30”', 'B · mover', 'blue'))):
        y = 38 + i * 54
        f.box(16, y, 388, 46, 'panel', 'line')
        f.text(28, y + 18, detector, 14, color, bold=True)
        f.text(28, y + 36, reason, 12, 'muted')
        f.text(389, y + 28, verdict, 17, color, 'end', True)
    f.text(210, 218, 'Copy version: Wu detects A too', 13, 'muted', 'middle')
    f.save()


def wu(theme):
    f = DetectorFigure(theme, 'wu', 420,
        'Wu checks the most-attended source token against the emitted token, '
        'and requires it to lie in the fact. No 30 exists in half a minute. '
        'In the copy version Head A attends most to 30 and scores 1.')
    f.box(16, 12, 388, 171, 'panel', 'line')
    f.text(30, 36, 'The timeout is half a minute.', 16)
    f.text(112, 65, 'Inside the fact', 12, 'muted', 'middle')
    f.text(320, 65, 'Emitted token', 12, 'muted', 'middle')
    f.box(44, 77, 136, 42, 'amber_bg', 'amber')
    f.text(112, 104, 'no “30” token', 16, 'amber', 'middle')
    f.text(218, 105, '≠', 25, 'muted', 'middle')
    f.box(284, 77, 72, 42, 'blue_bg', 'blue')
    f.text(320, 105, '30', 22, 'blue', 'middle', True)
    f.text(210, 157, 'No possible match · all heads score 0', 15, 'amber', 'middle', True)

    f.box(16, 199, 388, 205, 'panel', 'line')
    f.text(30, 225, 'Copy version: The timeout is 30 seconds.', 15)
    f.text(112, 255, 'A’s most-attended', 12, 'teal', 'middle')
    f.text(112, 273, 'token inside the fact', 12, 'teal', 'middle')
    f.arrow(112, 281, 112, 298, 'teal', 2)
    f.text(320, 273, 'Emitted token', 12, 'muted', 'middle')
    f.box(76, 304, 72, 42, 'teal_bg', 'teal')
    f.text(112, 332, '30', 22, 'teal', 'middle', True)
    f.text(218, 332, '=', 25, 'muted', 'middle')
    f.box(284, 304, 72, 42, 'blue_bg', 'blue')
    f.text(320, 332, '30', 22, 'blue', 'middle', True)
    f.text(210, 386, 'Same token, inside the fact · A scores 1', 15, 'teal', 'middle', True)
    f.save()


def qrhead(theme):
    f = DetectorFigure(theme, 'qrhead', 401,
        'Same head and fact with the question or N/A. Attention is averaged over '
        'question tokens and summed over fact tokens. Subtracting N/A gives '
        'A 0.60, B 0.15 and C 0.00. All bars share a scale.')
    for i, (head, color) in enumerate((('A', 'teal'), ('B', 'blue'), ('C', 'amber'))):
        y = 12 + i * 116
        h = HEADS[head]
        real, na = h['real'][2], h['na'][2]
        f.box(16, y, 388, 104, 'panel', 'line')
        f.text(28, y + 24, f'Head {head}', 15, color, bold=True)
        f.text(390, y + 24, f'{real:.2f} − {na:.2f} = {real-na:.2f}', 16, color, 'end', True)
        for label, val, by, ink in (('Question', real, y + 38, color), ('N/A', na, y + 70, 'muted')):
            f.text(28, by + 15, label, 13, 'muted')
            f.box(111, by, 236, 20, 'bg', radius=2)
            f.box(111, by, 236 * val, 20, ink, radius=2)
            f.text(390, by + 15, f'{val:.2f}', 13, ink, 'end')
    f.text(210, 373, 'Attention summed over fact tokens', 12, 'muted', 'middle')
    f.text(210, 391, 'Averaged over receiving question tokens', 12, 'muted', 'middle')
    f.save()


def locos_write(theme):
    f = DetectorFigure(theme, 'locos-write', 490,
        'A source value becomes a full-strength write, then an attention-scaled '
        'write, then a projection onto the answer direction. The example plots '
        'use identical scales and summarize the fact tokens: A has a projection '
        '0.07, B 0.90. Faint arrows are full strength, solid arrows are delivered.')
    v, w, a, u, phi = sym('v', 'j'), sym('W', 'O'), sym('α', 'j'), sym('u', '30'), sym('φ', 'j')
    steps = [(v, 'packaged source content'),
             (w + v, 'full-strength write'),
             (a + w + v, 'delivered write'),
             (phi + ' = ' + u + ' · (' + a + w + v + ')', 'push toward “30”')]
    for i, (formula, label) in enumerate(steps):
        y = 30 + i * 110
        f.box(14, y, 194, 72, 'blue_bg' if i == 3 else 'panel', 'blue' if i == 3 else 'line')
        f.math(111, y + 30, formula, 17 if i == 3 else 23, 'blue' if i == 3 else 'ink')
        f.text(111, y + 55, label, 12, 'muted', 'middle')
        if i < 3:
            f.arrow(111, y + 77, 111, y + 104, 'muted')
    for head, top, color in (('A', 31, 'teal'), ('B', 245, 'blue')):
        h = HEADS[head]
        att = h['real'][2]
        vx, vy = h['write']
        f.text(314, top + 14, f'Head {head} · attention {att:.2f}', 12, color, 'middle', True)
        f.text(314, top + 35, f'write ({vx:.1f}, {vy:.1f})', 12, 'muted', 'middle')
        x0, y0, scale = 243, top + 128, 47
        f.arrow(x0, y0, 401, y0, 'muted', 1)
        f.arrow(x0, y0, x0, y0 - 70, 'muted', 1)
        f.text(x0 + 5, y0 - 63, 'other', 11, 'muted')
        f.arrow(x0, y0, x0 + vx * scale, y0 - vy * scale, color, 2, .25)
        xend, yend = x0 + att * vx * scale, y0 - att * vy * scale
        f.arrow(x0, y0, xend, yend, color, 2.5)
        f.line(xend, yend, xend, y0, color=color, dashed=True)
        f.line(x0, y0, xend, y0, color=color, width=5)
        f.text(323, y0 + 22, 'toward “30”', 12, 'muted', 'middle')
        f.text(323, y0 + 47, f'fact push {att * vx:.2f}', 17, color, 'middle', True)
    f.text(210, 460, 'Faint: full strength · solid: delivered', 12, 'muted', 'middle')
    f.text(210, 480, 'Thick horizontal line: projection · same scales', 12, 'muted', 'middle')
    f.save()


def locos_contrast(theme):
    f = DetectorFigure(theme, 'locos-contrast', 402,
        'Sum the pushes from fact tokens. Subtract the sum from the rest of the '
        'context, rescaled to the fact length. Here the background is near zero, '
        'leaving A 0.07, B 0.90 and C 0.01. The question stays fixed.')
    for x, title, color, labels in ((16, 'Fact', 'blue', ('The timeout is', 'half a minute.')),
                                   (226, 'Rest of context', 'muted', ('Logs rotate daily.', 'Retries use backoff.'))):
        f.text(x + 89, 26, title, 14, color, 'middle', True)
        f.box(x, 40, 178, 70, color + '_bg' if color == 'blue' else 'panel', 'line')
        for i, label in enumerate(labels):
            f.text(x + 89, 66 + i * 23, label, 14, color, 'middle')
        f.arrow(x + 89, 117, x + 89, 144, color)
        f.box(x, 154, 178, 94, 'panel', 'line')
    fact_sum = '∑<tspan baseline-shift="sub" font-size="65%">' + sym('j') + '∈F</tspan> ' + sym('φ', 'j')
    rest_sum = '∑<tspan baseline-shift="sub" font-size="65%">' + sym('j') + '∈R</tspan> ' + sym('φ', 'j')
    f.math(105, 191, fact_sum, 24, 'blue')
    f.text(105, 227, 'push from fact tokens', 12, 'muted', 'middle')
    f.math(270, 179, '|F|', 16)
    f.line(256, 185, 284, 185, color='ink', width=1)
    f.math(270, 205, '|R|', 16)
    f.math(339, 194, rest_sum, 23)
    f.text(315, 227, 'rescaled background', 12, 'muted', 'middle')
    f.text(210, 199, '−', 26, 'ink', 'middle')
    f.line(105, 256, 105, 269, color='muted')
    f.line(315, 256, 315, 269, color='muted')
    f.line(105, 269, 315, 269, color='muted')
    f.arrow(210, 269, 210, 292, 'muted')
    f.text(210, 313, 'LOCOS · background near zero here', 14, 'muted', 'middle')
    for x, head, color in ((78, 'A', 'teal'), (210, 'B', 'blue'), (342, 'C', 'amber')):
        h = HEADS[head]
        f.box(x - 54, 330, 108, 46, color + '_bg' if head == 'B' else 'panel', color if head == 'B' else 'line')
        f.text(x, 359, f"{head}  {h['real'][2] * h['write'][0]:.2f}", 19, color, 'middle', head == 'B')
    f.text(210, 396, 'Sum of token pushes · question held fixed', 12, 'muted', 'middle')
    f.save()


def real(theme):
    data = json.loads((ASSETS / 'src' / 'heads_walkthrough.json').read_text())['synthesis']
    a, b = (data['detail'][k] for k in ('L13H4', 'L20H13'))
    f = DetectorFigure(theme, 'real', 329,
                       'Fact attention is summed over its tokens. QRHead measures at the question '
                       'above N/A; answer-step attention is raw. L13 H4: 0.151, 0.034, 0.010. '
                       'L20 H13: 0.0045, 0.315, 0.434. Scores describe the supplied answer 30.')
    f.text(274, 27, 'L13 H4', 16, 'teal', 'middle', True)
    f.text(367, 27, 'L20 H13', 16, 'blue', 'middle', True)
    f.text(274, 47, 'QRHead pick', 11, 'muted', 'middle')
    f.text(367, 47, 'LOCOS pick', 11, 'muted', 'middle')
    rows = [('QRHead', 'question above N/A', (f"{a['qr']:.3f}", f"{b['qr']:.4f}"), 0, 'teal'),
            ('Fact attention', 'at the answer step', (f"{a['att_answer'][data['fact_index']]:.3f}", f"{b['att_answer'][data['fact_index']]:.3f}"), None, 'muted'),
            ('LOCOS', 'write above rest of context', (f"{a['locos']:.3f}", f"{b['locos']:.3f}"), 1, 'blue')]
    for i, (label, detail, values, pick, color) in enumerate(rows):
        y = 65 + i * 68
        f.text(18, y + 19, label, 15, bold=True)
        f.text(18, y + 39, detail, 12, 'muted')
        for j, (x, value) in enumerate(zip((274, 367), values)):
            f.box(x - 39, y, 78, 46, color + '_bg' if j == pick else 'panel', color if j == pick else 'line')
            f.text(x, y + 29, value, 18, color if j == pick else 'ink', 'middle', j == pick)
    f.line(18, 265, 402, 265, color='line')
    f.text(18, 288, 'Shared in top 10: 0', 13, 'muted')
    f.text(402, 288, 'Rank correlation: -0.04', 13, 'muted', 'end')
    f.text(210, 316, 'Attention summed over fact tokens · one item', 12, 'muted', 'middle')
    f.save()


def main():
    drawings = (ablation, quality, direction, knockout) if '--sequel' in sys.argv else (
        positions, heads, wu, qrhead, locos_write, locos_contrast, verdicts, real)
    for theme in PALETTES:
        for draw in drawings:
            draw(theme)


if __name__ == '__main__':
    main()
