"""Blog illustration: Wu, QRHead and LOCOS on one toy item and its copy twin, Qwen3-0.6B, CPU.

Run from a checkout of egordm/skllm-dev (v0.1 line): uv run python <this file>
Writes heads_walkthrough.json beside this file. An illustration for a blog post, not evidence.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from skllm.attribution.locos import LocosContrast, LocosTrajectory
from skllm.attribution.locos.qwen3 import Qwen3LocosAttributor
from skllm.attribution.qrhead import QrHeadComparison, QuerySuffix
from skllm.attribution.qrhead.qwen3 import Qwen3QrHeadAttributor
from skllm.core.qwen3 import Qwen3Backbone

CKPT = "Qwen/Qwen3-0.6B"
REV = "c1899de289a04d12100db370d81485cdf75e47ca"
FILLER = [
    "The service writes its logs to a shared disk.",
    "Logs rotate daily.",
    "Each request carries a trace id.",
    None,  # the fact goes here
    "Retries use exponential backoff.",
    "The cache keeps the last hundred results.",
    "Errors go to the on-call channel.",
]
FACTS = {"synthesis": "The timeout is half a minute.", "copy": "The timeout is 30 seconds.",
         "synthesis_300": "The timeout is half a minute."}
REPLAY = {"synthesis": "30", "copy": "30", "synthesis_300": "300"}  # 300 = the model's own answer
FACT_INDEX = FILLER.index(None)
QUESTION = "What is the timeout in seconds? Reply with a single number."
HEAD = "Read the notes below.\n\nNotes: "
TOPK = 10

tok = AutoTokenizer.from_pretrained(CKPT, revision=REV)
model = AutoModelForCausalLM.from_pretrained(CKPT, revision=REV, dtype=torch.float32)
model.eval()
model.set_attn_implementation("sdpa")
backbone = Qwen3Backbone(model)
L, H = backbone.layer_count, backbone.query_heads
head_ids = np.arange(L * H, dtype=np.int64).reshape(L, H)


def enc(text):
    return tok.encode(text, add_special_tokens=False)


def render(user):
    return tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def run(kind):
    sentences = [s if s is not None else FACTS[kind] for s in FILLER]
    doc = " ".join(sentences)
    user = f"{HEAD}{doc}\n\nQuestion: {QUESTION}"
    text = render(user)
    q_start = text.index(QUESTION)
    # context = everything before the question, cut at sentence boundaries
    starts, pos = [], text.index(doc)
    for s in sentences:
        starts.append((pos, pos + len(s)))
        pos += len(s) + 1
    cuts = sorted({0, q_start, *(e for span in starts for e in span)})
    pieces = [enc(text[a:b]) for a, b in zip(cuts, cuts[1:])]
    at = dict(zip(cuts, np.cumsum([0] + [len(p) for p in pieces]).tolist()))
    context = [t for p in pieces for t in p]
    spans = [(at[a], at[b]) for a, b in starts]
    actual = enc(text[q_start:])
    null = enc("N/A" + text[q_start + len(QUESTION):])
    prompt = context + actual
    assert tok.decode(prompt) == text

    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=24, do_sample=False)
    own = out[0, len(prompt):].tolist()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    own = own[: own.index(end)] if end in own else own
    own_answer = tok.decode(own)
    print(kind, "own answer:", repr(own_answer))
    # the detectors replay a given answer: the correct "30", and the model's own "300"
    cont = enc(REPLAY[kind])
    answer = tok.decode(cont)

    model.set_attn_implementation("sdpa")
    # QRHead: query-token attention mass on each sentence, minus the N/A query
    qr = Qwen3QrHeadAttributor(backbone, max_capture_bytes=512_000_000).attribute(
        QrHeadComparison(
            context_token_ids=tuple(context),
            query=QuerySuffix(token_ids=tuple(actual), focus_span=(0, len(actual))),
            null_query=QuerySuffix(token_ids=tuple(null), focus_span=(0, len(null))),
            source_spans=np.asarray(spans, dtype=np.int64),
        ),
        head_ids,
    )
    qr_cal = qr.calibrated_qr_scores  # (sources, L, H)

    # LOCOS: write toward each answer token, source minus rescaled background
    steps = np.array([i for i, t in enumerate(cont) if tok.decode([t]).strip()], dtype=np.int64)
    traj = LocosTrajectory(prompt_token_ids=tuple(prompt), continuation_token_ids=tuple(cont))
    loc = Qwen3LocosAttributor(backbone).attribute(
        traj,
        tuple(
            LocosContrast(source_span=sp, steps=steps, direction_token_ids=np.asarray(cont)[steps])
            for sp in spans
        ),
    )

    model.set_attn_implementation("eager")
    # Wu et al.: at each answer step, does a head's most-attended context token equal the token
    # it emits, inside the fact? Score = hits / answer tokens.
    with torch.no_grad():
        att = model(torch.tensor([prompt + cont]), output_attentions=True).attentions
    fa, fb = spans[FACT_INDEX]
    wu = np.zeros((L, H))
    for s in steps:
        row = len(prompt) + s - 1
        for layer in range(L):
            arg = att[layer][0, :, row, : len(prompt)].argmax(-1).numpy()
            for h in range(H):
                j = arg[h]
                wu[layer, h] += (fa <= j < fb) and prompt[j] == cont[s]
    wu /= max(len(steps), 1)
    # attention at the answer steps on each sentence, per head
    ans_att = np.stack([r.source_attention.mean(0) for r in loc])  # (sources, L, H)

    return dict(
        kind=kind, answer=answer, own_answer=own_answer, answer_tokens=[tok.decode([t]) for t in cont],
        steps=steps.tolist(), sentences=sentences, spans=spans, n_prompt=len(prompt),
        qr=qr_cal[FACT_INDEX], locos=loc[FACT_INDEX].score,
        phi_plus=loc[FACT_INDEX].source_contribution.mean(0), wu=wu,
        ans_att=ans_att, qr_all=qr_cal,
    )


def top(m, k=TOPK):
    flat = np.argsort(-m.ravel())[:k]
    return [(int(i // H), int(i % H), float(m.ravel()[i])) for i in flat]


result = {}
for kind in FACTS:
    r = run(kind)
    tops = {name: top(r[name]) for name in ("wu", "qr", "locos")}
    sets = {n: {(a, b) for a, b, _ in v} for n, v in tops.items()}
    rho = spearmanr(r["qr"].ravel(), r["locos"].ravel()).statistic
    union = {(a, b) for v in tops.values() for a, b, _ in v}
    detail = {
        f"L{a}H{b}": dict(
            qr=float(r["qr"][a, b]), locos=float(r["locos"][a, b]), phi_plus=float(r["phi_plus"][a, b]),
            wu=float(r["wu"][a, b]), att_answer=[round(float(x), 3) for x in r["ans_att"][:, a, b]],
            att_query_cal=[round(float(x), 3) for x in r["qr_all"][:, a, b]],
        )
        for a, b in union
    }
    result[kind] = dict(
        answer=r["answer"], own_answer=r["own_answer"], answer_tokens=r["answer_tokens"], sentences=r["sentences"],
        fact_index=FACT_INDEX, n_prompt=r["n_prompt"], tops=tops,
        wu_nonzero=int((r["wu"] > 0).sum()),
        overlap=dict(qr_locos=len(sets["qr"] & sets["locos"]), wu_qr=len(sets["wu"] & sets["qr"]),
                     wu_locos=len(sets["wu"] & sets["locos"])),
        spearman_qr_locos=float(rho), detail=detail,
    )
    print(json.dumps({k: v for k, v in result[kind].items() if k != "detail"}, indent=1))

both = {n: [{(a, b) for a, b, _ in result[k]["tops"][n]} for k in ("synthesis", "copy")] for n in ("qr", "locos")}
result["locos_top10_shared_30_vs_300"] = len({(a, b) for a, b, _ in result["synthesis"]["tops"]["locos"]} & {(a, b) for a, b, _ in result["synthesis_300"]["tops"]["locos"]})
result["cross_twin_top10_overlap"] = {n: len(v[0] & v[1]) for n, v in both.items()}
print(result["cross_twin_top10_overlap"])
Path(__file__).with_name("heads_walkthrough.json").write_text(json.dumps(result, indent=1))
