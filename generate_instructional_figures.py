#!/usr/bin/env python3
"""Generate the book's deterministic causal-inference diagrams as SVG."""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "instructional"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0b2038"
TEAL = "#168c8c"
CORAL = "#e76f51"
GOLD = "#d6a84b"
IVORY = "#fbf8f1"
MUTED = "#66788a"


def wrap(title, subtitle, content, width=960, height=520):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{title}</title><desc id="desc">{subtitle}</desc>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{NAVY}"/></marker><filter id="shadow"><feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="{NAVY}" flood-opacity=".14"/></filter></defs>
<rect width="100%" height="100%" rx="28" fill="{IVORY}"/>
<text x="48" y="55" font-family="Inter,Arial,sans-serif" font-size="28" font-weight="700" fill="{NAVY}">{title}</text>
<text x="48" y="84" font-family="Inter,Arial,sans-serif" font-size="15" fill="{MUTED}">{subtitle}</text>
{content}</svg>'''


def box(x, y, w, h, label, fill="#ffffff", stroke=TEAL, size=18):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2" filter="url(#shadow)"/><text x="{x+w/2}" y="{y+h/2+6}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="{size}" font-weight="600" fill="{NAVY}">{label}</text>'


def line(x1, y1, x2, y2, color=NAVY, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"{dash_attr}/>'


figures = {}

figures["potential-outcomes.svg"] = wrap("The missing counterfactual", "Each unit has two potential outcomes, but treatment reveals only one.",
    box(70,155,210,86,"Unit i",fill="#fff") + line(280,198,390,155) + line(280,198,390,310) +
    box(395,115,230,82,"Yᵢ(1): treated",fill="#e8f5f3") + box(395,270,230,82,"Yᵢ(0): control",fill="#fff0eb",stroke=CORAL) +
    box(700,115,190,82,"Observed",fill="#e8f5f3") + box(700,270,190,82,"Missing",fill="#f4f1eb",stroke=MUTED) +
    line(625,156,700,156,color=TEAL) + line(625,311,700,311,color=MUTED,dash="8 7") +
    f'<text x="480" y="425" font-family="Inter,Arial" font-size="20" font-weight="700" fill="{NAVY}">τᵢ = Yᵢ(1) − Yᵢ(0)</text><text x="480" y="455" font-family="Inter,Arial" font-size="15" fill="{MUTED}">Individual effects require a comparison we can never directly observe.</text>')

figures["dag-motifs.svg"] = wrap("Three paths, three conditioning rules", "Chains and forks transmit association; colliders block it until conditioned on.",
    box(55,155,105,64,"X")+box(205,155,105,64,"M")+box(355,155,105,64,"Y")+line(160,187,205,187)+line(310,187,355,187)+
    f'<text x="258" y="250" text-anchor="middle" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{TEAL}">CHAIN — condition to close</text>'+
    box(500,155,105,64,"X")+box(650,120,105,64,"C")+box(800,155,105,64,"Y")+line(700,184,605,177)+line(705,184,800,177)+
    f'<text x="705" y="250" text-anchor="middle" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{TEAL}">FORK — condition to close</text>'+
    box(275,335,105,64,"X")+box(430,335,105,64,"S",fill="#fff0eb",stroke=CORAL)+box(585,335,105,64,"Y")+line(380,367,430,367)+line(585,367,535,367)+
    f'<text x="482" y="440" text-anchor="middle" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{CORAL}">COLLIDER — conditioning opens the path</text>')

figures["rct-flow.svg"] = wrap("Assignment, receipt, and analysis", "Random assignment identifies the effect of assignment; noncompliance separates it from treatment received.",
    box(50,210,150,72,"Eligible")+line(200,246,300,246)+box(305,200,180,92,"Randomize",fill="#fff7df",stroke=GOLD)+
    line(485,235,590,155)+line(485,255,590,350)+box(595,115,190,80,"Assigned A",fill="#e8f5f3")+box(595,310,190,80,"Assigned B",fill="#fff0eb",stroke=CORAL)+
    line(785,155,880,155)+line(785,350,880,350)+f'<text x="870" y="135" text-anchor="end" font-family="Inter,Arial" font-size="15" fill="{MUTED}">may not receive A</text><text x="870" y="330" text-anchor="end" font-family="Inter,Arial" font-size="15" fill="{MUTED}">may access A</text><text x="480" y="455" text-anchor="middle" font-family="Inter,Arial" font-size="20" font-weight="700" fill="{NAVY}">ITT compares groups as assigned</text>')

figures["propensity-overlap.svg"] = wrap("Overlap is a design property", "Propensity methods compare units only where both treatments are plausible.",
    f'<path d="M70 405 C160 400 180 170 350 170 C500 170 530 395 620 405" fill="none" stroke="{TEAL}" stroke-width="5"/><path d="M340 405 C440 395 470 190 630 190 C790 190 820 400 900 405" fill="none" stroke="{CORAL}" stroke-width="5"/><rect x="340" y="125" width="280" height="300" fill="{GOLD}" opacity=".12"/><line x1="70" y1="410" x2="900" y2="410" stroke="{NAVY}" stroke-width="2"/><text x="480" y="460" text-anchor="middle" font-family="Inter,Arial" font-size="18" fill="{NAVY}">Propensity score e(X): 0 ⟶ 1</text><text x="205" y="145" font-family="Inter,Arial" font-size="18" font-weight="700" fill="{TEAL}">Control</text><text x="720" y="165" font-family="Inter,Arial" font-size="18" font-weight="700" fill="{CORAL}">Treated</text><text x="480" y="115" text-anchor="middle" font-family="Inter,Arial" font-size="18" font-weight="700" fill="{GOLD}">credible comparison region</text>')

figures["iv-graph.svg"] = wrap("Instrumental variables identify through encouragement", "Relevance is visible in data; independence and exclusion require a design argument.",
    box(80,210,180,76,"Instrument Z",fill="#fff7df",stroke=GOLD)+box(390,210,180,76,"Treatment X",fill="#e8f5f3")+box(700,210,180,76,"Outcome Y",fill="#fff0eb",stroke=CORAL)+
    line(260,248,390,248)+line(570,248,700,248)+box(390,365,180,70,"Unmeasured U",fill="#f4f1eb",stroke=MUTED)+line(455,365,455,286,color=MUTED,dash="7 6")+line(525,365,760,286,color=MUTED,dash="7 6")+
    f'<path d="M260 225 C430 90 650 90 700 225" fill="none" stroke="{CORAL}" stroke-width="3" stroke-dasharray="8 7"/><text x="480" y="125" text-anchor="middle" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{CORAL}">this direct path must be absent</text>')

figures["did-trends.svg"] = wrap("Difference-in-differences", "The missing counterfactual is a trend, not a post-treatment level.",
    f'<line x1="100" y1="420" x2="890" y2="420" stroke="{NAVY}" stroke-width="2"/><line x1="100" y1="420" x2="100" y2="120" stroke="{NAVY}" stroke-width="2"/><line x1="500" y1="110" x2="500" y2="430" stroke="{GOLD}" stroke-width="3" stroke-dasharray="8 7"/><polyline points="110,355 260,315 410,275 560,235 710,195 870,155" fill="none" stroke="{TEAL}" stroke-width="5"/><polyline points="110,300 260,260 410,220 560,180 710,140 870,100" fill="none" stroke="{CORAL}" stroke-width="5"/><polyline points="410,220 560,180 710,140 870,100" fill="none" stroke="{MUTED}" stroke-width="4" stroke-dasharray="9 8"/><polyline points="410,220 560,255 710,275 870,300" fill="none" stroke="{CORAL}" stroke-width="5"/><text x="520" y="105" font-family="Inter,Arial" font-size="16" font-weight="700" fill="{GOLD}">policy</text><text x="730" y="330" font-family="Inter,Arial" font-size="16" font-weight="700" fill="{CORAL}">observed treated</text><text x="710" y="125" font-family="Inter,Arial" font-size="16" font-weight="700" fill="{MUTED}">counterfactual</text><text x="720" y="185" font-family="Inter,Arial" font-size="16" font-weight="700" fill="{TEAL}">control</text>')

figures["rdd-cutoff.svg"] = wrap("Regression discontinuity is local", "Estimate the jump at the assignment threshold using observations near the cutoff.",
    f'<line x1="100" y1="420" x2="890" y2="420" stroke="{NAVY}" stroke-width="2"/><line x1="100" y1="420" x2="100" y2="120" stroke="{NAVY}" stroke-width="2"/><line x1="500" y1="110" x2="500" y2="430" stroke="{GOLD}" stroke-width="4" stroke-dasharray="8 7"/><path d="M120 360 Q300 310 490 260" fill="none" stroke="{TEAL}" stroke-width="5"/><path d="M510 180 Q700 150 870 125" fill="none" stroke="{CORAL}" stroke-width="5"/><circle cx="490" cy="260" r="8" fill="{TEAL}"/><circle cx="510" cy="180" r="8" fill="{CORAL}"/><line x1="545" y1="255" x2="545" y2="185" stroke="{NAVY}" stroke-width="3" marker-end="url(#arrow)"/><text x="560" y="230" font-family="Inter,Arial" font-size="18" font-weight="700" fill="{NAVY}">local effect</text><text x="500" y="465" text-anchor="middle" font-family="Inter,Arial" font-size="17" fill="{NAVY}">running variable and cutoff c</text>')

figures["synthetic-control.svg"] = wrap("Synthetic control", "Match before treatment; interpret the post-treatment gap.",
    f'<line x1="100" y1="420" x2="890" y2="420" stroke="{NAVY}" stroke-width="2"/><line x1="100" y1="420" x2="100" y2="120" stroke="{NAVY}" stroke-width="2"/><line x1="520" y1="110" x2="520" y2="430" stroke="{GOLD}" stroke-width="3" stroke-dasharray="8 7"/><path d="M110 350 C220 300 320 330 420 250 C470 220 500 225 520 210 C620 155 740 120 880 110" fill="none" stroke="{CORAL}" stroke-width="5"/><path d="M110 355 C220 305 320 335 420 255 C470 225 500 230 520 215 C630 245 750 260 880 300" fill="none" stroke="{TEAL}" stroke-width="5" stroke-dasharray="10 7"/><text x="680" y="135" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{CORAL}">observed treated unit</text><text x="670" y="290" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{TEAL}">weighted donor units</text><text x="535" y="105" font-family="Inter,Arial" font-size="16" font-weight="700" fill="{GOLD}">intervention</text>')

figures["mediation-effects.svg"] = wrap("Mediation effects", "Natural effects compare nested counterfactuals and require stronger assumptions than the total effect.",
    box(85,280,180,76,"Treatment A",fill="#fff7df",stroke=GOLD)+box(390,130,180,76,"Mediator M",fill="#e8f5f3")+box(695,280,180,76,"Outcome Y",fill="#fff0eb",stroke=CORAL)+line(265,300,410,205)+line(570,168,750,280)+line(265,318,695,318)+
    f'<text x="480" y="365" text-anchor="middle" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{NAVY}">Natural direct effect: change A while holding M at M(0)</text><text x="480" y="405" text-anchor="middle" font-family="Inter,Arial" font-size="17" font-weight="700" fill="{TEAL}">Natural indirect effect: change M(0) → M(1) while holding A = 1</text>')

figures["longitudinal-feedback.svg"] = wrap("Treatment-confounder feedback", "A time-varying covariate can be both an effect of earlier treatment and a cause of later treatment.",
    box(55,150,150,68,"A₁")+box(290,150,150,68,"L₂",fill="#fff7df",stroke=GOLD)+box(525,150,150,68,"A₂")+box(760,150,150,68,"Y",fill="#fff0eb",stroke=CORAL)+line(205,184,290,184)+line(440,184,525,184)+line(675,184,760,184)+line(205,205,760,205,color=TEAL)+line(440,205,760,205,color=TEAL)+
    f'<text x="365" y="290" text-anchor="middle" font-family="Inter,Arial" font-size="17" fill="{MUTED}">L₂ is post-treatment for A₁</text><text x="600" y="320" text-anchor="middle" font-family="Inter,Arial" font-size="17" fill="{MUTED}">and a confounder for A₂ → Y</text><text x="480" y="405" text-anchor="middle" font-family="Inter,Arial" font-size="20" font-weight="700" fill="{CORAL}">Standard adjustment creates a dilemma; g-methods resolve it.</text>')

figures["transportability.svg"] = wrap("Transporting an effect", "Generalization requires an explicit account of what changes between source and target populations.",
    box(70,205,230,100,"Source trial S=1",fill="#e8f5f3")+box(660,205,230,100,"Target S=0",fill="#fff0eb",stroke=CORAL)+line(300,255,660,255,color=GOLD,dash="10 7")+
    box(365,130,230,70,"Effect modifiers X",fill="#fff7df",stroke=GOLD)+box(365,330,230,70,"Invariant mechanism",fill="#fff")+
    f'<text x="480" y="285" text-anchor="middle" font-family="Inter,Arial" font-size="18" font-weight="700" fill="{NAVY}">reweight X and defend invariance</text>')

figures["causal-workflow.svg"] = wrap("The causal workflow", "Design decisions precede estimation; diagnostics challenge rather than prove assumptions.",
    ''.join(box(35+i*152,190,130,72,label,fill=("#e8f5f3" if i in (0,3) else "#ffffff"),stroke=(TEAL if i<4 else CORAL),size=15) + (line(165+i*152,226,187+i*152,226) if i<5 else '') for i,label in enumerate(["Question","Estimand","Identification","Estimation","Diagnostics","Interpretation"]))+
    f'<text x="480" y="335" text-anchor="middle" font-family="Inter,Arial" font-size="19" font-weight="700" fill="{NAVY}">No estimator can repair an unidentified question.</text><text x="480" y="375" text-anchor="middle" font-family="Inter,Arial" font-size="17" fill="{MUTED}">Report the target, assumptions, evidence, uncertainty and limits together.</text>')

for filename, svg in figures.items():
    (OUT / filename).write_text(svg)

print(f"Generated {len(figures)} instructional figures in {OUT}")
