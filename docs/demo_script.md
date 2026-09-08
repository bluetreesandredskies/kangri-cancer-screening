# Demo Script — Paakzir (2 minutes)

*Team Paakzir: Aneek Debnath, Manu Bhagat, Jeetu Kumar, Sakshi Prajapati*

---

## Hook (15s)

"Every winter, thousands of people in Kashmir sit for hours beside a kangri —
a small clay firepot tucked under their pheran — to survive the cold. For
some, that same warmth is quietly causing skin cancer. We built a tool that
can catch it early, using nothing but a phone camera."

## Problem (20s)

"Kangri cancer — usually squamous cell carcinoma — comes from years of
chronic thermal exposure. Early lesions look like ordinary burns or calluses,
so people don't seek care until it's advanced. And dermatologists are scarce
in the region — many patients would have to travel hours just to get a
lesion looked at. There's no accessible, low-cost way to get a first opinion
on whether a spot on your skin is worth worrying about."

## Solution + Live Demo (40s)

"That's Paakzir. You open the web app, take or upload a photo of a lesion,
and hit Analyze. [Click Analyze on a sample image live.] In a couple of
seconds, our model — an EfficientNet-B0 trained on over 12,000 dermatology
images — returns a risk level, a confidence score, and a plain-language
recommendation. [Point at the badge as it appears.] Notice this isn't just a
single label — if our model detects meaningful signs of a serious pattern
even when it isn't the top guess, we escalate the risk level and explain why,
right here in this notice. [Point at the escalation box.] This is a screening
aid, not a diagnosis — we say that clearly on every screen — but it's a
first, fast signal that can tell someone 'go see a doctor now' instead of
'wait and see.'"

## Business Model (25s)

"Our path to sustainability is a freemium screening tool distributed through
community health workers and local pharmacies in affected regions, paired
with a paid tier for clinics and telehealth providers who want lesion
triage built into their intake flow. Over time, partnership data from
regional hospitals lets us license a kangri-specific model back to the health
system that helped us build it."

## Scalability (15s)

"The architecture is deliberately simple: a stateless FastAPI backend and a
static React frontend, so it scales horizontally on commodity infrastructure
with no per-user cost beyond compute. The same pipeline generalizes to any
region with a distinct thermal or occupational skin cancer risk — we just
need local labeled data."

## Ask (5s)

"We're looking for a clinical partner with regional lesion data, and
mentorship on the regulatory path to a real screening deployment."
