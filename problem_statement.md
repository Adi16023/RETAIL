QUESSATHON
NEXT, ENGINEERED
PROBLEM STATEMENT BANK

Can your AI agent survive reality?
Quessathon is not about building an impressive AI demo. It is about putting AI agents against genuinely hard enterprise problems where data is messy, variables are hidden, incentives conflict and the number of possible outcomes is impossible to fully predict.
Because a demo operates in controlled conditions. The real world doesn't.
The challenge isn't simply: can you build an agent? It's: can you build one that knows when it's out of its depth?
 
Different problems. One question.
Does your agent know what it doesn't know?

Common to All Challenges
The following applies uniformly across all nine challenges and is not repeated within each one. Read this section once, before selecting a challenge.
DATA & TOOLING RULES
What's allowed, and what isn't, is defined explicitly here so there is no ambiguity during the build window.
•  Public data: allowed and encouraged. Real, citable public sources, regulatory circulars, published industry reports, documented events, are cited within each challenge where they exist; ground your solution in them.
•  Synthetic or self-constructed data: allowed, and expected where no public equivalent exists. You are responsible for constructing realistic, representative data and must be able to defend your construction choices live; this realism is itself assessed.
•  Public APIs and web sources: allowed. You may call public APIs or retrieve public web content as part of your agent's reasoning, provided the source is disclosed.
•  LLM and model outputs: allowed as a standard build tool, for reasoning, synthetic data generation, or as part of your agent's own architecture. Not a substitute for your agent's own investigative reasoning.
•  Proprietary or third-party company data: not allowed. No team may use data proprietary to their own company or any third party, to preserve a level playing field across all nine challenges.
•  Physical or hardware components: not allowed. This is a software and reasoning challenge in its entirety; several challenges (including Counterfeit Parts Verification) are explicitly documents-and-records investigations, not physical inspection.
THE DELIVERABLE
Each team submits exactly three artefacts: a working, live-runnable prototype, not a recorded demonstration; a one-page architecture note describing what the system does at each step, what it depends on, and where it hands control to a human; and a live stage presentation of five minutes to pitch, three minutes to demonstrate, and two minutes of live jury questioning.
LIVE EVALUATION
During the questioning segment, staged on event day as the Reality Test, the jury introduces one scenario the team has not previously seen. The specific scenario used for each challenge is deliberately not disclosed in advance, so teams should build for genuine robustness rather than a known test case.
JURY ASSESSMENT
Every challenge, regardless of industry, is scored against the identical rubric below. The standard does not vary by sector; only the problem does. Judgement and robustness under uncertainty carries the greatest weight deliberately: it is the one dimension that most directly separates a system that has been engineered from one that has simply been adapted to look confident.
JURY ASSESSMENT (CONTINUED)

Criterion
Weight
High score looks like
Low score looks like
Problem understanding
15%
Grasps the real complexity behind the challenge, not just its literal wording
Solves a simplified version that would not survive contact with the actual data or stakeholders
Judgement & robustness under uncertainty
30%
Handles the jury's unseen scenario soundly, and defers or flags low confidence when the evidence genuinely doesn't support a clean answer
Fails on the unseen scenario, or forces a confident answer regardless of the evidence
Engineering quality
20%
Architecture is coherent, runs live without failure, and reflects deliberate design choices
Fragile, breaks under live conditions, or reads as assembled rather than designed
Explainability
15%
Every output comes with reasoning a business stakeholder could act on directly
A score or classification with no accompanying justification
Business impact
20%
The value delivered is specific, credible, and tied to a real role or decision inside the business
Impressive in isolation, with no clear path to who would use it or what it would change


RETAIL


CHALLENGE: RETAIL | Where Did the Revenue Go?

THE CHALLENGE
Build an AI agent that investigates an account's transaction history to identify where and why revenue is leaking, even when the account continues to place orders, and recommends where the business should focus its intervention.

THE CONTEXT
An account does not have to stop buying to become less valuable. It may quietly shift its product mix, reduce purchases in high-value categories, change order sizes or move part of its requirements elsewhere while remaining an active customer.
The account can therefore appear healthy at an overall revenue level while its underlying economics are deteriorating.
The challenge is to uncover what changed, where the value was lost, and whether the change is temporary or structural.

THE AGENT MUST
Detect → Investigate → Attribute → Prioritise

MINIMUM CAPABILITY
Establish the account's historical purchasing pattern and identify meaningful changes in revenue, product mix, order size and purchase behaviour
Locate the specific products, categories or purchasing behaviours responsible for the change, rather than reporting only an overall revenue decline
Distinguish temporary purchasing variation from structural revenue leakage using historical evidence
Quantify or clearly articulate the commercial significance of the identified leakage and prioritise where intervention could have the greatest impact
Explain the evidence behind its diagnosis and flag when the available transaction history is insufficient to establish a reliable cause

THE UNSEEN SCENARIO
The jury introduces an account whose total revenue has remained broadly stable.
A surface-level analysis marks the account as healthy.
The agent must discover that the account has stopped purchasing historically high-value categories while increasing purchases of lower-value products, and determine whether this represents temporary mix change or a structural loss of valuable business.

EXPECTED IMPACT
A working system designed to surface hidden revenue leakage before it becomes visible as account churn, helping revenue teams identify where an apparently healthy account is quietly losing commercial value.