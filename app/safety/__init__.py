"""Code-level safety enforcement.

The system prompt already asks the model to behave — to refuse a diagnosis, to
point a distressed user at a helpline, not to promise certainty. That is
guidance, not a guarantee: a prompt instruction is a request the model is free
to lose track of halfway through a long turn, and the one turn where it does
is exactly the turn that matters.

So the rules that carry real-world consequence are enforced in code, on both
sides of the model:

* **before** the LLM runs — `input_guard` (prompt injection, card/phone
  numbers), `crisis` (self-harm), `regulated` (medical/legal/financial
  directives), `boundary` (contact details, meeting requests), `scope`
  (out-of-domain). A hit here short-circuits the turn with a canned reply and
  the model is never called at all, so there is nothing for it to override.
* **after** the LLM runs — `output_guard`, which catches the loud structural
  failures (an "as an AI language model" leak, a hard certainty promise, a
  phone number in the answer).

Everything in this package is a pure function over text. No I/O, no network,
no model call, so the gates cost microseconds and cannot themselves fail open
because a provider is down.
"""

from __future__ import annotations

from app.safety.block_messages import block_message, crisis_block_message
from app.safety.boundary import BoundaryProbe, detect_boundary_probe
from app.safety.categories import SafetyCategory
from app.safety.crisis import CrisisCheck, check_crisis
from app.safety.input_guard import InputCheck, check_input
from app.safety.output_guard import OutputCheck, check_output
from app.safety.regulated import RegulatedDirective, detect_regulated_directive
from app.safety.scope import ScopeCheck, check_scope

__all__ = [
    "BoundaryProbe",
    "CrisisCheck",
    "InputCheck",
    "OutputCheck",
    "RegulatedDirective",
    "SafetyCategory",
    "ScopeCheck",
    "block_message",
    "check_crisis",
    "check_input",
    "check_output",
    "check_scope",
    "crisis_block_message",
    "detect_boundary_probe",
    "detect_regulated_directive",
]
