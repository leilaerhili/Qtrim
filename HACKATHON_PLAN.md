# Team Structure and Execution Plan
12-hour hackathon | 4 technical contributors + 1 presentation lead

Goal
Minimize dependencies so no one is blocked. Everyone owns a subsystem, mocks the rest, and integrates late via simple JSON contracts.

Team roles and responsibilities

Person 1: RL Environment and Rewrite Engine (core logic)
Owner: JASPER
Files: core/rewrites.py, core/metrics.py, core/env_quantum_opt.py
Responsibilities:
- Define action space (rewrite rules)
- Implement rewrite logic
- Define cost metrics (gate count, depth, weighted cost)
- Implement the Gymnasium environment
Why unblocked:
- Uses hardcoded baseline circuits
- No dependency on UI, phone, or training loop
Deliverable by hour ~5:
- Environment runs end to end
- Random/naive agent occasionally improves circuits
- Cost metrics decrease after valid rewrites

Person 2: RL Training and Policy Interface
Owner: YASSINE
Files: core/train_policy.py, core/policy_store/, core/shared_schema.py
Responsibilities:
- Choose and configure RL algorithm (PPO recommended)
- Train a policy on the environment
- Save and load trained policies
- Define observation vector schema
- Expose get_action(observation) interface
Why unblocked:
- Can use a mock or stub environment initially
- Can stub rewards early before real rewrites exist
- Only depends on environment interface, not UI or phone
Deliverable by hour ~7:
- Trained policy with improving reward signal
- Callable function that returns an action given an observation

Person 3: Phone Side Client (last-mile adapter)
Owner: MANA
Files: phone/
Responsibilities:
- Implement a phone client (Android app or Termux Python client) that:
  - Receives current observation
  - Applies a constraint profile (low noise vs low latency)
  - Returns an action choice
- Simulate backend constraints (noise budgets, backend preferences)
- Ensure reliable communication with the PC API
Why unblocked:
- Can start with a mock PC API
- Decision logic can be simple and rule-based
- RL policy inference can be swapped in later
Deliverable by hour ~9:
- Phone sends real HTTP requests
- PC logs show decisions from the phone

Person 4: PC Control Surface and Orchestration
Owner: MAHSA
Files: pc/app_streamlit.py, pc/api_server.py, pc/demo_run.ps1
Responsibilities:
- Build the Streamlit control surface
- Implement the FastAPI server
- Orchestrate between components
- Visualize before/after metrics
- Define the demo flow
Why unblocked:
- Can start with fake or random optimization outputs
- Can mock agent decisions early
- UI and API contracts defined upfront
Deliverable by hour ~9:
- One-click demo UI
- Clear visualization of circuit improvement

Person 5: Presentation Lead (non-blocking)
Owner: Presentation lead
Responsibilities:
- Craft pitch narrative and slides
- Demo script and Q&A prep
- Align messaging with Snapdragon and multi-device framing
Important:
- Starts immediately and never waits for final code

Parallel timeline

Hour 0-1
- Agree on action set, cost metrics, API + JSON schema
- Presentation lead drafts pitch outline

Hour 1-4
- Person 1 builds rewrite rules and metrics
- Person 2 builds training loop skeleton
- Person 3 builds phone client skeleton
- Person 4 builds Streamlit UI skeleton
- Presentation lead drafts slides

Hour 4-7
- Person 1 finalizes the environment
- Person 2 trains the RL policy
- Person 3 tests phone to PC communication
- Person 4 integrates mock optimization results
- Presentation lead refines narrative

Hour 7-9
- Integrate trained policy into PC API
- Phone switches from mock to real action selection
- UI displays real optimization
- Presentation lead scripts demo walkthrough

Hour 9-12
- Polish and bug fixes
- Demo practice
- README and submission prep

Integration rule (critical)
All interfaces must be JSON-first and intentionally simple.
Example observation payload:
{
  "gate_count": 72,
  "depth": 28,
  "num_cnot": 20,
  "num_rz": 32,
  "constraint_profile": "low_noise"
}

Guiding principle
Everyone owns a full subsystem and mocks the rest. Integration is shallow and late.
