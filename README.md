# Smart Inquiry Triage Assistant

A local automotive support prototype that classifies inquiries, derives priority from historical cases, selects a support queue, and drafts suggested next actions. Streamlit displays the result, retrieved evidence, and a human-review flag.

Built from the supplied [case study repository](https://github.com/Shubham7806171/AI-Engineer-Case-Study). See the [assignment PDF](case_study_smart_inquiry_triage.pdf).

## Setup

Prerequisites: Git, Python, and Ollama. Developed on Windows with **Python 3.13.9**. Commands below use **Windows Command Prompt (CMD)**.

### 1. Install dependencies

```bat
git clone https://github.com/jassimh/smart-inquiry-triage.git
cd smart-inquiry-triage
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements-lock.txt
python -m pip check
```

For an existing checkout, activate its environment and run commands from the repository root. The lock file records development package versions; a fresh installation has not been verified.

### 2. Prepare the models

```bat
ollama pull qwen3:4b
ollama pull nomic-embed-text:latest
```

Keep Ollama running at `http://127.0.0.1:11434`. If needed, run `ollama serve` in a separate terminal. Downloads require internet access; inference runs locally.

### 3. Build the knowledge base

```bat
python -m src.knowledge_base --ingest
```

This validates and embeds the **300 supplied inquiries across eight categories**, storing vectors and metadata in `.chroma/`. The index is excluded from Git and must be built after cloning.

## Run

Start the interface:

```bat
python -m streamlit run app/app.py
```

Open the address printed by Streamlit. The sidebar controls **Top-K** (1–10, default 3) and **confidence threshold** (0–1, default 0.5). Previous results remain visible within the session; they are not conversational memory or permanent storage.

For terminal output, including completed workflow nodes:

```bat
python -m src.main --top-k 5 --threshold 0.5
```

Enter an inquiry when prompted. Stop the interface with `Ctrl+C`. If retrieval reports a missing or incomplete index, rerun ingestion with Ollama running.

## Models and stack

| Component | Choice |
| --- | --- |
| Classification and resolution drafting | `qwen3:4b` through Ollama |
| Embeddings | `nomic-embed-text:latest`, 768 dimensions |
| Retrieval | Persistent Chroma, cosine similarity |
| Orchestration and integrations | LangGraph and LangChain |
| Output validation and interface | Pydantic and Streamlit |

Both chat calls use temperature `0`, thinking disabled, a 512-token output limit, and a 120-second timeout. These baseline settings keep generation bounded; they do not guarantee correct or identical answers.

## Key design decisions

```text
Classify → Retrieve Top-K → Assess priority/confidence → Select queue
         → Draft notes → Flag for review or finish
```

- **Separate responsibilities:** LangGraph connects explicit steps. Qwen handles classification and drafting; Python handles voting, queue lookup, and review decisions.
- **Taxonomy-based classification:** full category descriptions provide context beyond keywords; Pydantic restricts the allowed output labels.
- **One document per inquiry:** short historical texts need no splitting. Labels remain metadata, and Nomic query/document prefixes support semantic retrieval across categories.
- **Evidence-based priority:** matching-category cases vote using clipped similarity weights. The largest total wins; ties favor higher urgency. This limits unrelated-category influence, but several weaker matches can outweigh a stronger one.
- **Deterministic routing:** each category maps to one queue in the CSV, avoiding generated team names.
- **Suggested resolutions:** a second model call uses the inquiry, decisions, and matching-category context. The CSV has no verified fixes; the one or two notes are generated suggestions.

### Confidence and review

```text
confidence = similarity strength × category agreement × priority agreement
```

Similarity strength is the mean of similarities individually clipped to [0, 1] across all retrieved cases. Category agreement is the fraction with positive weight matching the prediction. Priority agreement is the winning priority's share of matching-category weight.

Multiplication penalizes weak or conflicting evidence without another model call, but often produces low scores. An average or similarity-only score is less conservative and can hide disagreement. **This score is not a calibrated probability and does not assess note quality.**

Review is flagged when confidence is below the selected threshold. Missing positive matching evidence always forces review and a `medium` priority placeholder. The flag does not contact a reviewer or pause for approval.

## Current limitations

- Priority does not independently interpret urgency or negation in the original inquiry; urgent cases can be underrated and routine cases overrated.
- Notes may assume missing facts, give unsuitable advice, or fulfill unrelated requests. Schema validation checks structure, not correctness.
- Conservative confidence can flag most inquiries; reducing the threshold does not correct the underlying decisions.
- No production accuracy or time-saving claim is made. Urgency handling, improved resolution context, and calibrated confidence remain future work.

For module responsibilities, the backend contract, and a worked confidence example, see the [architecture and code guide](docs/README.md).
