"""All prompts in one place (benchmark construction, systems/agents, judges)."""

# =============================================================================
# Benchmark construction (run once with the judge model, shared by all policies)
# =============================================================================
QGEN_SYSTEM = "You design evaluation questions for a benchmark of scientific literature synthesis. Reply with JSON only."

QGEN_USER = """Below are the title and abstract of a research paper, followed by the works it cites (one line each).

TITLE: {title}
ABSTRACT: {abstract}

CITED WORKS:
{cited}

Generate up to {k} open-ended research questions about the specific research SUB-AREA that these cited works collectively address.
Each question MUST:
1. Be answerable only by finding and comparing several specific studies (at least 3 of the cited works) - not from general knowledge and not from this paper alone.
2. Target a FOCUSED sub-topic, NOT a whole field.
   Bad (too broad): "What are the challenges of large language models?"
   Good (focused): "How do retrieval-based methods reduce hallucination in long-form question answering, and how do they compare?"
3. NOT name any specific method, model, dataset, tool or acronym mentioned in the abstract. Ask about the problem they address.
4. Require specific research findings to answer well - not textbook knowledge.
5. Be open-ended and comparative ("What are the main approaches to ...", "How do methods for ... compare ...", "What are the trade-offs in ...").
6. Not be a yes/no or factoid question.

For each question also give:
- "breadth": "focused" or "field" (be honest: "field" if it could be the title of a whole survey of a big area)
- "general_knowledge": "yes" if a strong LLM could answer it well from memory without reading these works, else "no"
- "relevant_cited": the numbers of the cited works that are needed to answer it

Return JSON: {{"questions": [{{"question": "...", "breadth": "focused", "general_knowledge": "no", "relevant_cited": [1, 4, 7]}}]}}"""

KP_SYSTEM = "You extract gold evaluation points for scientific literature synthesis. Be specific and faithful to the documents. Reply with JSON only."

KP_USER_QA = """RESEARCH QUESTION: {question}

DOCUMENTS (cited studies):
{docs}

1. List "relevant_doc_ids": the documents that are relevant to answering the question.
2. Extract {min_kp}-{max_kp} atomic KEY POINTS that an excellent, evidence-grounded synthesis answering the question should contain.
   Each key point is ONE sentence stating a specific finding, method idea, comparison, trade-off, limitation or disagreement
   (prefer cross-study comparisons and disagreements over single-paper facts). Paraphrase; do not copy sentences.
   Give the supporting doc ids and an importance: 3 = central, 2 = important, 1 = peripheral.

Return JSON: {{"relevant_doc_ids": ["D1", ...], "key_points": [{{"text": "...", "doc_ids": ["D1"], "importance": 3}}]}}"""

KP_USER_RW = """TARGET PAPER
TITLE: {title}
ABSTRACT: {abstract}

DOCUMENTS (prior work cited by the target paper):
{docs}

Extract {min_kp}-{max_kp} atomic KEY POINTS that an excellent Related Work section for the target paper should contain:
specific contributions/findings of the prior work, how lines of prior work compare, their limitations, and how the target
paper relates to or differs from them. One sentence each, paraphrased, with supporting doc ids and importance
(3 = central, 2 = important, 1 = peripheral).

Return JSON: {{"relevant_doc_ids": ["D1", ...], "key_points": [{{"text": "...", "doc_ids": ["D1"], "importance": 3}}]}}"""

RWPOINTS_SYSTEM = "You summarise human-written related-work sections into atomic content points. Reply with JSON only."
RWPOINTS_USER = """TARGET PAPER TITLE: {title}

HUMAN-WRITTEN RELATED WORK SECTION:
{related_work}

Extract {min_kp}-{max_kp} atomic content points (themes of prior work, specific claims about prior work, and how the target
paper positions itself). One sentence each, paraphrased, with importance (3 = central, 2 = important, 1 = peripheral).
Return JSON: {{"points": [{{"text": "...", "importance": 2}}]}}"""

# =============================================================================
# Systems (policies)
# =============================================================================
QA_TASK = "RESEARCH QUESTION: {question}"
RW_TASK = "TARGET PAPER\nTITLE: {title}\nABSTRACT: {abstract}"
RW_TASK_INTRO = "TARGET PAPER\nTITLE: {title}\nABSTRACT: {abstract}\nINTRODUCTION (excerpt): {intro}"

WRITE_RULES_QA = (
    "Write a concise, well-organized literature synthesis ({min_w}-{max_w} words) that answers the research question. "
    "Compare and contrast approaches and findings across studies (do not summarize papers one by one), and mention "
    "disagreements, limitations, or conditions under which findings differ."
)
WRITE_RULES_RW = (
    "Write the Related Work section for the target paper ({min_w}-{max_w} words). Group prior work by theme, compare lines "
    "of work, and explain how the target paper builds on or differs from them."
)
CITE_RULES = (
    "Ground every factual claim in the provided documents and cite them with their IDs in square brackets, e.g. [D12] "
    "or [D3, D17]. Only cite IDs that appear in the provided documents. No title, no bullet lists, no reference list."
)
NOCITE_RULES = (
    "Rely on your own knowledge. Refer to specific studies (first author and year) only when you are confident they "
    "exist; never invent references or results. No title, no bullet lists, no reference list."
)

ZERO_SHOT_SYSTEM = "You are an expert scientist writing scholarly text. {rules} {nocite}"
COT_SYSTEM = (
    "You are an expert scientist writing scholarly text. {rules} {nocite} First write a brief outline inside "
    "<plan>...</plan> (themes, key studies you know, how they compare). Then write the final text inside <answer>...</answer>."
)
RAG_SYSTEM = "You are an expert scientist writing scholarly text from retrieved documents. {rules} {cite}"
RAG_USER = "{task}\n\nDOCUMENTS:\n{docs}\n\nWrite the text now."

REACT_SYSTEM = (
    "You are a research agent with access to a search engine over a database of scientific papers.\n"
    "Each turn, do exactly ONE of the following:\n"
    "  (a) search: write one or two short keyword queries, each inside <search>...</search>; you will receive results "
    "as [ID] snippets;\n"
    "  (b) answer: when you have enough evidence (or no searches are left), write the final text inside <answer>...</answer>.\n"
    "Search for different aspects (competing methods, findings, limitations), not the same query twice. "
    "Final text requirements: {rules} {cite}"
)

SEARCHER_SYSTEM = (
    "You are the SEARCHER in a team of three agents (Searcher -> Reader -> Synthesizer) that writes evidence-grounded "
    "scientific syntheses. You query a search engine over a database of scientific papers. Each turn, write one or two "
    "short keyword queries, each inside <search>...</search>. After seeing the results, search for aspects not yet "
    "covered (alternative methods, competing or contradicting findings, limitations, applications) rather than "
    "repeating queries. When you have collected enough evidence, reply with <done/>. Output only <search> tags or <done/>."
)
SEARCHER_USER = "{task}\n\nYou may run up to {n_turns} search turns. Start searching."
SEARCH_OBS = "Results for <search>{query}</search>:\n{results}"
SEARCH_OBS_FOOTER = "\n\nSearch turns left: {left}. Search again, or reply <done/>."

READER_SYSTEM = (
    "You are the READER in a team of three agents (Searcher -> Reader -> Synthesizer). You receive the task and the "
    "documents found by the Searcher. Extract the evidence the Synthesizer needs. For each RELEVANT document write one "
    "or two lines of the form\n[D#] <specific finding, method or result relevant to the task, in your own words>\n"
    "Skip irrelevant documents. Never state anything the document does not say. After the evidence lines, write one "
    "line starting with 'CONFLICTS:' that names disagreements or differing conditions between documents (or 'none')."
)
READER_USER = "{task}\n\nDOCUMENTS:\n{docs}\n\nWrite the evidence notes now."

SYNTH_SYSTEM = "You are the SYNTHESIZER in a team of three agents (Searcher -> Reader -> Synthesizer). {rules} {cite}"
SYNTH_USER = "{task}\n\nEVIDENCE NOTES FROM THE READER:\n{notes}\n\nDOCUMENTS (for reference):\n{docs}\n\nWrite the text now."

# =============================================================================
# Judges (reward + evaluation)
# =============================================================================
SUPPORT_SYSTEM = (
    "You are a meticulous scientific fact-checker. For each numbered claim, decide whether the CITED documents "
    "support it. 'supported': the claim's content is stated in or directly implied by at least one cited document. "
    "'partial': only part is supported, or the claim overgeneralizes/exaggerates. 'unsupported': not supported by the "
    "cited documents (including citations to irrelevant documents). Judge only against the documents, not your own "
    "knowledge. Reply with JSON only."
)
SUPPORT_USER = """DOCUMENTS:
{docs}

CLAIMS:
{claims}

Return JSON: {{"results": [{{"id": 1, "label": "supported|partial|unsupported"}}]}}"""

GROUND_SYSTEM = (
    "You are a meticulous scientific fact-checker. For each numbered claim, decide whether it is backed by the "
    "REFERENCE documents: 'supported' (stated in or directly implied by at least one document), 'contradicted' "
    "(conflicts with a document), or 'not_found' (not addressed by the documents). Generic background statements that "
    "need no evidence should be labelled 'generic'. Reply with JSON only."
)
GROUND_USER = """REFERENCE DOCUMENTS:
{docs}

CLAIMS:
{claims}

Return JSON: {{"results": [{{"id": 1, "label": "supported|contradicted|not_found|generic"}}]}}"""

COVER_SYSTEM = (
    "You evaluate whether a text conveys given key points. 'yes': the point (its specific content) is clearly "
    "conveyed, even if worded differently. 'partial': only partly or vaguely conveyed. 'no': absent or contradicted. "
    "Reply with JSON only."
)
COVER_USER = """TEXT:
{answer}

KEY POINTS:
{points}

Return JSON: {{"results": [{{"id": 1, "label": "yes|partial|no"}}]}}"""

NOTES_SYSTEM = (
    "You verify evidence notes. For each numbered note, decide if the referenced document supports it: 'supported', "
    "'partial', or 'unsupported'. Judge only against the document text. Reply with JSON only."
)
NOTES_USER = """DOCUMENTS:
{docs}

NOTES:
{notes}

Return JSON: {{"results": [{{"id": 1, "label": "supported|partial|unsupported"}}]}}"""

HOLISTIC_SYSTEM = (
    "You are an expert reviewer of scientific writing. Score the text on each criterion from 1 (very poor) to 10 "
    "(excellent). Be strict and calibrated; a fluent but generic text should not exceed 5 on synthesis or specificity. "
    "Reply with JSON only."
)
HOLISTIC_USER = """TASK:
{task}

TEXT TO EVALUATE:
{answer}

Criteria:
- synthesis: compares/contrasts studies and integrates findings rather than listing papers one by one
- specificity: concrete methods, findings and conditions rather than vague generalities
- coverage: addresses the main aspects the task requires
- critical: identifies disagreements, limitations, open problems
- organization: logical structure and flow
{extra}
Return JSON: {{"synthesis": s, "specificity": s, "coverage": s, "critical": s, "organization": s{extra_key}, "overall": s}}"""
HOLISTIC_EXTRA_RW = "- positioning: explains how the TARGET paper relates to / differs from the prior work\n"
