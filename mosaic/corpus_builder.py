from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import dump_json, slugify, write_text

COMPANIES = [
    {"name": "Acadia Securities", "desk": "energy derivatives", "region": "the Gulf Coast", "client_base": "utilities and transport hedgers"},
    {"name": "Bluewater Capital", "desk": "rates options", "region": "the Northeast corridor", "client_base": "regional banks and insurers"},
    {"name": "Cedar Harbor Markets", "desk": "foreign exchange", "region": "the Pacific Rim", "client_base": "exporters and shipping firms"},
    {"name": "Driftline Bancorp", "desk": "credit index trading", "region": "the Mid-Atlantic", "client_base": "asset managers and pensions"},
    {"name": "Eastgate Financial", "desk": "equities execution", "region": "the Great Lakes", "client_base": "corporate treasury desks"},
    {"name": "Falcon Ridge Advisors", "desk": "macro futures", "region": "Texas and the Southwest", "client_base": "commodity producers"},
    {"name": "Granite Peak Markets", "desk": "commodity spreads", "region": "the Mountain West", "client_base": "industrial hedgers"},
    {"name": "Harborlight Securities", "desk": "structured products", "region": "the Southeast", "client_base": "wealth platforms"},
    {"name": "Ironwood Capital", "desk": "ETF market making", "region": "the Midwest", "client_base": "retail aggregators"},
    {"name": "Juniper Street Partners", "desk": "municipal finance", "region": "California", "client_base": "public issuers and banks"},
]

RISK_FOCI = [
    "concentrated exposure to two regional lenders",
    "basis mismatch in long-dated power hedges",
    "settlement timing pressure from Asian shipping clients",
    "spread widening in lower-tier industrial credits",
    "single-name concentration in semiconductor collateral",
    "margin volatility in weather-linked futures",
    "inventory financing dependence among metals clients",
    "valuation drift in callable structured notes",
    "creation-redemption timing stress in thin ETFs",
    "dealer balance-sheet strain in municipal remarketing",
]

FREEZE_WORKFLOWS = [
    "voice-trade affirmation routing",
    "cross-border client onboarding",
    "late-day collateral substitutions",
    "manual spread override approvals",
    "after-hours basket rebalancing",
    "non-standard futures give-ups",
    "physical delivery exception handling",
    "structured-note term sheet issuance",
    "ETF create-redeem exception tickets",
    "municipal remarketing override entries",
]

REMEDIATION_OWNERS = [
    "the surveillance engineering lead",
    "the treasury controls program manager",
    "the international approvals team",
    "the desk supervision officer",
    "the market structure review lead",
    "the derivatives operations manager",
    "the commodities controls director",
    "the product governance chair",
    "the ETF workflow governance lead",
    "the public finance controls head",
]

OUTLOOK_SIGNALS = [
    "client hedging demand should remain above trend for another quarter",
    "rates clients may pause until curve volatility normalizes",
    "shipping-linked FX volumes should recover as settlement queues clear",
    "credit index volumes may rise while cash credit remains soft",
    "equities execution should normalize after index rebalance season",
    "macro futures activity should stay elevated through the next inflation print",
    "commodity spreads should widen if inventory financing remains tight",
    "structured-note issuance may recover only after valuation controls settle",
    "ETF market-making spreads should tighten once exception tickets clear",
    "municipal finance activity should improve if remarketing capacity stabilizes",
]

BOARD_PRIORITIES = [
    "compress duplicated market-risk language and highlight only truly new desk signals",
    "push a faster collateral telemetry rollout",
    "reduce settlement handoffs across the Asia booking chain",
    "tighten exception thresholds in lower-tier credit names",
    "prioritize concentration reporting for semiconductor clients",
    "accelerate margin forecasting for weather-linked contracts",
    "reprice financing terms for industrial hedgers",
    "slow structured-note issuance until governance metrics improve",
    "automate ETF exception handling before the next rebalance cycle",
    "strengthen remarketing oversight during summer issuance windows",
]

RECURRING_MARKET_RISK = (
    "Across rates, liquidity, counterparty exposure, and basis dislocation, management repeated the same core market-risk language: "
    "rapid moves in benchmark curves can reprice hedges, stressed liquidity can widen execution costs, counterparty weakness can delay settlements, "
    "and basis gaps between cash and derivatives books can create short-term earnings volatility even when client franchises remain healthy. "
    "The disclosure also repeated that these pressures matter most when client activity bunches into the same narrow trading window."
)

RECURRING_OPERATING_CONTEXT = (
    "The reports also reused a long operating-context section that described how treasury funding, client collateral, and supervisory review move together during heavy volume. "
    "That section repeated that desk heads escalate unusual flow clusters, treasury teams monitor intraday liquidity buffers, and control officers review manual interventions before close. "
    "Its purpose was mostly explanatory rather than informative, which makes it ideal redundancy for a context-selection benchmark because it appears nearly verbatim in many documents."
)

RECURRING_CONTROL_LANGUAGE = (
    "Compliance repeated the same control themes across memoranda: communications retention must be complete, approvals must be attributable, exception monitoring must be timely, "
    "and counterparty escalations must be documented before a workflow returns to steady state. The memo again noted that control remediation should reduce manual intervention, shorten review queues, "
    "and make surveillance evidence easier to audit across desks that reuse the same approval patterns."
)

RECURRING_MARKET_RISK_ANSWER = "The repeated market-risk factors were rates volatility, liquidity shocks, counterparty stress, and basis dislocation across client and treasury books."
RECURRING_CONTROL_ANSWER = "The recurring control themes were communications retention, approval hygiene, exception monitoring, and counterparty escalation discipline."

RECURRING_OPERATING_CONTEXT_ANSWER = "The repeated operating-context section said desk heads escalate unusual flow clusters, treasury monitors intraday liquidity buffers, and control officers review manual interventions before close."
TRUE_UNKNOWN_TOPICS = [
    "the orion lattice initiative",
    "its holographic asset vault",
    "the zephyr branch sunset ledger",
    "a quasar ransom amnesty",
    "the nebula carbon vault",
    "its synthetic avatar workforce",
    "the aurora origination lattice",
    "an atlas moonbase expansion",
    "the saffron quantum lattice",
]


@dataclass(slots=True)
class DocumentSpec:
    id: str
    title: str
    doc_type: str
    roles: list[str]
    body: str
    metadata: dict[str, Any]

    def to_catalog_entry(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("body")
        return payload


def _signed_percent(value: int) -> str:
    return f"{value}% increase" if value >= 0 else f"{abs(value)}% decline"


def _frontmatter(doc: DocumentSpec) -> str:
    lines = ["---", f"title: {doc.title}", f"roles: {', '.join(doc.roles)}", f"doc_type: {doc.doc_type}"]
    for key, value in doc.metadata.items():
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def _render_document(doc: DocumentSpec) -> str:
    return f"{_frontmatter(doc)}\n{doc.body.strip()}\n"


def _earnings_report(company: dict[str, str], company_index: int, quarter: int) -> DocumentSpec:
    slug = slugify(company["name"])
    revenue_change = (company_index * 3 + quarter * 2) - 2
    margin_change_bps = 8 + company_index * 3 + quarter * 5
    collateral_days = 2 + ((company_index + quarter) % 4)
    workflow = FREEZE_WORKFLOWS[company_index]
    owner = REMEDIATION_OWNERS[company_index]
    outlook = OUTLOOK_SIGNALS[company_index]
    body = "\n\n".join(
        [
            (
                f"{company['name']} said quarter {quarter} performance reflected steadier activity in {company['desk']} across {company['region']}. "
                f"Management told analysts that the desk benefited from repeat business from {company['client_base']}, although intra-quarter execution quality moved around with client batching patterns. "
                f"The filing highlighted a {_signed_percent(revenue_change)} in trading-desk revenue, a {margin_change_bps} basis-point lift in managed spread capture, and a collateral turn time that settled near {collateral_days} business days after exception review."
            ),
            RECURRING_MARKET_RISK,
            RECURRING_OPERATING_CONTEXT,
            (
                f"The unique section for quarter {quarter} said desk leaders concentrated on {company['desk']} workflow quality rather than raw volume growth. "
                f"Controllers noted that revenue held up because client hedges were repriced faster, error queues closed earlier, and treasury allocation meetings shortened after opening imbalance reports were standardized. "
                f"Management also disclosed that temporary controls around {workflow} kept some activity on a slower path while {owner} completed remediation and prepared a cleaner supervisory handoff."
            ),
            (
                f"Management closed by telling investors that {outlook}. The filing emphasized that better information density, not longer context packets, led to clearer desk decisions. "
                f"That framing is useful for the MOSAIC benchmark because the report contains one genuinely novel quarter-specific revenue fact surrounded by repeated explanatory language that offers very little marginal information after the first copy."
            ),
        ]
    )
    return DocumentSpec(
        id=f"{slug}_q{quarter}_earnings",
        title=f"{company['name']} Q{quarter} Earnings Report",
        doc_type="earnings_report",
        roles=["analyst"],
        body=body,
        metadata={
            "company": company["name"],
            "company_slug": slug,
            "quarter": quarter,
            "revenue_change_pct": revenue_change,
            "margin_change_bps": margin_change_bps,
            "workflow": workflow,
            "remediation_owner": owner,
            "outlook_signal": outlook,
        },
    )


def _risk_disclosure(company: dict[str, str], company_index: int, period: str) -> DocumentSpec:
    slug = slugify(company["name"])
    focus = RISK_FOCI[company_index]
    surveillance_window = "midyear" if period == "midyear" else "year-end"
    body = "\n\n".join(
        [
            (
                f"{company['name']} issued a {surveillance_window} risk disclosure focused on how repeated market-risk language can bury genuinely new risk detail. "
                f"The document explained that large sections were intentionally retained from prior periods so legal review stayed stable across cycles, even when only one or two facts truly changed."
            ),
            RECURRING_MARKET_RISK,
            RECURRING_OPERATING_CONTEXT,
            (
                f"The novel risk detail for this disclosure was {focus}. The filing said that supervisors ran a tighter watch on that issue because it could widen losses quickly if clients clustered their hedging flows into the same session. "
                f"It also noted that analysts should not confuse the recycled boilerplate with the real risk update, which was confined to a short section near the middle of the document."
            ),
            (
                f"The disclosure concluded that portfolio managers should separate duplicated market-risk phrasing from the one factual update about {focus}. "
                f"That exact structure is useful for the benchmark because a naive retriever can easily return several nearly identical risk paragraphs while skipping the compact period-specific detail."
            ),
        ]
    )
    return DocumentSpec(
        id=f"{slug}_{period}_risk_disclosure",
        title=f"{company['name']} {surveillance_window.title()} Risk Disclosure",
        doc_type="risk_disclosure",
        roles=["senior_analyst"],
        body=body,
        metadata={
            "company": company["name"],
            "company_slug": slug,
            "period": period,
            "risk_focus": focus,
        },
    )


def _compliance_memo(company: dict[str, str], company_index: int, period: str) -> DocumentSpec:
    slug = slugify(company["name"])
    workflow = FREEZE_WORKFLOWS[company_index]
    owner = REMEDIATION_OWNERS[company_index]
    due_week = 2 + company_index % 3
    body = "\n\n".join(
        [
            (
                f"Internal compliance reviewed {company['name']}'s controls around {company['desk']} during the {period.replace('_', ' ')} cycle. "
                f"The memo said queue times had become harder to interpret because repeated descriptive sections made it difficult for reviewers to spot the one workflow that was actually under restriction."
            ),
            RECURRING_CONTROL_LANGUAGE,
            (
                f"The unique action item was a temporary freeze on {workflow}. Compliance wrote that the restriction should remain in place until {owner} verified evidence quality, replay coverage, and exception attribution. "
                f"The memo targeted completion by week {due_week} of the remediation cycle and warned that manual workarounds would inflate review noise if the desk reintroduced the workflow too early."
            ),
            (
                f"The memo also cross-referenced the same repeated market-risk language seen in external reports, but its operational takeaway was much narrower: freeze {workflow}, complete the remediation checklist, and do not restore the workflow until supervisors could trace every intervention."
            ),
        ]
    )
    return DocumentSpec(
        id=f"{slug}_{period}_compliance_memo",
        title=f"{company['name']} {period.replace('_', ' ').title()} Compliance Memo",
        doc_type="compliance_memo",
        roles=["compliance"],
        body=body,
        metadata={
            "company": company["name"],
            "company_slug": slug,
            "period": period,
            "workflow": workflow,
            "remediation_owner": owner,
        },
    )


def _research_note(company: dict[str, str], company_index: int) -> DocumentSpec:
    slug = slugify(company["name"])
    outlook = OUTLOOK_SIGNALS[company_index]
    workflow = FREEZE_WORKFLOWS[company_index]
    owner = REMEDIATION_OWNERS[company_index]
    body = "\n\n".join(
        [
            (
                f"Desk researchers at {company['name']} wrote that the next quarter should hinge on whether {company['desk']} clients keep rolling hedges at the same cadence. "
                f"The note argued that flow quality mattered more than raw page count because analysts were already drowning in repeated context that restated the same risk backdrop."
            ),
            (
                f"The note's central view was that {outlook}. Analysts also wrote that the market was overreacting to duplicated boilerplate sections and underweighting shorter operational updates embedded elsewhere in the corpus."
            ),
            (
                f"In the only operational section, the researchers said temporary controls were still slowing {workflow} while remediation ran under {owner}. "
                f"They treated that as a drag on near-term throughput rather than a thesis-breaking problem, because the desk had already rerouted most urgent activity through supervised fallback lanes."
            ),
            (
                f"The conclusion urged readers to separate recurring descriptive language from fresh signal. That framing gives MOSAIC a useful analyst-visible alternative source when a query refers obliquely to the compliance issue but the user lacks direct memo access."
            ),
        ]
    )
    return DocumentSpec(
        id=f"{slug}_research_note",
        title=f"{company['name']} Trading Desk Research Note",
        doc_type="research_note",
        roles=["analyst"],
        body=body,
        metadata={
            "company": company["name"],
            "company_slug": slug,
            "outlook_signal": outlook,
            "workflow": workflow,
            "remediation_owner": owner,
        },
    )


def _executive_briefing(company: dict[str, str], company_index: int) -> DocumentSpec:
    slug = slugify(company["name"])
    priority = BOARD_PRIORITIES[company_index]
    workflow = FREEZE_WORKFLOWS[company_index]
    body = "\n\n".join(
        [
            (
                f"This executive briefing summarized {company['name']}'s year-to-date story for leadership readers who wanted only the highest-signal updates. "
                f"It explicitly warned that the supporting corpus reused long market-risk explanations, so executives should not mistake longer packets for broader information coverage."
            ),
            (
                f"The briefing highlighted one board priority: {priority}. It also noted that repeated market-risk phrasing about rates, liquidity, counterparties, and basis gaps appeared throughout the broader corpus, but only a handful of operational facts were actually new."
            ),
            (
                f"Leadership also received a sanitized operational note that temporary restrictions around {workflow} were still flowing through monitored fallback processes. "
                f"The briefing did not expose the full compliance memo, but it preserved enough signal for users without compliance access to understand the business impact."
            ),
            (
                f"Because the briefing condenses several source types into one high-level view, it helps demonstrate the difference between permission-aware retrieval and naive top-k retrieval that may over-select duplicated supporting language."
            ),
        ]
    )
    return DocumentSpec(
        id=f"{slug}_executive_briefing",
        title=f"{company['name']} Executive Briefing",
        doc_type="executive_briefing",
        roles=["public", "executive"],
        body=body,
        metadata={
            "company": company["name"],
            "company_slug": slug,
            "board_priority": priority,
            "workflow": workflow,
        },
    )


def build_corpus_documents() -> list[DocumentSpec]:
    documents: list[DocumentSpec] = []
    for company_index, company in enumerate(COMPANIES):
        for quarter in range(1, 5):
            documents.append(_earnings_report(company, company_index, quarter))
        for period in ("midyear", "yearend"):
            documents.append(_risk_disclosure(company, company_index, period))
        for period in ("ops_review", "surveillance_followup"):
            documents.append(_compliance_memo(company, company_index, period))
        documents.append(_research_note(company, company_index))
        documents.append(_executive_briefing(company, company_index))
    return documents


def generate_corpus(
    output_dir: str | Path,
    catalog_path: str | Path | None = None,
    clean: bool = False,
) -> list[dict[str, Any]]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for existing in target_dir.rglob("*.md"):
            existing.unlink()

    documents = build_corpus_documents()
    for document in documents:
        write_text(target_dir / f"{document.id}.md", _render_document(document))

    catalog = [document.to_catalog_entry() for document in documents]
    if catalog_path is not None:
        dump_json(catalog_path, catalog)
    return catalog
def _catalog_by_company(catalog: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_company: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in catalog:
        company_slug = str(entry["metadata"]["company_slug"])
        by_company.setdefault(company_slug, {}).setdefault(entry["doc_type"], []).append(entry)

    for company_docs in by_company.values():
        for docs in company_docs.values():
            docs.sort(key=lambda item: item["id"])
    return by_company


def _find_doc(docs: list[dict[str, Any]], **metadata_match: str | int) -> dict[str, Any]:
    for doc in docs:
        if all(doc["metadata"].get(key) == value for key, value in metadata_match.items()):
            return doc
    raise KeyError(f"No document matched {metadata_match}")


def _append_redundancy_queries(queries: list[dict[str, Any]], by_company: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    for index, company in enumerate(COMPANIES):
        slug = slugify(company["name"])
        earnings = by_company[slug]["earnings_report"]
        risk_midyear = _find_doc(by_company[slug]["risk_disclosure"], period="midyear")
        risk_yearend = _find_doc(by_company[slug]["risk_disclosure"], period="yearend")
        compliance_ops = _find_doc(by_company[slug]["compliance_memo"], period="ops_review")
        compliance_followup = _find_doc(by_company[slug]["compliance_memo"], period="surveillance_followup")
        q1_earnings = _find_doc(earnings, quarter=1)
        q4_earnings = _find_doc(earnings, quarter=4)

        queries.append(
            {
                "id": f"rt_market_{slug}",
                "category": "redundancy_trap",
                "query": f"Which recurring market-risk factors are repeated across the filings for {company['name']}?",
                "ground_truth_answer": RECURRING_MARKET_RISK_ANSWER,
                "source_doc_ids": [q4_earnings["id"], risk_yearend["id"]],
                "user_roles": ["senior_analyst"],
                "notes": "Near-identical market-risk boilerplate appears across multiple earnings and risk documents.",
            }
        )
        queries.append(
            {
                "id": f"rt_controls_{slug}",
                "category": "redundancy_trap",
                "query": f"What control themes recur across the compliance memoranda for {company['name']}?",
                "ground_truth_answer": RECURRING_CONTROL_ANSWER,
                "source_doc_ids": [compliance_ops["id"], compliance_followup["id"]],
                "user_roles": ["compliance"],
                "notes": "Repeated compliance boilerplate should collapse to one chunk, not many copies.",
            }
        )
        if index < 5:
            queries.append(
                {
                    "id": f"rt_ops_{slug}",
                    "category": "redundancy_trap",
                    "query": f"What repeated operating-context guidance shows up across {company['name']}'s earnings and risk filings?",
                    "ground_truth_answer": RECURRING_OPERATING_CONTEXT_ANSWER,
                    "source_doc_ids": [q1_earnings["id"], risk_midyear["id"]],
                    "user_roles": ["senior_analyst"],
                    "notes": "The operating-context section is intentionally duplicated and should not crowd out novel signal.",
                }
            )

def _append_multi_hop_queries(queries: list[dict[str, Any]], by_company: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    for company in COMPANIES:
        slug = slugify(company["name"])
        earnings = by_company[slug]["earnings_report"]
        research = by_company[slug]["research_note"][0]
        risk_midyear = _find_doc(by_company[slug]["risk_disclosure"], period="midyear")
        q2_earnings = _find_doc(earnings, quarter=2)
        q3_earnings = _find_doc(earnings, quarter=3)
        q4_earnings = _find_doc(earnings, quarter=4)

        queries.append(
            {
                "id": f"mh_outlook_{slug}",
                "category": "multi_hop",
                "query": f"For {company['name']}, what was the Q4 trading-desk revenue move and what outlook did the research note give?",
                "ground_truth_answer": f"{company['name']} reported a {q4_earnings['metadata']['revenue_change_pct']}% Q4 trading-desk revenue move, and the research note said {research['metadata']['outlook_signal']}.",
                "source_doc_ids": [q4_earnings["id"], research["id"]],
                "user_roles": ["analyst"],
                "notes": "Requires earnings plus analyst research rather than repeated chunks from one report.",
            }
        )
        queries.append(
            {
                "id": f"mh_compare_{slug}",
                "category": "multi_hop",
                "query": f"For {company['name']}, what Q2 revenue change coincided with the midyear risk disclosure, and how did Q3 compare?",
                "ground_truth_answer": f"{company['name']} paired a {q2_earnings['metadata']['revenue_change_pct']}% Q2 revenue move with a risk disclosure focused on {risk_midyear['metadata']['risk_focus']}, and Q3 revenue later moved {q3_earnings['metadata']['revenue_change_pct']}%.",
                "source_doc_ids": [q2_earnings["id"], q3_earnings["id"], risk_midyear["id"]],
                "user_roles": ["senior_analyst"],
                "notes": "Requires crossing earnings quarters with the risk disclosure instead of over-selecting one document family.",
            }
        )


def _append_failure_queries(queries: list[dict[str, Any]], by_company: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    for index, company in enumerate(COMPANIES[:9]):
        slug = slugify(company["name"])
        compliance_ops = _find_doc(by_company[slug]["compliance_memo"], period="ops_review")
        risk_midyear = _find_doc(by_company[slug]["risk_disclosure"], period="midyear")

        if index < 5:
            queries.append(
                {
                    "id": f"fc_permission_{slug}",
                    "category": "failure_classification",
                    "query": f"What workflow did the operations review compliance memo place under a temporary freeze for {company['name']}?",
                    "ground_truth_type": "PERMISSION_GAP",
                    "required_role": "compliance",
                    "ground_truth_answer": None,
                    "source_doc_ids": [compliance_ops["id"]],
                    "user_roles": ["analyst"],
                    "notes": "The answer exists only in a compliance-gated memo; analyst-visible summaries are incomplete.",
                }
            )
        else:
            queries.append(
                {
                    "id": f"fc_permission_{slug}",
                    "category": "failure_classification",
                    "query": f"Which specific midyear risk focus was flagged for {company['name']} in the restricted disclosure set?",
                    "ground_truth_type": "PERMISSION_GAP",
                    "required_role": "senior_analyst",
                    "ground_truth_answer": None,
                    "source_doc_ids": [risk_midyear["id"]],
                    "user_roles": ["analyst"],
                    "notes": "The precise risk update exists but sits above the analyst role.",
                }
            )

    for index, company in enumerate(COMPANIES[:9]):
        slug = slugify(company["name"])
        topic = TRUE_UNKNOWN_TOPICS[index]
        queries.append(
            {
                "id": f"fc_unknown_{slug}",
                "category": "failure_classification",
                "query": f"What did {company['name']} disclose about {topic}?",
                "ground_truth_type": "TRUE_UNKNOWN",
                "ground_truth_answer": None,
                "source_doc_ids": [],
                "user_roles": ["analyst"],
                "notes": "This topic is intentionally absent from the corpus.",
            }
        )

    retrieval_templates = [
        (
            "Which workflow lane stayed on a supervisory throughput throttle for {name}?",
            "Analyst-visible materials said temporary controls were slowing {workflow} while remediation ran under {owner}.",
            "research_note",
            ["analyst"],
        ),
        (
            "What compact board directive did leadership keep emphasizing for {name}?",
            "Leadership kept emphasizing the board priority to {priority}.",
            "executive_briefing",
            ["analyst"],
        ),
        (
            "Which risk focus was the seasonal surveillance package really worried about for {name}?",
            "The period-specific risk focus was {risk_focus}.",
            "risk_disclosure",
            ["senior_analyst"],
        ),
    ]

    for index, company in enumerate(COMPANIES[:7]):
        slug = slugify(company["name"])
        research = by_company[slug]["research_note"][0]
        briefing = by_company[slug]["executive_briefing"][0]
        risk_midyear = _find_doc(by_company[slug]["risk_disclosure"], period="midyear")
        template, answer_template, target_type, user_roles = retrieval_templates[index % len(retrieval_templates)]
        if target_type == "research_note":
            answer = answer_template.format(workflow=research["metadata"]["workflow"], owner=research["metadata"]["remediation_owner"])
            source_ids = [research["id"], briefing["id"]]
        elif target_type == "executive_briefing":
            answer = answer_template.format(priority=briefing["metadata"]["board_priority"])
            source_ids = [briefing["id"]]
        else:
            answer = answer_template.format(risk_focus=risk_midyear["metadata"]["risk_focus"])
            source_ids = [risk_midyear["id"]]

        queries.append(
            {
                "id": f"fc_retrieval_{slug}",
                "category": "failure_classification",
                "query": template.format(name=company["name"]),
                "ground_truth_type": "RETRIEVAL_FAILURE",
                "ground_truth_answer": answer,
                "source_doc_ids": source_ids,
                "user_roles": list(user_roles),
                "notes": "The answer is accessible, but the phrasing intentionally mismatches the corpus wording.",
            }
        )


def _append_multi_turn_queries(queries: list[dict[str, Any]], by_company: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    for company in COMPANIES:
        slug = slugify(company["name"])
        scenario_id = f"scenario_{slug}"
        earnings = by_company[slug]["earnings_report"]
        research = by_company[slug]["research_note"][0]
        briefing = by_company[slug]["executive_briefing"][0]
        risk_midyear = _find_doc(by_company[slug]["risk_disclosure"], period="midyear")
        risk_yearend = _find_doc(by_company[slug]["risk_disclosure"], period="yearend")
        q1_earnings = _find_doc(earnings, quarter=1)
        q2_earnings = _find_doc(earnings, quarter=2)
        q3_earnings = _find_doc(earnings, quarter=3)
        q4_earnings = _find_doc(earnings, quarter=4)

        turns = [
            {
                "turn": 1,
                "query": f"For {company['name']}, what Q1 revenue change and research outlook opened the year?",
                "ground_truth_answer": f"{company['name']} opened the year with a {q1_earnings['metadata']['revenue_change_pct']}% Q1 revenue move, and the research note said {research['metadata']['outlook_signal']}.",
                "source_doc_ids": [q1_earnings["id"], research["id"]],
            },
            {
                "turn": 2,
                "query": f"Which repeated market-risk factors kept appearing, and what midyear risk focus mattered most for {company['name']}?",
                "ground_truth_answer": f"The repeated risks were rates volatility, liquidity shocks, counterparty stress, and basis dislocation, while the midyear risk focus was {risk_midyear['metadata']['risk_focus']}.",
                "source_doc_ids": [q2_earnings["id"], risk_midyear["id"]],
            },
            {
                "turn": 3,
                "query": f"How did Q3 compare with Q2 on trading-desk revenue for {company['name']}?",
                "ground_truth_answer": f"Q2 revenue moved {q2_earnings['metadata']['revenue_change_pct']}%, and Q3 revenue moved {q3_earnings['metadata']['revenue_change_pct']}%.",
                "source_doc_ids": [q2_earnings["id"], q3_earnings["id"]],
            },
            {
                "turn": 4,
                "query": f"What board priority and workflow slowdown did leadership highlight for {company['name']}?",
                "ground_truth_answer": f"Leadership highlighted the board priority to {briefing['metadata']['board_priority']}, and analyst-visible materials said temporary controls were slowing {research['metadata']['workflow']}.",
                "source_doc_ids": [briefing["id"], research["id"]],
            },
            {
                "turn": 5,
                "query": f"What year-end risk focus and Q4 revenue move closed the year for {company['name']}?",
                "ground_truth_answer": f"The year-end risk focus was {risk_yearend['metadata']['risk_focus']}, and Q4 revenue moved {q4_earnings['metadata']['revenue_change_pct']}%.",
                "source_doc_ids": [risk_yearend["id"], q4_earnings["id"]],
            },
        ]

        for turn in turns:
            queries.append(
                {
                    "id": f"mt_{slug}_{turn['turn']:02d}",
                    "category": "multi_turn",
                    "scenario_id": scenario_id,
                    "turn": turn["turn"],
                    "query": turn["query"],
                    "ground_truth_answer": turn["ground_truth_answer"],
                    "source_doc_ids": turn["source_doc_ids"],
                    "user_roles": ["senior_analyst"],
                    "notes": f"Turn {turn['turn']} of a five-turn analyst review scenario.",
                }
            )


def generate_eval_suite(catalog: list[dict[str, Any]], output_path: str | Path | None = None) -> dict[str, Any]:
    by_company = _catalog_by_company(catalog)
    queries: list[dict[str, Any]] = []

    _append_redundancy_queries(queries, by_company)
    _append_multi_hop_queries(queries, by_company)
    _append_failure_queries(queries, by_company)
    _append_multi_turn_queries(queries, by_company)

    payload = {
        "metadata": {
            "total_queries": len(queries),
            "categories": ["redundancy_trap", "multi_hop", "failure_classification", "multi_turn"],
            "baselines": ["topk", "mmr", "mosaic_no_ledger", "mosaic_full"],
        },
        "queries": queries,
    }
    if output_path is not None:
        dump_json(output_path, payload)
    return payload
