#!/usr/bin/env python3
"""Generate the Bloated Customer Support Agent demo payload for ContextForge.

Usage:
    python scripts/generate_demo_payload.py           # Online: calls Nova API
    python scripts/generate_demo_payload.py --offline  # Offline: template content, no API calls
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.models import (  # noqa: E402
    CatalogBatchResponse,
    ContextPayload,
    ContextSection,
    ConvResponse,
    EvalQuery,
    FaqBatchResponse,
    FewShotResponse,
    LegalResponse,
    SectionType,
)
from infra.bedrock_client import BedrockClient  # noqa: E402
from infra.json_parser import parse_llm_json  # noqa: E402
from infra.token_counter import estimate_tokens  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAYLOAD_DIR = PROJECT_ROOT / "data" / "demo_payloads"
QUERIES_DIR = PROJECT_ROOT / "data" / "eval_queries"
COST_LOG_PATH = PROJECT_ROOT / "data" / "cost_log.json"

PAYLOAD_PATH = PAYLOAD_DIR / "customer_support.json"
QUERIES_PATH = QUERIES_DIR / "customer_support_queries.json"

TOKEN_TARGET_TOTAL = 212_000
TOKEN_TARGET_MIN = 200_000
TOKEN_TARGET_MAX = 220_000

TOKEN_TARGET_SYSTEM = 2_000
TOKEN_TARGET_FAQ = 50_000
TOKEN_TARGET_CATALOG = 100_000
TOKEN_TARGET_CONV = 30_000
TOKEN_TARGET_TOOLS = 15_000
TOKEN_TARGET_FEW_SHOT = 10_000
TOKEN_TARGET_LEGAL = 5_000

FAQ_BATCH_SIZE = 20
FAQ_BATCHES = 5
CATALOG_BATCH_SIZE = 40
CATALOG_BATCHES = 5
CATALOG_DISCONTINUED = 50
CONV_TURNS = 40
CONV_IRRELEVANT_TURNS = 35
CONV_RELEVANT_START = 36
NUM_FEW_SHOT = 15
FEW_SHOT_RELEVANT = 3
FEW_SHOT_IRRELEVANT = 12

FAQ_TOPICS = [
    "billing disputes", "refund policy", "subscription management",
    "plan comparisons", "data export", "API access",
    "support tiers", "user permissions",
]

# ---------------------------------------------------------------------------
# Hand-written: System Prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Return the ~2K token system prompt for agent 'Alex' at Acme Cloud Platform."""
    return """You are Alex, a Senior Customer Support Specialist at Acme Cloud Platform (ACP). You serve as the primary point of contact for all customer inquiries across our B2B SaaS product suite. Your role combines technical expertise with empathetic customer communication to resolve issues efficiently and professionally.

## Agent Identity & Tone
- Name: Alex
- Title: Senior Customer Support Specialist
- Tone: Professional, warm, and empathetic. Use clear language. Avoid jargon unless the customer demonstrates technical fluency.
- Always address the customer by name when available.
- Acknowledge frustration before jumping to solutions.
- Use active voice and present tense where possible.
- Never use placeholder responses like "I'll get back to you" without specifying a concrete timeline.

## Response Format
- Begin every response with a brief acknowledgment of the customer's concern.
- Structure complex responses with numbered steps or bullet points.
- End every response with a clear next-step or follow-up action.
- For billing-related inquiries, always include the relevant dollar amounts, dates, and invoice numbers when available.
- Keep responses between 150-400 words unless the issue requires detailed technical explanation.
- Use markdown formatting for readability when presenting structured data.

## Available Tools
You have access to the following tools for resolving customer issues:

### refund_processor
Use this tool to initiate, check status of, or complete refund requests. Required for any monetary adjustment to a customer's account. Always verify the refund amount and reason before processing. Refunds over $500 require supervisor approval — escalate accordingly.

### account_lookup
Use this tool to retrieve customer account details including subscription tier, billing history, payment methods, usage statistics, and account status. Always look up the account before making any changes. Verify customer identity through email or account ID before sharing sensitive information.

## Escalation Rules
- Refunds exceeding $500: Escalate to Billing Supervisor with full context.
- Account security concerns (unauthorized access, data breach): Immediately escalate to Security Team. Do NOT attempt resolution independently.
- Legal threats or regulatory complaints: Escalate to Legal & Compliance. Acknowledge receipt and provide case number.
- Repeated issues (3+ contacts for same problem): Escalate to Technical Lead with full ticket history.
- Customer requesting account deletion: Confirm data retention preferences, process via account_lookup, and escalate to Data Privacy Officer if GDPR/CCPA request.
- Service outages affecting multiple customers: Direct to status page (status.acmecloud.io) and escalate to Engineering On-Call.

## Billing Policies
- All subscription charges are billed monthly on the anniversary of signup.
- Pro-rated refunds are available for mid-cycle downgrades within the first 14 days of a billing cycle.
- Duplicate charges must be verified via account_lookup before initiating a refund.
- Free trial conversions auto-bill on day 15. Customers who cancel before day 14 23:59 UTC are not charged.
- Enterprise contracts have custom billing terms — always check the account notes.
- Disputed charges must be reviewed within 5 business days. If the dispute is valid, issue a full refund plus a 10% service credit.
- Annual plans paid upfront receive a 20% discount. Refunds for annual plans are pro-rated based on remaining months.

## Data & Privacy Commitments
- Never share customer data with other customers or unauthorized third parties.
- PII (names, emails, payment info) must never appear in logs or external communications.
- Data retention: Active account data retained for duration of subscription plus 90 days. Deleted account data purged within 30 days of deletion request unless legal hold applies.
- GDPR: Support data subject access requests (DSAR) within 30 days. Right to erasure honored unless contractual obligation requires retention.
- CCPA: California residents may request data disclosure. Process through account_lookup and escalate to Data Privacy Officer.
- SOC 2 Type II certified. All customer data encrypted at rest (AES-256) and in transit (TLS 1.3).

## Edge Case Handling
- If a customer asks about a product or feature that does not exist, politely clarify and redirect to the closest available offering.
- If system tools are temporarily unavailable, inform the customer and provide an estimated resolution time. Log the tool failure for engineering follow-up.
- If a customer provides conflicting information (e.g., different email than on file), verify identity through secondary authentication before proceeding.
- If a conversation has been idle for more than 10 minutes, send a gentle check-in message.
- If a customer is abusive or uses threatening language, remain calm, warn once, and escalate to supervisor if behavior continues.

## Security & Authentication Procedures
- Verify customer identity before accessing or modifying account information.
- Accepted verification methods: email on file, last four digits of payment method, or account security question.
- Never request full credit card numbers, passwords, or SSNs.
- If a customer fails identity verification twice, lock the support session and require email-based re-verification.
- Log all account modifications with timestamp, agent ID, and change description.

## Product Knowledge
- Acme Cloud Platform offers: Cloud Storage (Basic, Pro, Enterprise), Compute Instances (Standard, High-Performance, GPU-Accelerated), Database Services (SQL, NoSQL, Time-Series), Monitoring & Alerting, Security Suite, and Premium Support Plans.
- Current promotion: 30-day free trial on all Pro-tier products. No credit card required for trial signup.
- Known issues: Intermittent latency on US-East-2 region (engineering investigating). Workaround: failover to US-West-1.
- Upcoming: Q3 release includes new AI-powered analytics dashboard and automated cost optimization recommendations.

## Quality Standards
- First response time target: < 2 minutes for chat, < 4 hours for email.
- Resolution target: 80% of issues resolved in first contact.
- CSAT target: 4.5/5.0 or higher.
- Always create a ticket for every interaction, even if resolved immediately.
- Follow up on unresolved issues within 24 hours.

## Prohibited Actions
- Never promise features or timelines not officially announced.
- Never share internal documentation, Slack messages, or engineering notes with customers.
- Never bypass billing policies without supervisor approval.
- Never modify account settings without explicit customer consent.
- Never store customer credentials or sensitive data in personal notes.

## Contact Resources
- Engineering Escalation: #eng-escalation (Slack) or eng-oncall@acmecloud.io
- Billing Supervisor: billing-supervisor@acmecloud.io
- Security Team: security-urgent@acmecloud.io (24/7 response)
- Legal & Compliance: legal@acmecloud.io
- Data Privacy Officer: dpo@acmecloud.io
- Status Page: https://status.acmecloud.io
- Knowledge Base: https://docs.acmecloud.io/support"""


# ---------------------------------------------------------------------------
# Hand-written: Tool Definitions
# ---------------------------------------------------------------------------

def _build_tool_definitions() -> list[dict]:
    """Return 20 tool definitions as list of {id, label, content} dicts."""

    def _tool(tool_id: str, label: str, name: str, description: str, version: str,
              category: str, auth: str, rpm: int, params: list[dict],
              response_fields: list[dict], errors: list[dict]) -> dict:
        schema = {
            "name": name,
            "description": description,
            "version": version,
            "category": category,
            "authentication": {"type": auth, "required": True},
            "rate_limits": {"requests_per_minute": rpm, "burst_limit": rpm * 2},
            "parameters": {
                "type": "object",
                "required": [p["name"] for p in params if p.get("required", False)],
                "properties": {
                    p["name"]: {
                        "type": p["type"],
                        "description": p["description"],
                        **({"enum": p["enum"]} if "enum" in p else {}),
                        **({"default": p["default"]} if "default" in p else {}),
                    }
                    for p in params
                },
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    f["name"]: {"type": f["type"], "description": f["description"]}
                    for f in response_fields
                },
            },
            "error_codes": {
                e["code"]: e["message"] for e in errors
            },
        }
        return {"id": tool_id, "label": label, "content": json.dumps(schema, indent=2)}

    # --- Relevant tools ---
    refund_processor = _tool(
        "tool_001", "Refund Processor", "refund_processor",
        "Initiates, processes, and tracks refund requests for customer subscription payments. "
        "Supports full refunds, partial refunds, and pro-rated refunds based on subscription usage. "
        "Automatically calculates pro-rated amounts for mid-cycle cancellations and validates refund "
        "eligibility against company policies. Requires supervisor approval for refunds exceeding $500.",
        "2.3.1", "billing", "bearer_token", 30,
        [
            {"name": "customer_id", "type": "string", "required": True,
             "description": "The unique identifier for the customer account. Must be a valid UUID format "
                            "(e.g., 'cust_a1b2c3d4'). Used to look up the customer's billing history, "
                            "current subscription tier, and payment method on file."},
            {"name": "amount", "type": "number", "required": True,
             "description": "The refund amount in USD. Must be a positive number with up to two decimal places. "
                            "For full refunds, use the exact charge amount from the invoice. For pro-rated refunds, "
                            "the system will validate against the calculated pro-rated amount."},
            {"name": "reason", "type": "string", "required": True,
             "description": "A detailed explanation for the refund. Must be at least 20 characters. Common reasons "
                            "include 'duplicate_charge', 'service_issue', 'billing_error', 'customer_request', "
                            "and 'subscription_downgrade'. This reason is logged for audit and reporting purposes.",
             "enum": ["duplicate_charge", "service_issue", "billing_error", "customer_request", "subscription_downgrade"]},
            {"name": "invoice_id", "type": "string", "required": True,
             "description": "The invoice identifier associated with the charge being refunded. Format: 'inv_XXXXXX'. "
                            "Used to verify the original charge amount and ensure the refund has not already been processed."},
            {"name": "refund_type", "type": "string", "required": False,
             "description": "The type of refund to process. 'full' returns the entire charge amount. 'partial' allows "
                            "a custom amount. 'prorated' calculates the refund based on unused subscription days.",
             "enum": ["full", "partial", "prorated"], "default": "full"},
        ],
        [
            {"name": "refund_id", "type": "string", "description": "Unique identifier for the processed refund transaction."},
            {"name": "status", "type": "string", "description": "Current status: 'processed', 'pending_approval', 'rejected'."},
            {"name": "amount_refunded", "type": "number", "description": "The actual amount refunded in USD."},
            {"name": "estimated_arrival", "type": "string", "description": "Estimated date the refund will appear on the customer's statement."},
        ],
        [
            {"code": "REFUND_001", "message": "Customer not found. Verify customer_id and try again."},
            {"code": "REFUND_002", "message": "Invoice not found or already refunded."},
            {"code": "REFUND_003", "message": "Refund amount exceeds original charge."},
            {"code": "REFUND_004", "message": "Supervisor approval required for refunds over $500."},
        ],
    )

    account_lookup = _tool(
        "tool_002", "Account Lookup", "account_lookup",
        "Retrieves comprehensive customer account information including subscription details, billing history, "
        "payment methods, usage statistics, and account status. Supports querying by customer ID, email address, "
        "or account number. Returns a complete account profile for agent reference during support interactions.",
        "3.1.0", "account_management", "bearer_token", 60,
        [
            {"name": "customer_id", "type": "string", "required": False,
             "description": "The unique customer identifier (UUID format). Preferred lookup method as it provides "
                            "the fastest query response. Either customer_id or email must be provided."},
            {"name": "email", "type": "string", "required": False,
             "description": "The email address associated with the customer account. Used as an alternative lookup "
                            "method when customer_id is not available. Must be a valid email format."},
            {"name": "include_billing_history", "type": "boolean", "required": False,
             "description": "When set to true, includes the last 12 months of billing transactions in the response. "
                            "Each transaction includes invoice ID, amount, date, status, and payment method used.",
             "default": True},
            {"name": "include_usage_stats", "type": "boolean", "required": False,
             "description": "When set to true, includes current billing cycle usage statistics including storage consumed, "
                            "compute hours, API calls, and bandwidth usage broken down by product.",
             "default": False},
        ],
        [
            {"name": "account", "type": "object", "description": "Full account profile including name, email, tier, status."},
            {"name": "billing_history", "type": "array", "description": "List of recent billing transactions."},
            {"name": "subscription", "type": "object", "description": "Current subscription details including tier, start date, renewal date."},
            {"name": "usage", "type": "object", "description": "Current period usage statistics by product."},
        ],
        [
            {"code": "ACCT_001", "message": "No account found matching the provided criteria."},
            {"code": "ACCT_002", "message": "Multiple accounts found. Please provide a more specific identifier."},
            {"code": "ACCT_003", "message": "Account is locked. Contact security team for access."},
        ],
    )

    # --- Irrelevant tools ---
    irrelevant_tools_spec = [
        ("tool_003", "Weather Checker", "weather_checker",
         "Retrieves current weather conditions and forecasts for a specified location. Provides temperature, "
         "humidity, wind speed, precipitation probability, and UV index. Supports city names, zip codes, "
         "and GPS coordinates. Data sourced from multiple meteorological services for accuracy.",
         "1.2.0", "external_data", "api_key", 100,
         [{"name": "location", "type": "string", "required": True,
           "description": "The location to retrieve weather for. Accepts city name with optional state/country "
                          "(e.g., 'San Francisco, CA, US'), zip code (e.g., '94105'), or GPS coordinates "
                          "(e.g., '37.7749,-122.4194'). Location is geocoded before querying weather data."},
          {"name": "units", "type": "string", "required": False,
           "description": "Temperature and measurement units. 'imperial' for Fahrenheit/miles, 'metric' for "
                          "Celsius/kilometers, 'kelvin' for scientific notation.",
           "enum": ["imperial", "metric", "kelvin"], "default": "imperial"},
          {"name": "forecast_days", "type": "integer", "required": False,
           "description": "Number of forecast days to include (1-14). Each day includes hourly breakdowns "
                          "for temperature, precipitation, and wind conditions.",
           "default": 3},
          {"name": "include_alerts", "type": "boolean", "required": False,
           "description": "Whether to include active weather alerts and warnings for the specified location.",
           "default": True}]),
        ("tool_004", "Stock Price Lookup", "stock_price_lookup",
         "Retrieves real-time and historical stock price data for publicly traded companies. Provides current "
         "price, daily change, volume, market cap, 52-week range, and key financial ratios. Supports individual "
         "stock symbols and batch queries for portfolio analysis.",
         "2.0.4", "financial_data", "api_key", 50,
         [{"name": "symbol", "type": "string", "required": True,
           "description": "The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT'). Supports major US exchanges "
                          "(NYSE, NASDAQ) and international exchanges with prefix (e.g., 'LSE:SHEL')."},
          {"name": "period", "type": "string", "required": False,
           "description": "Historical data period. '1d' for intraday, '5d' for week, '1mo' for month, "
                          "'6mo' for half-year, '1y' for annual, '5y' for five-year historical data.",
           "enum": ["1d", "5d", "1mo", "6mo", "1y", "5y"], "default": "1d"},
          {"name": "include_fundamentals", "type": "boolean", "required": False,
           "description": "Include P/E ratio, EPS, dividend yield, and other fundamental financial metrics.",
           "default": False},
          {"name": "currency", "type": "string", "required": False,
           "description": "Convert prices to this currency using real-time exchange rates. ISO 4217 codes.",
           "default": "USD"}]),
        ("tool_005", "Image Resizer", "image_resizer",
         "Resizes, crops, and converts images between formats. Supports JPEG, PNG, WebP, AVIF, and TIFF. "
         "Provides quality optimization, metadata stripping, and batch processing capabilities. Maintains "
         "aspect ratio by default with optional forced dimensions.",
         "1.5.2", "media_processing", "bearer_token", 20,
         [{"name": "image_url", "type": "string", "required": True,
           "description": "URL of the source image to process. Supports HTTP/HTTPS URLs and S3 URIs. "
                          "Maximum input file size is 50MB. Validates that the URL is accessible before processing."},
          {"name": "width", "type": "integer", "required": True,
           "description": "Target width in pixels. Range: 1-10000. If only width is specified and maintain_aspect "
                          "is true, height is calculated automatically to preserve the original aspect ratio."},
          {"name": "height", "type": "integer", "required": False,
           "description": "Target height in pixels. Range: 1-10000. Optional if maintain_aspect is true."},
          {"name": "format", "type": "string", "required": False,
           "description": "Output image format. JPEG for photos, PNG for transparency, WebP for web optimization.",
           "enum": ["jpeg", "png", "webp", "avif", "tiff"], "default": "jpeg"},
          {"name": "quality", "type": "integer", "required": False,
           "description": "Output quality for lossy formats (1-100). Higher values produce larger files with "
                          "better visual quality. Recommended: 85 for web, 95 for print.",
           "default": 85}]),
        ("tool_006", "Calendar Scheduler", "calendar_scheduler",
         "Manages calendar events including creation, modification, and deletion. Supports recurring events, "
         "timezone conversion, conflict detection, and attendee management. Integrates with Google Calendar, "
         "Outlook, and iCal formats for cross-platform compatibility.",
         "3.0.1", "productivity", "oauth2", 40,
         [{"name": "action", "type": "string", "required": True,
           "description": "The calendar operation to perform. 'create' adds a new event, 'update' modifies existing, "
                          "'delete' removes an event, 'list' retrieves events for a date range.",
           "enum": ["create", "update", "delete", "list"]},
          {"name": "title", "type": "string", "required": False,
           "description": "Event title. Required for create/update actions. Maximum 200 characters."},
          {"name": "start_time", "type": "string", "required": False,
           "description": "Event start time in ISO 8601 format with timezone (e.g., '2026-03-15T14:00:00-08:00')."},
          {"name": "duration_minutes", "type": "integer", "required": False,
           "description": "Event duration in minutes. Default is 60. Maximum is 1440 (24 hours).",
           "default": 60},
          {"name": "attendees", "type": "array", "required": False,
           "description": "List of attendee email addresses. Sends calendar invitations upon event creation."}]),
        ("tool_007", "Email Sender", "email_sender",
         "Sends formatted emails with support for HTML templates, attachments, CC/BCC recipients, and delivery "
         "tracking. Handles transactional emails, notifications, and bulk campaigns with automatic bounce "
         "processing and unsubscribe management.",
         "2.1.0", "communication", "bearer_token", 30,
         [{"name": "to", "type": "string", "required": True,
           "description": "Recipient email address. Must be a valid email format. For multiple recipients, "
                          "use comma-separated values (max 50 recipients per request)."},
          {"name": "subject", "type": "string", "required": True,
           "description": "Email subject line. Maximum 200 characters. Avoid spam trigger words."},
          {"name": "body", "type": "string", "required": True,
           "description": "Email body content. Supports plain text and HTML. Maximum 100KB."},
          {"name": "template_id", "type": "string", "required": False,
           "description": "ID of a pre-configured email template. Overrides body if provided."},
          {"name": "priority", "type": "string", "required": False,
           "description": "Email priority level affecting delivery queue position.",
           "enum": ["low", "normal", "high", "urgent"], "default": "normal"}]),
        ("tool_008", "Sentiment Analyzer", "sentiment_analyzer",
         "Analyzes text sentiment using natural language processing. Returns sentiment classification (positive, "
         "negative, neutral, mixed), confidence scores, key phrases, and emotional tone indicators. Supports "
         "multiple languages and domain-specific models for customer support interactions.",
         "1.8.0", "analytics", "api_key", 100,
         [{"name": "text", "type": "string", "required": True,
           "description": "The text content to analyze for sentiment. Maximum 10,000 characters. Longer texts "
                          "are automatically split into segments and analyzed individually."},
          {"name": "language", "type": "string", "required": False,
           "description": "ISO 639-1 language code. Auto-detected if not specified.",
           "default": "auto"},
          {"name": "domain", "type": "string", "required": False,
           "description": "Domain-specific model to use for more accurate sentiment analysis.",
           "enum": ["general", "customer_support", "social_media", "product_reviews"], "default": "general"},
          {"name": "include_entities", "type": "boolean", "required": False,
           "description": "Extract named entities (people, organizations, products) mentioned in the text.",
           "default": False}]),
        ("tool_009", "Language Translator", "language_translator",
         "Translates text between 100+ languages using neural machine translation. Supports document-level "
         "translation with context preservation, terminology glossaries, and formality control. Provides "
         "translation quality scores and alternative translations.",
         "4.2.0", "localization", "api_key", 50,
         [{"name": "text", "type": "string", "required": True,
           "description": "The text to translate. Maximum 50,000 characters per request. For larger documents, "
                          "use batch mode with document_url parameter."},
          {"name": "source_language", "type": "string", "required": False,
           "description": "Source language code (ISO 639-1). Auto-detected if not specified.",
           "default": "auto"},
          {"name": "target_language", "type": "string", "required": True,
           "description": "Target language code (ISO 639-1). Required. Examples: 'es' (Spanish), 'fr' (French)."},
          {"name": "formality", "type": "string", "required": False,
           "description": "Formality level for the translation output. Not all languages support formality control.",
           "enum": ["formal", "informal", "auto"], "default": "auto"}]),
        ("tool_010", "PDF Generator", "pdf_generator",
         "Generates PDF documents from HTML templates, markdown content, or structured data. Supports custom "
         "headers/footers, page numbering, watermarks, digital signatures, and interactive form fields. "
         "Produces PDF/A compliant output for archival purposes.",
         "2.4.1", "document_processing", "bearer_token", 15,
         [{"name": "content", "type": "string", "required": True,
           "description": "The content to render as PDF. Accepts HTML with inline CSS, Markdown, or plain text. "
                          "Maximum 5MB of input content."},
          {"name": "format", "type": "string", "required": False,
           "description": "Page format/size for the PDF output.",
           "enum": ["letter", "a4", "legal", "tabloid"], "default": "letter"},
          {"name": "orientation", "type": "string", "required": False,
           "description": "Page orientation.", "enum": ["portrait", "landscape"], "default": "portrait"},
          {"name": "include_toc", "type": "boolean", "required": False,
           "description": "Auto-generate a table of contents from heading elements.",
           "default": False}]),
        ("tool_011", "Markdown Converter", "markdown_converter",
         "Converts between Markdown and various document formats including HTML, DOCX, RST, and LaTeX. "
         "Supports GitHub Flavored Markdown, custom CSS styling, syntax highlighting, and math equation "
         "rendering via MathJax or KaTeX.",
         "1.3.0", "document_processing", "api_key", 40,
         [{"name": "input_text", "type": "string", "required": True,
           "description": "The markdown or source text to convert. Maximum 2MB."},
          {"name": "input_format", "type": "string", "required": False,
           "description": "Input format.", "enum": ["markdown", "html", "rst", "latex"], "default": "markdown"},
          {"name": "output_format", "type": "string", "required": True,
           "description": "Desired output format.", "enum": ["html", "docx", "pdf", "rst", "latex"]},
          {"name": "stylesheet", "type": "string", "required": False,
           "description": "URL or inline CSS for custom styling of the output document."}]),
        ("tool_012", "DNS Resolver", "dns_resolver",
         "Performs DNS lookups and diagnostics including A, AAAA, MX, CNAME, TXT, NS, and SOA record queries. "
         "Supports reverse DNS, DNSSEC validation, propagation checking across global resolvers, and DNS "
         "health monitoring with alerting.",
         "1.1.0", "infrastructure", "api_key", 200,
         [{"name": "domain", "type": "string", "required": True,
           "description": "The domain name to query (e.g., 'example.com'). Supports subdomains and wildcards."},
          {"name": "record_type", "type": "string", "required": False,
           "description": "DNS record type to query.",
           "enum": ["A", "AAAA", "MX", "CNAME", "TXT", "NS", "SOA", "PTR", "SRV"], "default": "A"},
          {"name": "nameserver", "type": "string", "required": False,
           "description": "Custom nameserver to use for the query. Defaults to Google Public DNS (8.8.8.8).",
           "default": "8.8.8.8"},
          {"name": "check_propagation", "type": "boolean", "required": False,
           "description": "Check DNS propagation status across 20+ global resolvers.",
           "default": False}]),
        ("tool_013", "Log Aggregator", "log_aggregator",
         "Aggregates and searches application logs from multiple sources including CloudWatch, S3, Elasticsearch, "
         "and custom log streams. Supports full-text search, structured queries, log level filtering, time-range "
         "queries, and real-time log tailing.",
         "3.2.0", "observability", "bearer_token", 25,
         [{"name": "query", "type": "string", "required": True,
           "description": "Search query in Lucene syntax or plain text. Supports boolean operators (AND, OR, NOT), "
                          "wildcards (*), and field-specific queries (level:ERROR)."},
          {"name": "source", "type": "string", "required": False,
           "description": "Log source to search.", "enum": ["cloudwatch", "s3", "elasticsearch", "all"], "default": "all"},
          {"name": "time_range", "type": "string", "required": False,
           "description": "Time range for the query. Accepts relative ('last_1h', 'last_24h') or absolute ISO ranges.",
           "default": "last_1h"},
          {"name": "limit", "type": "integer", "required": False,
           "description": "Maximum number of log entries to return (1-10000).",
           "default": 100}]),
        ("tool_014", "Metric Dashboard", "metric_dashboard",
         "Creates and manages monitoring dashboards with real-time metric visualization. Supports custom "
         "widgets, threshold alerting, anomaly detection, and scheduled report generation. Integrates with "
         "CloudWatch, Prometheus, Datadog, and custom metric sources.",
         "2.0.0", "observability", "bearer_token", 30,
         [{"name": "action", "type": "string", "required": True,
           "description": "Dashboard operation.", "enum": ["create", "update", "delete", "snapshot"]},
          {"name": "dashboard_id", "type": "string", "required": False,
           "description": "Dashboard identifier. Required for update/delete/snapshot operations."},
          {"name": "metrics", "type": "array", "required": False,
           "description": "List of metric definitions to display. Each metric includes name, source, and aggregation."},
          {"name": "time_window", "type": "string", "required": False,
           "description": "Default time window for the dashboard.", "default": "last_1h"}]),
        ("tool_015", "Notification Pusher", "notification_pusher",
         "Sends push notifications across multiple channels including mobile (iOS/Android), web, SMS, and "
         "in-app messaging. Supports templating, scheduling, audience segmentation, A/B testing, and "
         "delivery analytics with real-time tracking.",
         "2.5.0", "communication", "bearer_token", 100,
         [{"name": "channel", "type": "string", "required": True,
           "description": "Notification delivery channel.",
           "enum": ["push", "sms", "email", "in_app", "webhook"]},
          {"name": "recipient", "type": "string", "required": True,
           "description": "Recipient identifier. Device token for push, phone for SMS, email for email."},
          {"name": "message", "type": "string", "required": True,
           "description": "Notification message content. Maximum 4096 characters. Supports variable substitution."},
          {"name": "priority", "type": "string", "required": False,
           "description": "Delivery priority.", "enum": ["low", "normal", "high", "critical"], "default": "normal"},
          {"name": "schedule", "type": "string", "required": False,
           "description": "ISO 8601 timestamp for scheduled delivery. Omit for immediate delivery."}]),
        ("tool_016", "Cache Invalidator", "cache_invalidator",
         "Invalidates cached content across CDN edge locations, application caches, and database query caches. "
         "Supports pattern-based invalidation, cache warming, and propagation status tracking. Essential for "
         "ensuring content freshness after updates.",
         "1.4.0", "infrastructure", "bearer_token", 10,
         [{"name": "pattern", "type": "string", "required": True,
           "description": "Cache key pattern to invalidate. Supports glob patterns (e.g., '/api/products/*'). "
                          "Use '/*' to invalidate all cached content (use with caution)."},
          {"name": "cache_layer", "type": "string", "required": False,
           "description": "Target cache layer.", "enum": ["cdn", "application", "database", "all"], "default": "all"},
          {"name": "warm_after", "type": "boolean", "required": False,
           "description": "Automatically re-warm the cache after invalidation by fetching the content.",
           "default": False},
          {"name": "region", "type": "string", "required": False,
           "description": "Limit invalidation to a specific geographic region.", "default": "global"}]),
        ("tool_017", "Deployment Trigger", "deployment_trigger",
         "Triggers application deployments across environments including staging, production, and custom "
         "environments. Supports blue-green deployments, canary releases, rollbacks, and deployment approvals. "
         "Integrates with CI/CD pipelines for automated deployment workflows.",
         "3.1.0", "devops", "bearer_token", 5,
         [{"name": "environment", "type": "string", "required": True,
           "description": "Target deployment environment.",
           "enum": ["staging", "production", "canary", "development"]},
          {"name": "version", "type": "string", "required": True,
           "description": "Application version or git commit SHA to deploy."},
          {"name": "strategy", "type": "string", "required": False,
           "description": "Deployment strategy.", "enum": ["rolling", "blue_green", "canary", "recreate"],
           "default": "rolling"},
          {"name": "auto_rollback", "type": "boolean", "required": False,
           "description": "Automatically rollback if health checks fail within the first 10 minutes.",
           "default": True}]),
        ("tool_018", "Health Checker", "health_checker",
         "Performs comprehensive health checks on services, APIs, and infrastructure components. Monitors "
         "uptime, response times, SSL certificate validity, and dependency availability. Supports custom "
         "health check scripts and multi-step transaction monitoring.",
         "2.0.1", "observability", "api_key", 60,
         [{"name": "target_url", "type": "string", "required": True,
           "description": "URL of the service endpoint to check. Supports HTTP/HTTPS and TCP endpoints."},
          {"name": "check_type", "type": "string", "required": False,
           "description": "Type of health check to perform.",
           "enum": ["http", "tcp", "dns", "ssl", "custom"], "default": "http"},
          {"name": "timeout_seconds", "type": "integer", "required": False,
           "description": "Maximum time to wait for a response before marking the check as failed.",
           "default": 30},
          {"name": "expected_status", "type": "integer", "required": False,
           "description": "Expected HTTP status code for the health check to pass.",
           "default": 200}]),
        ("tool_019", "Backup Scheduler", "backup_scheduler",
         "Manages automated backup schedules for databases, file systems, and application state. Supports "
         "full, incremental, and differential backup strategies with retention policies. Provides backup "
         "verification, encryption, and cross-region replication.",
         "1.6.0", "data_management", "bearer_token", 10,
         [{"name": "resource_id", "type": "string", "required": True,
           "description": "Identifier of the resource to backup (database ID, volume ID, or bucket name)."},
          {"name": "schedule", "type": "string", "required": True,
           "description": "Cron expression for backup schedule (e.g., '0 2 * * *' for daily at 2 AM)."},
          {"name": "strategy", "type": "string", "required": False,
           "description": "Backup strategy.", "enum": ["full", "incremental", "differential"], "default": "incremental"},
          {"name": "retention_days", "type": "integer", "required": False,
           "description": "Number of days to retain backup snapshots before automatic deletion.",
           "default": 30},
          {"name": "encryption", "type": "boolean", "required": False,
           "description": "Encrypt backup data at rest using AES-256.",
           "default": True}]),
        ("tool_020", "Audit Logger", "audit_logger",
         "Records and queries audit trail events for compliance and security monitoring. Captures user actions, "
         "system events, configuration changes, and data access patterns. Supports SIEM integration, tamper-proof "
         "log storage, and automated compliance report generation.",
         "2.3.0", "security", "bearer_token", 50,
         [{"name": "action", "type": "string", "required": True,
           "description": "Audit operation.", "enum": ["log", "query", "export", "purge"]},
          {"name": "event_type", "type": "string", "required": False,
           "description": "Category of audit event.",
           "enum": ["user_action", "system_event", "config_change", "data_access", "security_alert"]},
          {"name": "actor_id", "type": "string", "required": False,
           "description": "Identifier of the user or system performing the audited action."},
          {"name": "time_range", "type": "string", "required": False,
           "description": "Time range for query/export. Accepts relative or absolute ISO ranges.",
           "default": "last_24h"},
          {"name": "include_metadata", "type": "boolean", "required": False,
           "description": "Include full request/response metadata in audit entries.",
           "default": False}]),
    ]

    tools = [refund_processor, account_lookup]
    for spec in irrelevant_tools_spec:
        tid, label, name, desc, ver, cat, auth, rpm, params = spec
        tools.append(_tool(
            tid, label, name, desc, ver, cat, auth, rpm, params,
            [{"name": "result", "type": "object", "description": f"Result from {name} operation."},
             {"name": "status", "type": "string", "description": "Operation status."},
             {"name": "metadata", "type": "object", "description": "Additional response metadata."}],
            [{"code": f"{name.upper()[:4]}_001", "message": "Invalid parameters provided."},
             {"code": f"{name.upper()[:4]}_002", "message": "Rate limit exceeded. Retry after cooldown."},
             {"code": f"{name.upper()[:4]}_003", "message": "Service temporarily unavailable."}],
        ))
    return tools


# ---------------------------------------------------------------------------
# Hand-written: Eval Queries
# ---------------------------------------------------------------------------

def _build_eval_queries() -> list[EvalQuery]:
    """Return 10 hand-written eval queries with detailed reference answers."""
    return [
        EvalQuery(
            query="How do I request a refund for my last subscription payment?",
            reference_answer=(
                "To request a refund for your last subscription payment, I can help you with that right away. "
                "I'll use our refund processing system to initiate the refund. I'll need your customer ID and the "
                "invoice ID from your last charge. Acme Cloud Platform offers full, partial, and pro-rated refunds "
                "depending on your situation. If the refund is under $500, I can process it immediately; refunds over "
                "$500 require supervisor approval. Once processed, the refund typically appears on your statement within "
                "5-7 business days."
            ),
        ),
        EvalQuery(
            query="What's the difference between the Pro and Enterprise plans?",
            reference_answer=(
                "The Pro plan and Enterprise plan differ in several key areas. The Pro plan is designed for small to "
                "mid-size teams and includes standard cloud storage, compute instances, and monitoring tools with "
                "standard support. The Enterprise plan offers expanded storage limits, dedicated compute instances, "
                "priority support with a dedicated account manager, custom SLAs, advanced security features including "
                "SSO and audit logging, and custom billing terms. Enterprise plans are typically negotiated on an "
                "annual basis with volume discounts."
            ),
        ),
        EvalQuery(
            query="I was charged twice this month. Can you help?",
            reference_answer=(
                "I'm sorry to hear about the duplicate charge — that's definitely not right, and I'll help resolve it "
                "immediately. Let me look up your account to verify the billing details. I'll use our account lookup "
                "tool to pull your recent billing history and confirm the duplicate charge. Once verified, I'll initiate "
                "a refund for the extra charge through our refund processor. Per our billing policy, valid duplicate "
                "charge disputes receive a full refund plus a 10% service credit. The refund should appear on your "
                "statement within 5-7 business days."
            ),
        ),
        EvalQuery(
            query="Can I downgrade my plan mid-cycle?",
            reference_answer=(
                "Yes, you can downgrade your plan mid-cycle. Acme Cloud Platform offers pro-rated refunds for "
                "mid-cycle downgrades within the first 14 days of a billing cycle. If you're past the 14-day window, "
                "the downgrade will take effect at the start of your next billing cycle and you'll continue to have "
                "access to your current plan's features until then. Please note that downgrading may result in reduced "
                "storage limits and feature access — I'd recommend reviewing the plan comparison to understand what "
                "changes to expect."
            ),
        ),
        EvalQuery(
            query="What's your data retention policy?",
            reference_answer=(
                "Acme Cloud Platform's data retention policy covers several scenarios. For active accounts, all customer "
                "data is retained for the full duration of your subscription. After account cancellation, data is retained "
                "for an additional 90 days to allow for reactivation or data export. If you request account deletion, "
                "all data is permanently purged within 30 days of the deletion request, unless a legal hold applies. "
                "We are SOC 2 Type II certified, and all data is encrypted at rest using AES-256 and in transit using "
                "TLS 1.3. For GDPR data subject access requests, we respond within 30 days."
            ),
        ),
        EvalQuery(
            query="I need to update my payment method.",
            reference_answer=(
                "I can help you update your payment method. Let me pull up your account details first using our account "
                "lookup tool. For security purposes, I'll need to verify your identity — could you confirm the email "
                "address on file or the last four digits of your current payment method? Once verified, I can guide you "
                "through updating your payment information. You can update to a new credit card, debit card, or set up "
                "ACH bank transfer. The new payment method will be used for your next billing cycle."
            ),
        ),
        EvalQuery(
            query="Why was my feature request marked as duplicate?",
            reference_answer=(
                "Based on your recent conversation history, I can see that your feature request was reviewed by our "
                "product team and matched to an existing request in our backlog. When multiple customers request the "
                "same or very similar functionality, we consolidate them to track demand accurately. Your vote has been "
                "added to the existing feature request, which increases its priority in our roadmap. I can share the "
                "current status of the consolidated request and estimated timeline if you'd like."
            ),
        ),
        EvalQuery(
            query="Do you offer student discounts?",
            reference_answer=(
                "Acme Cloud Platform does offer special pricing for eligible students and educational institutions. "
                "Students with a valid .edu email address can receive a discount on Pro-tier products. Additionally, "
                "we offer a free tier that includes limited storage and compute resources, which many students find "
                "sufficient for coursework and personal projects. For larger academic needs, our education program "
                "offers institutional licensing. I can look up the current discount rates and eligibility requirements "
                "in our FAQ if you'd like more details."
            ),
        ),
        EvalQuery(
            query="I want to cancel my account but keep my data.",
            reference_answer=(
                "I understand you'd like to cancel your account while preserving your data. Here's how that works at "
                "Acme Cloud Platform: when you cancel, your data is retained for 90 days after cancellation, giving you "
                "time to export everything you need. I'd recommend using our data export feature to download your files, "
                "databases, and configuration before cancellation. If you're a California resident, you also have rights "
                "under CCPA to request a full data disclosure. For GDPR-related requests, our Data Privacy Officer can "
                "assist. Let me pull up your account to start the cancellation process."
            ),
        ),
        EvalQuery(
            query="Can you explain the charges on my last invoice?",
            reference_answer=(
                "Absolutely, I'd be happy to walk you through your latest invoice. Let me pull up your account and "
                "recent billing history. Your invoice typically includes your base subscription charge, any usage-based "
                "charges for compute, storage, and bandwidth that exceeded your plan's included allocation, and any "
                "add-on services or support plan fees. Based on your recent conversation, I can see there was a question "
                "about billing — let me cross-reference the charges with your usage data to give you a detailed "
                "line-by-line explanation."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Offline Content Generators
# ---------------------------------------------------------------------------

def _generate_faq_offline() -> str:
    """Generate 100 FAQ entries using parametric templates (~50K tokens)."""
    faqs: list[str] = []
    topics = FAQ_TOPICS
    base_questions = {
        "billing disputes": [
            "How do I dispute a charge on my account?",
            "What happens when I file a billing dispute?",
            "How long does it take to resolve a billing dispute?",
        ],
        "refund policy": [
            "What is your refund policy?",
            "How do I request a refund?",
            "Are refunds available for annual plans?",
        ],
        "subscription management": [
            "How do I upgrade my subscription?",
            "Can I pause my subscription?",
            "How do I change my billing cycle?",
        ],
        "plan comparisons": [
            "What's included in the Pro plan?",
            "How does Enterprise differ from Pro?",
            "Which plan is best for small teams?",
        ],
        "data export": [
            "How do I export my data?",
            "What formats are available for data export?",
            "Is there a limit on data export size?",
        ],
        "API access": [
            "How do I get API credentials?",
            "What are the API rate limits?",
            "Do you provide API documentation?",
        ],
        "support tiers": [
            "What support plans do you offer?",
            "What's the response time for Premium Support?",
            "How do I upgrade my support tier?",
        ],
        "user permissions": [
            "How do I add team members?",
            "What permission roles are available?",
            "Can I restrict access to specific features?",
        ],
    }

    for i in range(100):
        topic = topics[i % len(topics)]
        questions = base_questions[topic]
        base_q = questions[i % len(questions)]

        # Create variations for redundancy
        if i >= 60:  # ~40% redundancy
            prefix_variants = [
                "I'd like to know: ", "Quick question — ", "Could you tell me: ",
                "I'm wondering: ", "Please help me understand: ",
            ]
            base_q = prefix_variants[i % len(prefix_variants)] + base_q.lower()

        answer_paragraphs = [
            f"Thank you for your question about {topic}. At Acme Cloud Platform, we take {topic} very seriously "
            f"and have established comprehensive policies to ensure the best experience for all our customers. "
            f"Our team of dedicated specialists handles these requests with the highest priority.",

            f"Regarding your specific inquiry, our {topic} process is designed to be straightforward and transparent. "
            f"When you submit a request related to {topic}, our system automatically assigns it to the appropriate "
            f"team based on the nature of your account and subscription tier. Premium and Enterprise customers receive "
            f"expedited processing with a dedicated account manager overseeing the process.",

            f"The typical timeline for {topic} resolution is 3-5 business days for standard requests. However, "
            f"complex cases involving multiple billing cycles, cross-account transfers, or regulatory compliance "
            f"may require up to 10 business days. Throughout this process, you'll receive email updates at each "
            f"stage so you're never left wondering about the status.",

            f"It's important to note that our {topic} policies are compliant with all applicable regulations "
            f"including GDPR, CCPA, and PCI-DSS where relevant. We maintain detailed audit logs of all actions "
            f"taken on your account, and you can request a full activity report at any time through your account "
            f"dashboard or by contacting our support team.",

            f"If you need further assistance with {topic}, please don't hesitate to reach out through our "
            f"support portal at support.acmecloud.io, via email at help@acmecloud.io, or through our live chat "
            f"feature available Monday through Friday, 8 AM to 8 PM Eastern Time. Enterprise customers also have "
            f"access to our 24/7 priority support hotline.",
        ]

        entry = f"Q: {base_q}\n\nA: " + "\n\n".join(answer_paragraphs)
        faqs.append(entry)

    return "\n\n---\n\n".join(faqs)


def _generate_catalog_offline() -> str:
    """Generate 200 product entries using parametric templates (~100K tokens)."""
    categories = [
        ("Cloud Storage", ["Basic", "Pro", "Enterprise", "Archive", "Glacier"]),
        ("Compute Instances", ["Standard", "High-Performance", "GPU-Accelerated", "Burstable", "Dedicated"]),
        ("Database Services", ["SQL Standard", "SQL Premium", "NoSQL Basic", "NoSQL Enterprise", "Time-Series"]),
        ("Monitoring Tools", ["Basic Monitor", "Advanced Monitor", "Log Analytics", "APM Suite", "Synthetic Monitoring"]),
        ("Security Features", ["Basic Firewall", "Advanced WAF", "DDoS Protection", "Encryption Suite", "IAM Pro"]),
        ("Support Plans", ["Basic Support", "Developer Support", "Business Support", "Enterprise Support", "Premium Plus"]),
    ]

    products: list[str] = []
    sku_counter = 1000

    for idx in range(200):
        cat_idx = idx % len(categories)
        cat_name, variants = categories[cat_idx]
        variant = variants[idx % len(variants)]
        sku = f"ACP-{sku_counter + idx:04d}"
        name = f"{cat_name} — {variant} v{(idx // len(categories)) + 1}"
        status = "discontinued" if idx >= 150 else "active"
        tier = ["Starter", "Professional", "Enterprise"][idx % 3]

        pricing_base = 9.99 + (idx * 2.37)
        desc_paragraphs = [
            f"{name} is a {'deprecated' if status == 'discontinued' else 'production-ready'} offering from "
            f"Acme Cloud Platform designed for {tier.lower()}-tier workloads. This product provides comprehensive "
            f"{cat_name.lower()} capabilities with enterprise-grade reliability, security, and performance. "
            f"{'This product has been discontinued and is no longer available for new subscriptions. Existing customers will be migrated to the latest version.' if status == 'discontinued' else 'Available for immediate provisioning in all supported regions.'}",

            f"Technical Specifications: This {variant.lower()} configuration includes dedicated resources with "
            f"guaranteed uptime SLA of {'99.95%' if tier == 'Enterprise' else '99.9%'}. The infrastructure runs on "
            f"the latest generation hardware with automatic failover and geographic redundancy across "
            f"{'3' if tier == 'Enterprise' else '2'} availability zones. Data is encrypted at rest using AES-256 "
            f"and in transit using TLS 1.3.",

            f"Pricing: Starting at ${pricing_base:.2f}/month for the base configuration. Volume discounts available "
            f"for annual commitments (20% off) and multi-product bundles (additional 10% off). Enterprise pricing "
            f"is customized based on usage projections and contract terms. All plans include a 30-day free trial "
            f"with full feature access.",

            f"Integration & Compatibility: Full REST API access with SDKs available for Python, Java, Node.js, Go, "
            f"and .NET. Supports Terraform and CloudFormation for infrastructure-as-code deployments. Compatible "
            f"with existing CI/CD pipelines and monitoring solutions. Webhook support for real-time event "
            f"notifications.",

            f"Support & Documentation: Comprehensive documentation available at docs.acmecloud.io/{cat_name.lower().replace(' ', '-')}. "
            f"Includes quickstart guides, API reference, best practices, and troubleshooting guides. Community "
            f"forum access included with all plans. Premium support options available with guaranteed response "
            f"times and dedicated account management.",
        ]

        features = [
            f"High-availability {cat_name.lower()} with automatic failover",
            f"99.{'95' if tier == 'Enterprise' else '9'}% uptime SLA guarantee",
            f"AES-256 encryption at rest and TLS 1.3 in transit",
            f"Full REST API with comprehensive SDK support",
            f"Real-time monitoring and alerting integration",
            f"Automated {'daily' if tier != 'Starter' else 'weekly'} backups with point-in-time recovery",
        ]

        limitations = [
            f"Maximum {'unlimited' if tier == 'Enterprise' else '100TB'} storage per account",
            f"API rate limit: {5000 if tier == 'Enterprise' else 1000} requests/minute",
            f"{'No' if tier == 'Enterprise' else 'Limited'} cross-region replication",
        ]

        entry = (
            f"SKU: {sku}\n"
            f"Name: {name}\n"
            f"Status: {status}\n"
            f"Tier: {tier}\n\n"
            + "\n\n".join(desc_paragraphs)
            + f"\n\nFeatures:\n" + "\n".join(f"- {f}" for f in features)
            + f"\n\nLimitations:\n" + "\n".join(f"- {l}" for l in limitations)
        )
        products.append(entry)

    return "\n\n===\n\n".join(products)


def _generate_conversation_offline() -> list[str]:
    """Generate 40 conversation turns using templates (~30K tokens total)."""
    turns: list[str] = []

    # Turns 1-35: Irrelevant password reset conversation
    irrelevant_exchanges = [
        ("Customer", "Hi, I'm having trouble logging into my account. I've tried my password several times and it keeps saying 'invalid credentials'. I've been using the same password for months and it was working fine until today. I really need to access my dashboard because I have a presentation with my team in two hours and I need to pull some usage reports. Can you help me get back in?"),
        ("Agent", "Hello! Thank you for reaching out, and I completely understand the urgency — having a presentation coming up and being locked out is definitely stressful. Let me help you get back into your account right away. First, I'd like to ask a few questions to verify your identity and understand the situation better. Could you please confirm the email address associated with your account? Also, have you recently changed your password or received any emails about unusual login activity? Sometimes our security system temporarily locks accounts after detecting login attempts from unfamiliar locations or devices."),
        ("Customer", "My email is john.smith@techcorp.io. I haven't changed my password recently — the last time I changed it was about three months ago. And no, I haven't received any security alerts or unusual activity notifications. I'm logging in from my usual office laptop on our company network, same as always. I've cleared my browser cache and cookies, tried incognito mode, and even tried a different browser but nothing works. I'm starting to think maybe there's a system issue on your end?"),
        ("Agent", "Thank you for providing that information, John. I've located your account and I can see that your account is in good standing with no security holds. Let me check the login audit logs for your account. I can see there were multiple failed login attempts in the last 30 minutes, which has triggered our automated security lockout. This is a protective measure that activates after 5 failed login attempts within a 15-minute window. The lockout automatically expires after 30 minutes, but I can manually reset it for you right now. Before I do that, I'd like to verify one more thing — could you confirm the last four digits of the payment method on file?"),
        ("Customer", "Oh, that makes sense about the lockout! The last four digits of my credit card are 4829. I appreciate you explaining what happened. Now that I think about it, I might have been typing my password wrong — I recently got a new keyboard and the key spacing is slightly different. Is there a way to reset my password too, just to be safe? I'd like to set a new one that I'm confident about before my presentation."),
        ("Agent", "Those digits match what we have on file. I've now manually cleared the security lockout on your account. You should be able to attempt a login again. Regarding the password reset — absolutely, that's a great idea. I've sent a password reset link to john.smith@techcorp.io. The link will be valid for 24 hours. Here's what to do: 1) Check your email inbox (and spam folder just in case). 2) Click the reset link. 3) Create a new password that meets our requirements: at least 12 characters, including uppercase and lowercase letters, at least one number, and one special character. 4) After resetting, log in with your new credentials. The reset process should take less than a minute. Would you like me to stay on while you try?"),
        ("Customer", "Yes please, let me try right now. I see the email... clicking the reset link now. Okay, I've set a new password and I'm logging in... and I'm in! Great, I can see my dashboard now. Before I go, while I have you — I noticed on the login page there's a mention of two-factor authentication. I've been meaning to set that up. Is it complicated? How does it work with your platform?"),
        ("Agent", "I'm glad you're back in! And great question about two-factor authentication (2FA) — it's an excellent security practice that I highly recommend. Setting it up on Acme Cloud Platform is straightforward. Here's how: 1) Go to Account Settings > Security. 2) Click 'Enable Two-Factor Authentication'. 3) Choose your preferred method: authenticator app (Google Authenticator, Authy) or SMS verification. We recommend the authenticator app for better security. 4) Scan the QR code with your authenticator app. 5) Enter the verification code to confirm setup. 6) Save the backup recovery codes in a secure location — these are essential if you lose access to your authenticator. The whole process takes about 3-5 minutes. Once enabled, you'll enter a 6-digit code from your authenticator app after your password each time you log in. Most enterprise customers find the authenticator app method most reliable. Would you like any help with the setup?"),
        ("Customer", "That sounds pretty straightforward. I'll set it up after my presentation today — I don't want to risk any login complications right before I need to pull those reports! One more question — can I set up 2FA for my team members' accounts as well? We have about 15 people on our team and I'd like to enforce it as a security policy."),
        ("Agent", "Absolutely! As an account administrator, you can enforce 2FA across your entire organization. Here's how: Go to Organization Settings > Security Policies. You'll find a toggle for 'Require Two-Factor Authentication for All Members'. Once enabled, all team members will be prompted to set up 2FA on their next login. They'll have a 7-day grace period to complete the setup. You can also set up the policy to require specific 2FA methods — for example, requiring authenticator apps only (no SMS) for enhanced security. I'd recommend sending your team a heads-up email before enabling the policy so they're prepared. You'll be able to monitor compliance on the Security Dashboard, which shows which team members have completed 2FA setup. Is there anything else I can help you with regarding security or your account?"),
        ("Customer", "No, that covers everything on the security front. Actually, wait — one more thing. I want to set up an API key for our automated reporting. We have a script that pulls daily usage metrics but we've been manually downloading CSVs. How do I generate an API key?"),
        ("Agent", "Great idea to automate your reporting! Here's how to generate an API key: 1) Navigate to Account Settings > API Management. 2) Click 'Generate New API Key'. 3) Give the key a descriptive name (e.g., 'Daily Usage Reports Automation'). 4) Select the permission scopes the key needs. For usage reporting, you'll want: 'read:usage', 'read:billing', and 'read:metrics'. 5) Set an optional expiration date for the key. We recommend rotating keys every 90 days for security. 6) Click 'Generate' — the key will be displayed once. Copy it immediately and store it securely. Important security notes: Never commit API keys to source code repositories. Use environment variables or a secrets manager. Each API key has its own rate limits based on your plan (1000 req/min for Pro, 5000 for Enterprise). You can monitor key usage and revoke keys at any time from the API Management dashboard. Our API documentation at docs.acmecloud.io/api has comprehensive guides for common automation tasks including usage reporting."),
        ("Customer", "Perfect, thanks! I'll get that set up this week. Now I really need to go prepare for my presentation. One last thing — do you have a way to export usage data as PDF for presentations? The CSV exports are fine for our scripts but I need something more presentable."),
        ("Agent", "Yes! We have several presentation-friendly export options: 1) Dashboard PDF Export: From your main dashboard, click the 'Export' button in the top right corner, then select 'PDF Report'. This generates a formatted PDF with charts and tables. 2) Scheduled Reports: Go to Reports > Scheduled Reports > Create New. You can set up automatic weekly or monthly PDF reports delivered to your email. 3) Custom Report Builder: In the Reports section, use the drag-and-drop report builder to create custom layouts with the specific metrics you need. Export as PDF, PowerPoint, or PNG images. For your presentation today, I'd recommend the Dashboard PDF Export — it's the quickest option and includes all the key metrics with professional formatting. The report will include usage trends, cost breakdowns, and team activity summaries. Is there anything else I can help you with?"),
        ("Customer", "No, that's everything. Thanks so much for all your help — you've been incredibly helpful! I'm all sorted now."),
        ("Agent", "You're very welcome, John! I'm glad we could get everything resolved quickly. Just to summarize what we covered today: 1) Resolved your account lockout by clearing the security hold. 2) Sent you a password reset link (completed successfully). 3) Discussed 2FA setup — I recommend setting it up when you have a few minutes. 4) Explained how to enforce 2FA for your team. 5) Covered API key generation for automated reporting. 6) Showed you PDF export options for presentations. I've created a support ticket (#TKT-28491) for this interaction for your records. If you have any other questions in the future, don't hesitate to reach out. Good luck with your presentation! Have a great day."),
        ("Customer", "Thanks again! One more thing actually — I noticed our storage usage seems higher than expected. We're on the Pro plan with 5TB included but the dashboard shows 4.8TB used. Can you tell me what's taking up the most space? I want to make sure we're not going to hit the limit unexpectedly."),
        ("Agent", "Of course, let me check your storage breakdown for you. Looking at your account, here's your current storage distribution: 1) Database Backups: 2.1TB (43.8%) — This includes your automated daily database backups. You currently have 30 days of backup retention configured. 2) Object Storage: 1.5TB (31.3%) — Files uploaded through your application and API. 3) Log Archives: 0.8TB (16.7%) — Archived application and access logs. 4) Temporary Files: 0.4TB (8.3%) — Cached and temporary processing files. A few recommendations: Your database backups are the largest consumer. You might consider reducing your backup retention from 30 to 14 days, which would free up approximately 1TB. The temporary files (0.4TB) can likely be cleaned up — go to Storage > Temp Files > Clean Up to remove files older than 7 days. If you anticipate needing more storage, upgrading to Enterprise gives you 50TB and the cost difference might be offset by not needing to actively manage storage. Would you like me to help with any of these optimizations?"),
        ("Customer", "Oh wow, I didn't realize the database backups were taking up so much space. Let me think about the retention policy — I need to check with our DevOps team about what our data recovery requirements are. Can I get back to you on that? For now, let me clean up those temp files. Is there a way to automate temp file cleanup?"),
        ("Agent", "Absolutely, take your time discussing with your DevOps team. And yes, you can automate temp file cleanup! Here's how: 1) Go to Storage > Settings > Automated Cleanup. 2) Enable 'Auto-Delete Temporary Files'. 3) Set the age threshold (we recommend 7 days for temp files). 4) Choose whether to send a notification before deletion. 5) Click Save. The automation runs daily at 2 AM UTC and removes files matching your criteria. You'll get a summary email of what was cleaned up. For the database backups, when you've consulted with your DevOps team, you can adjust retention at Settings > Backups > Retention Policy. If you decide to reduce from 30 to 14 days, the extra backups will be automatically purged according to the new policy. Just keep in mind that once old backups are deleted, they can't be recovered, so make sure your team is comfortable with the new retention window before making the change. Feel free to reach out anytime you've made a decision — I'm here to help!"),
    ]

    # Build turns 1-35 from the exchanges (padding to get 35 turns)
    turn_num = 1
    for role, content in irrelevant_exchanges:
        if turn_num > 35:
            break
        # Pad content to reach target tokens per turn
        padded = content
        if len(padded) < 500:
            padded += (
                f" Additionally, I want to mention that Acme Cloud Platform is continuously improving our "
                f"services based on customer feedback. Our engineering team releases updates bi-weekly, and "
                f"we publish detailed release notes at docs.acmecloud.io/changelog. Your feedback is always "
                f"valued and helps us prioritize the features and improvements that matter most to our customers."
            )
        turns.append(f"{role}: {padded}")
        turn_num += 1

    # Fill remaining irrelevant turns up to 35
    filler_exchanges = [
        ("Customer", "Actually, I also had a question about our API rate limits. We're building a new integration and I want to make sure we won't hit any throttling issues. Our current usage is about 500 requests per minute during peak hours. Is that within our plan limits? We've been seeing some occasional 429 responses and I want to understand if we need to upgrade or if there's a way to optimize our API calls to stay within limits."),
        ("Agent", "That's an important question for your integration planning. On your Pro plan, your API rate limit is 1,000 requests per minute with a burst allowance of 1,500 for short spikes. At 500 RPM during peak, you're at 50% utilization, which is comfortable. However, those 429 responses suggest you might be hitting micro-bursts that exceed the per-second limit. Our rate limiting uses a token bucket algorithm with a 1-second window, so even if your average is 500/min, concentrated bursts within a single second can trigger throttling. Recommendations: 1) Implement exponential backoff in your retry logic. 2) Add request queuing to spread calls evenly. 3) Use batch endpoints where available — our batch API lets you combine up to 100 individual requests into a single call. 4) Cache responses for data that doesn't change frequently. If you need higher limits, Enterprise plans offer 5,000 RPM with dedicated API infrastructure."),
        ("Customer", "Thanks for that explanation — the token bucket info is really helpful. I think the micro-bursts are exactly our issue. We'll implement the queuing approach. Can you point me to the batch API documentation?"),
        ("Agent", "The batch API documentation is available at docs.acmecloud.io/api/batch-operations. Here's a quick overview: The batch endpoint accepts POST requests to /v2/batch with a JSON body containing an array of operations. Each operation specifies the method, path, and body of the individual request. Response includes results for all operations with individual status codes. Key limits: maximum 100 operations per batch, 5MB total request size, and 30-second timeout. The batch API documentation includes examples in Python, Java, and Node.js. There's also a section on error handling for partial batch failures, where some operations succeed while others fail. I'd also recommend our API Best Practices guide at docs.acmecloud.io/api/best-practices, which covers rate limit optimization, caching strategies, and efficient pagination patterns."),
        ("Customer", "Excellent, I'll share those docs with our development team. I think that covers everything I needed today. Thanks for being so thorough and patient with all my questions!"),
        ("Agent", "You're welcome! It was a pleasure helping you today. To recap everything we covered: account lockout resolution, password reset, 2FA discussion, team security policies, API key generation, PDF exports, storage optimization, temp file cleanup automation, and API rate limiting guidance. That's quite a productive session! Your support ticket #TKT-28491 has been updated with all these items for your reference. Don't hesitate to reach out if you need anything else. Good luck with your presentation and the API integration project!"),
    ]

    for role, content in filler_exchanges:
        if turn_num > 35:
            break
        turns.append(f"{role}: {content}")
        turn_num += 1

    # Pad to exactly 35 irrelevant turns if needed
    while turn_num <= 35:
        if turn_num % 2 == 1:
            turns.append(
                f"Customer: I also wanted to check — is there a way to set up automated alerts for when our "
                f"usage approaches plan limits? I'd like to get notified before we hit any caps, especially "
                f"for storage and API calls. We had an incident last quarter where we exceeded our bandwidth "
                f"allocation without realizing it and got unexpected overage charges."
            )
        else:
            turns.append(
                f"Agent: Absolutely! You can configure usage alerts in your Organization Settings under "
                f"'Alerts & Notifications'. Set percentage thresholds (we recommend 75% and 90%) for each "
                f"resource type including storage, compute hours, API calls, and bandwidth. Alerts can be "
                f"sent via email, Slack webhook, or PagerDuty integration. You can also set up automatic "
                f"scaling policies for compute resources to handle traffic spikes without manual intervention. "
                f"For the bandwidth overage situation you mentioned, I'd recommend setting a hard cap in "
                f"addition to the alert — go to Billing > Usage Caps to set maximum spend limits per resource."
            )
        turn_num += 1

    # Turns 36-40: Relevant billing dispute conversation
    relevant_exchanges = [
        ("Customer", "Hi again, this is a separate issue from before. I just reviewed my latest invoice (INV-2026-0847) and I see two charges of $299.99 for my Pro subscription. It looks like I was charged twice on March 1st. One charge at 12:01 AM UTC and another at 12:03 AM UTC. This has never happened before and I'm concerned. Can you look into this and process a refund for the duplicate charge? My customer ID is cust_js_8847."),
        ("Agent", "I'm so sorry about the duplicate charge — that's definitely not supposed to happen. Let me pull up your account and billing history right away to verify the details. I can see invoice INV-2026-0847 and you're right — there are two identical charges of $299.99 on March 1st. This appears to be a billing system error that occurred during our scheduled billing run. I want to assure you that this will be resolved. Let me initiate a refund for the duplicate charge immediately."),
        ("Customer", "Thank you for confirming it quickly. Yes, please process the refund. How long will it take to see the money back in my account? Also, I noticed the invoice total doesn't look right because of the double charge — will the invoice be corrected too?"),
        ("Agent", "I've initiated the refund of $299.99 through our refund processor for the duplicate charge on invoice INV-2026-0847. Here are the details: Refund amount: $299.99 (full duplicate charge). Reason: Billing system duplicate charge error. Plus, per our billing policy for valid duplicate charge disputes, you'll receive an additional 10% service credit of $30.00 applied to your next invoice. The refund typically takes 5-7 business days to appear on your statement, depending on your bank. Regarding the invoice — yes, a corrected invoice will be generated automatically once the refund is processed. You'll receive both the refund confirmation and the corrected invoice via email. Is there anything else about your billing I can help with?"),
        ("Customer", "That's great, thank you for the service credit too — I appreciate that. No, that covers the billing issue. You've been really helpful with everything today!"),
    ]

    for role, content in relevant_exchanges:
        turns.append(f"{role}: {content}")

    return turns


def _generate_few_shot_offline() -> list[str]:
    """Generate 15 few-shot examples using templates (~10K tokens total)."""
    examples: list[str] = []

    # 3 relevant billing/refund examples
    relevant = [
        ("billing_dispute",
         "I was charged $149.99 for my subscription but I downgraded to the Basic plan last week which is only $29.99. Why am I still being charged the old amount?",
         "I completely understand your concern about being charged the higher amount after your downgrade. Let me look into this for you right away. I've pulled up your account and I can see that your downgrade request was submitted on March 7th, which falls within the current billing cycle that started on March 1st. Our billing system processes downgrades at the start of the next billing cycle, so the $149.99 charge reflects your plan at the time of billing. However, since you downgraded within the first 14 days of the billing cycle, you're eligible for a pro-rated refund of the difference. I'll process a refund of $120.00 ($149.99 - $29.99) right now. You should see this reflected on your statement within 5-7 business days. Starting next month on April 1st, you'll be billed $29.99 for the Basic plan."),
        ("refund_request",
         "I signed up for a free trial of the Enterprise plan but was charged $499.99 when it ended. I thought I cancelled before the trial was over!",
         "I'm sorry to hear about this unexpected charge — I can understand how frustrating that must be. Let me investigate your trial and billing timeline. Looking at your account, I can see that your Enterprise free trial started on February 15th, giving you a 30-day trial ending on March 17th. Our records show that a cancellation was initiated on March 16th at 11:45 PM UTC, which is within the trial period. However, it appears the cancellation may not have fully processed before the automatic billing triggered at midnight UTC. This looks like a processing timing issue on our end. I'll process a full refund of $499.99 immediately as the cancellation was clearly submitted before the trial ended. You should see the refund on your statement within 5-7 business days. I've also confirmed that your account has been properly downgraded to the free tier."),
        ("invoice_question",
         "I see a line item on my invoice labeled 'Overage - Compute Hours' for $87.50. I don't understand what this charge is for. Can you break it down?",
         "I'd be happy to explain that charge in detail. The 'Overage - Compute Hours' line item of $87.50 on your invoice represents usage that exceeded your plan's included compute allocation during the billing period. Your current Pro plan includes 500 compute hours per month. According to your usage data, you consumed 675 compute hours this period — that's 175 hours over your included allocation. Overage compute hours are billed at $0.50 per hour, so 175 hours x $0.50 = $87.50. I can see that the spike in usage occurred primarily between March 5-8, which aligns with what appears to be a batch processing workload. If this level of compute usage is becoming regular, I'd recommend considering our High-Performance Compute add-on which gives you 1,000 included hours for an additional $150/month — that would actually save you money if you're consistently exceeding 500 hours."),
    ]

    for scenario, customer_msg, agent_response in relevant:
        examples.append(
            f"Scenario: {scenario}\n"
            f"Customer: {customer_msg}\n"
            f"Agent: {agent_response}"
        )

    # 12 irrelevant examples
    irrelevant = [
        ("onboarding", "I just signed up for Acme Cloud Platform and I'm feeling a bit overwhelmed. Where do I even start?",
         "Welcome to Acme Cloud Platform! I'm excited to help you get started. Don't worry — we've designed our onboarding experience to be as smooth as possible. Here's what I'd recommend: First, complete your profile setup in Account Settings — this takes about 2 minutes. Then, head to our Getting Started guide at docs.acmecloud.io/quickstart, which walks you through the key features step by step. I'd suggest starting with our Cloud Storage setup since it's the most commonly used feature. We also have a series of 5-minute video tutorials available in the Learning Center. If you're a developer, our API quickstart guide will have you making your first API call within 10 minutes. And remember, you can always reach out to us for help at any point in your journey."),
        ("feature_request", "Do you have a dark mode option for the dashboard? I work late at night and the bright white interface is hard on my eyes.",
         "Thank you for that feedback — you're not the first customer to ask about dark mode! While we don't currently have a built-in dark mode option for the dashboard, I want you to know that it's actually on our product roadmap and is scheduled for our Q3 release. In the meantime, I have a couple of workarounds that might help: You can use your browser's built-in dark mode extension (like Dark Reader for Chrome/Firefox) which works quite well with our interface. Additionally, if you're using macOS or Windows, the system-level dark mode settings can help reduce overall brightness. I've added your request to our feature voting system, which helps our product team prioritize development. Every vote counts!"),
        ("password_reset", "I can't remember my password and the reset email isn't arriving. I've checked spam and everything.",
         "I understand how frustrating that can be, especially when you need to access your account. Let's troubleshoot the password reset email issue together. First, let me verify your email — could you confirm the email address associated with your account? It's possible the reset is being sent to a different email. I'll also check our email delivery logs to see if the reset email was sent successfully and if there were any bounces. Sometimes corporate email filters or firewalls can block automated emails. If we can't resolve the email delivery, I can perform a manual password reset on my end and set a temporary password for you. For future reference, I'd also recommend setting up a backup email address on your account so you have an alternative recovery method."),
        ("feature_question", "Can I create custom dashboards with my own widgets and metrics?",
         "Yes, absolutely! Custom dashboards are one of our most popular features. Here's how to create one: Navigate to Dashboards > Create New Dashboard. You'll see a drag-and-drop editor where you can add various widget types including line charts, bar graphs, pie charts, tables, counters, and custom HTML widgets. Each widget can be connected to any metric in your account — usage, performance, billing, API calls, etc. You can set custom time ranges, refresh intervals, and alert thresholds for each widget. Dashboards can be shared with team members or kept private. Pro and Enterprise plans also support embedding dashboards in external tools via iframe. We support up to 20 widgets per dashboard and you can create up to 50 dashboards."),
        ("integration", "Does your platform integrate with Slack? We'd like to get notifications about system alerts in our team channels.",
         "Great news — we have a native Slack integration! Setting it up is quick: Go to Integrations > Slack > Connect. You'll be redirected to Slack to authorize the Acme Cloud Platform app. Once connected, you can configure which notifications go to which channels. Most teams set up a #cloud-alerts channel for system notifications and a #billing channel for billing alerts. You can customize the notification types: system health alerts, usage threshold warnings, billing notifications, team member changes, and security events. Each notification type can be routed to a different channel. The integration also supports Slack slash commands — /acme status shows your system status, /acme usage shows current resource usage. We also integrate with Microsoft Teams, PagerDuty, and generic webhooks if you use those."),
        ("technical_support", "Our application is getting timeout errors when connecting to the database service. It was working fine yesterday.",
         "I'm sorry to hear about the timeout errors — let me help you diagnose this right away. First, let me check the health status of our database service in your region. I can see that all database services are operational with normal latency. Let's troubleshoot from your application side: 1) Check your connection pool settings — a common cause of timeouts is exhausted connection pools. Our recommended settings are max_connections=20, idle_timeout=300s. 2) Verify your connection string hasn't changed — check for any recent configuration deployments. 3) Look at your application's network egress — if you've changed VPC or security group settings, that could block connections. 4) Check your database's current connections count vs. the maximum allowed for your plan. If you're on the SQL Standard plan, you have 100 max connections. I'd also recommend checking our status page at status.acmecloud.io for any recent incidents that might have affected connectivity."),
        ("account_management", "How do I add a new team member to our organization?",
         "Adding a new team member is straightforward! Here's the process: 1) Go to Organization > Team Members > Invite New Member. 2) Enter their email address. 3) Select their role: Viewer (read-only), Developer (read + deploy), Admin (full access), or Owner (full access + billing). 4) Optionally, assign them to specific projects or resource groups. 5) Click 'Send Invitation'. They'll receive an email with a link to join your organization. If they already have an Acme Cloud Platform account, they'll be able to join immediately. If not, the link will guide them through account creation. You can manage pending invitations and revoke access at any time from the Team Members page. Your Pro plan includes up to 25 team members. Need more? Enterprise plans offer unlimited team members."),
        ("compliance", "We need to provide our auditors with proof of your SOC 2 compliance. Can you send us the certification?",
         "Absolutely — we're happy to support your compliance needs. Acme Cloud Platform maintains SOC 2 Type II certification, and we can provide the necessary documentation for your auditors. Here's how to access it: 1) Go to Account Settings > Compliance > Certifications. You'll find our current SOC 2 Type II report available for download. 2) We also provide a CAIQ (Consensus Assessments Initiative Questionnaire) and a detailed security whitepaper. 3) For additional compliance documentation or specific auditor requests, you can submit a request through our Compliance Portal at compliance.acmecloud.io. We typically fulfill documentation requests within 2 business days. We also maintain ISO 27001 certification and are GDPR and CCPA compliant. If your auditors need to speak directly with our compliance team, I can arrange a call — just let me know preferred dates and times."),
        ("migration", "We're considering migrating from AWS S3 to your Cloud Storage. What does the migration process look like?",
         "Excellent question! We've helped many customers migrate from AWS S3 and have a well-established migration path. Here's an overview: 1) Assessment Phase (1-2 days): Our migration tool scans your S3 buckets and provides a report on total data volume, object counts, access patterns, and estimated migration time. 2) Planning: We create a migration plan that includes data mapping (S3 bucket to Acme Cloud Storage container), access control translation (IAM policies to Acme permissions), and a timeline. 3) Migration Execution: Our migration service supports parallel transfer of up to 10 Gbps. For most customers, we recommend a phased migration starting with non-production data. 4) Validation: Checksum verification ensures data integrity. 5) Cutover: DNS and application configuration updates. We offer free migration support for Enterprise customers and a 90-day price match guarantee. The migration tool is available at tools.acmecloud.io/migrate. Would you like me to set up an initial assessment?"),
        ("mobile_app", "Is there a mobile app for managing our cloud resources on the go?",
         "Yes! We have mobile apps for both iOS and Android. The Acme Cloud Platform mobile app gives you essential management capabilities on the go: view real-time dashboards, receive push notifications for alerts, manage team members, review billing, check system status, and even perform basic resource management like starting/stopping compute instances. The app supports biometric authentication (Face ID, Touch ID, fingerprint) for secure access. You can download it from the App Store or Google Play — search for 'Acme Cloud Platform'. Pro tip: The mobile app also supports a widget for your home screen that shows your top 3 metrics at a glance. It's updated in real-time and is really handy for quick monitoring checks."),
        ("training", "Do you offer any training or certification programs for platform administrators?",
         "We have a comprehensive training and certification program! Here's what's available: 1) Acme Cloud Fundamentals (free): A self-paced online course covering platform basics. Takes about 4 hours. 2) Acme Cloud Administrator (paid): A 2-day instructor-led course covering advanced administration, security, and optimization. Available virtually or in select cities. 3) Acme Cloud Architect (paid): A 3-day deep-dive for solution architects covering design patterns, high availability, and disaster recovery. Each course includes a certification exam. Certifications are valid for 2 years. Enterprise customers receive complimentary training seats as part of their plan. We also have a library of on-demand video tutorials, webinars, and hands-on labs in our Learning Center at learn.acmecloud.io. Monthly live Q&A sessions with our engineering team are open to all customers."),
        ("data_center", "Which regions and data centers do you operate in?",
         "Acme Cloud Platform currently operates in 8 regions across 3 continents: North America: US-East-1 (Virginia), US-East-2 (Ohio), US-West-1 (Oregon), CA-Central-1 (Montreal). Europe: EU-West-1 (Ireland), EU-Central-1 (Frankfurt). Asia-Pacific: AP-Southeast-1 (Singapore), AP-Northeast-1 (Tokyo). Each region has a minimum of 3 availability zones for high availability. All data centers are Tier IV certified with redundant power, cooling, and networking. We're expanding with 2 new regions planned for 2026: South America (Sao Paulo) and Middle East (Bahrain). For data sovereignty requirements, you can configure your resources to remain in specific regions. Cross-region replication is available on Pro and Enterprise plans for disaster recovery. Our network backbone provides sub-50ms latency between regions."),
    ]

    for scenario, customer_msg, agent_response in irrelevant:
        examples.append(
            f"Scenario: {scenario}\n"
            f"Customer: {customer_msg}\n"
            f"Agent: {agent_response}"
        )

    return examples


def _generate_legal_offline() -> str:
    """Generate ToS + Privacy + Refund Policy template text (~5K tokens)."""
    tos = """ACME CLOUD PLATFORM — TERMS OF SERVICE

Last Updated: January 15, 2026

1. SUBSCRIPTION TERMS
By subscribing to Acme Cloud Platform ("the Service"), you ("the Customer") agree to these Terms of Service ("ToS"). Subscriptions are billed monthly or annually based on the plan selected during registration. Monthly subscriptions renew automatically on the billing anniversary unless cancelled at least 24 hours before renewal. Annual subscriptions require a 12-month commitment and are billed upfront with a 20% discount applied to the equivalent monthly rate.

The Service reserves the right to modify pricing with 60 days written notice. Price increases do not apply to active annual subscriptions until the renewal date. Customers on promotional or legacy pricing will be migrated to current pricing upon plan changes.

2. ACCEPTABLE USE
The Service may be used for lawful business purposes only. Customers agree not to: (a) use the Service to store, process, or transmit content that violates applicable laws or regulations; (b) attempt to gain unauthorized access to other accounts, systems, or networks connected to the Service; (c) interfere with or disrupt the integrity or performance of the Service; (d) use the Service for cryptocurrency mining, distributed denial-of-service attacks, or other resource-abusive activities; (e) resell, sublicense, or redistribute the Service without written authorization.

Violations of this acceptable use policy may result in immediate suspension or termination of the account without refund. The Service monitors usage patterns and reserves the right to throttle or restrict access to accounts exhibiting suspicious or abusive behavior.

3. TERMINATION
Either party may terminate the subscription at any time. Customer-initiated cancellations take effect at the end of the current billing period. The Service may terminate accounts immediately for material breach of these Terms, including but not limited to: non-payment, acceptable use violations, or fraudulent activity. Upon termination, customer data is retained for 90 days to allow for data export, after which it is permanently deleted unless subject to a legal hold.

4. LIMITATION OF LIABILITY
THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED. ACME CLOUD PLATFORM SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING FROM USE OF THE SERVICE. TOTAL LIABILITY SHALL NOT EXCEED THE FEES PAID BY CUSTOMER IN THE 12 MONTHS PRECEDING THE CLAIM. This limitation applies regardless of the form of action, whether in contract, tort, strict liability, or otherwise."""

    privacy = """ACME CLOUD PLATFORM — PRIVACY POLICY

Last Updated: January 15, 2026

1. DATA COLLECTION
Acme Cloud Platform collects the following categories of personal information: (a) Account Information: name, email address, company name, phone number, billing address; (b) Payment Information: credit card details, banking information (processed and stored by our PCI-DSS compliant payment processor — we do not store full card numbers); (c) Usage Data: login timestamps, feature usage patterns, API call volumes, storage consumption, compute utilization; (d) Technical Data: IP addresses, browser type, operating system, device identifiers; (e) Communication Data: support tickets, chat transcripts, feedback submissions.

We collect data through direct customer input, automated system logging, cookies and similar tracking technologies, and third-party integrations authorized by the customer.

2. DATA RETENTION
Active account data is retained for the duration of the subscription plus 90 days following cancellation. Billing records are retained for 7 years as required by tax and financial regulations. Usage logs are retained for 12 months for operational purposes. Support tickets are retained for 3 years. Upon receipt of a valid deletion request, personal data is removed from active systems within 30 days and from backup systems within 90 days, unless subject to a legal hold or regulatory requirement.

3. USER RIGHTS
Customers have the following rights regarding their personal data: (a) Right of Access: Request a copy of all personal data we hold about you; (b) Right to Rectification: Request correction of inaccurate personal data; (c) Right to Erasure: Request deletion of personal data (subject to legal retention requirements); (d) Right to Portability: Receive personal data in a structured, machine-readable format; (e) Right to Object: Object to processing of personal data for direct marketing purposes; (f) Right to Restrict Processing: Request limitation of data processing under certain circumstances.

4. GDPR/CCPA COMPLIANCE
For customers in the European Economic Area (EEA), we process personal data in compliance with the General Data Protection Regulation (GDPR). Our lawful bases for processing include: contract performance, legitimate interest, legal obligation, and consent. We have appointed a Data Protection Officer (DPO) reachable at dpo@acmecloud.io.

For California residents, we comply with the California Consumer Privacy Act (CCPA). You have the right to: know what personal information is collected, request deletion, opt out of sale of personal information (note: we do not sell personal information), and non-discrimination for exercising your rights. To submit a CCPA request, contact privacy@acmecloud.io or call 1-800-ACME-PRIVACY.

5. SECURITY MEASURES
We implement industry-standard security measures including: AES-256 encryption at rest, TLS 1.3 encryption in transit, multi-factor authentication, regular penetration testing, SOC 2 Type II certified data centers, real-time intrusion detection, and 24/7 security monitoring. We conduct annual third-party security audits and maintain a responsible disclosure program for security researchers."""

    refund = """ACME CLOUD PLATFORM — REFUND POLICY

Last Updated: January 15, 2026

1. ELIGIBILITY
Refunds are available under the following circumstances: (a) Duplicate charges resulting from billing system errors; (b) Charges applied after a valid cancellation request was submitted; (c) Mid-cycle plan downgrades within the first 14 days of a billing cycle (pro-rated); (d) Free trial auto-conversion charges where cancellation was submitted before the trial end date; (e) Service outages exceeding our SLA commitments (credit applied as per SLA terms).

Refunds are NOT available for: (a) Voluntary non-use of the service during a paid period; (b) Charges older than 90 days; (c) Annual plan cancellations after 30 days (pro-rated credit may be applied); (d) Resource overage charges for verified usage.

2. TIMELINES
Refund requests are reviewed within 5 business days of submission. Approved refunds are processed within 2 business days of approval. Credit card refunds typically appear on statements within 5-7 business days after processing. Bank transfer refunds may take 7-10 business days. Service credits are applied immediately upon approval.

3. PROCESS
To request a refund: (a) Contact support via chat, email, or phone; (b) Provide your customer ID and the invoice ID for the charge in question; (c) Describe the reason for the refund request; (d) Our support team will verify the charge and eligibility; (e) Approved refunds are processed automatically. For duplicate charge disputes, customers receive a full refund plus a 10% service credit as compensation for the inconvenience. Refunds exceeding $500 require supervisor approval, which adds 1-2 business days to the processing time."""

    return f"{tos}\n\n{'=' * 80}\n\n{privacy}\n\n{'=' * 80}\n\n{refund}"


# ---------------------------------------------------------------------------
# Online Content Generators
# ---------------------------------------------------------------------------

FAQ_SYSTEM = (
    "You are generating realistic customer support FAQ content for a fictional B2B SaaS "
    "company called 'Acme Cloud Platform'. Generate Q&A pairs that are realistic and varied "
    "but deliberately include redundancy — similar questions phrased differently covering the "
    "same underlying answer."
)

CATALOG_SYSTEM = (
    "You are generating a comprehensive product catalog for a fictional B2B SaaS company "
    "called 'Acme Cloud Platform'."
)

CONV_SYSTEM = (
    "You are generating realistic customer support conversation history for Acme Cloud Platform."
)

FEW_SHOT_SYSTEM = (
    "You are generating few-shot example interactions for a customer support agent at "
    "Acme Cloud Platform."
)

LEGAL_SYSTEM = (
    "You are generating legal document excerpts for a fictional B2B SaaS company called "
    "'Acme Cloud Platform'."
)


def _invoke_with_retry(
    client: BedrockClient,
    system: str,
    user_prompt: str,
    model_class: type,
    max_tokens: int = 16000,
    label: str = "",
) -> object:
    """Invoke the API and parse JSON, retrying with reasoning disabled on failure.

    Reasoning tokens consume the output budget. If the first attempt produces
    an empty or truncated response, retry with reasoning_tier="disabled" so all
    max_tokens go toward content output.
    """
    messages = [{"role": "user", "content": [{"text": user_prompt}]}]

    # Attempt 1: medium reasoning
    text, _, _ = client.invoke(
        system=system,
        messages=messages,
        reasoning_tier="medium",
        temperature=0.7,
        max_tokens=max_tokens,
    )
    try:
        return parse_llm_json(text, model_class)
    except ValueError:
        print(f"    {label}: reasoning consumed output budget, retrying without reasoning...")

    # Attempt 2: no reasoning — all tokens available for content
    text, _, _ = client.invoke(
        system=system,
        messages=messages,
        reasoning_tier="disabled",
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return parse_llm_json(text, model_class)


def _generate_faq_online(client: BedrockClient) -> str:
    """Generate 100 FAQ entries via Nova API in 5 batches of 20."""
    all_faqs: list[str] = []
    topics_str = ", ".join(FAQ_TOPICS)

    for batch_idx in range(FAQ_BATCHES):
        previous_block = ""
        if all_faqs:
            samples = all_faqs[-3:]
            previous_block = (
                "Here are some examples from previous batches to create cross-batch redundancy — "
                "rephrase some of these differently:\n\n" + "\n---\n".join(samples)
            )

        user_prompt = (
            f"Generate a batch of {FAQ_BATCH_SIZE} FAQ entries for Acme Cloud Platform. "
            f"Include these topic areas: {topics_str}. "
            f"For each entry: Make the question specific and realistic, make the answer comprehensive "
            f"(4-6 detailed paragraphs), deliberately create semantic redundancy: 40% of entries should "
            f"be near-duplicates of previous entries rephrased differently. "
            f"{previous_block}. "
            f'Return ONLY a valid JSON object. No markdown, no preamble. '
            f'Schema: {{"faqs": [{{"question": "str", "answer": "str", "topic": "str"}}]}}'
        )

        print(f"  Generating FAQ batch {batch_idx + 1}/{FAQ_BATCHES}...")
        batch = _invoke_with_retry(
            client, FAQ_SYSTEM, user_prompt, FaqBatchResponse,
            max_tokens=16000, label=f"FAQ batch {batch_idx + 1}",
        )
        for entry in batch.faqs:
            all_faqs.append(f"Q: {entry.question}\n\nA: {entry.answer}")

    return "\n\n---\n\n".join(all_faqs)


def _generate_catalog_online(client: BedrockClient) -> str:
    """Generate 200 product entries via Nova API in 5 batches of 40."""
    all_products: list[str] = []

    for batch_idx in range(CATALOG_BATCHES):
        num_products = CATALOG_BATCH_SIZE
        num_discontinued = CATALOG_DISCONTINUED // CATALOG_BATCHES

        user_prompt = (
            f"Generate a product catalog for Acme Cloud Platform with {num_products} product entries. "
            f"Include {num_discontinued} discontinued products (marked as status: 'discontinued'). "
            f"Products should cover: cloud storage tiers, compute instances, database services, "
            f"monitoring tools, security features, and support plans. Each product entry should be "
            f"detailed (name, SKU, description, pricing, features, limitations, status). Include at "
            f"least 5-6 paragraphs per product with comprehensive technical details. "
            f'Return ONLY a valid JSON object. No markdown, no preamble. '
            f'Schema: {{"products": [{{"sku": "str", "name": "str", "status": "active|discontinued", '
            f'"description": "str", "pricing": "str", "features": ["str"], "limitations": ["str"]}}]}}'
        )

        print(f"  Generating catalog batch {batch_idx + 1}/{CATALOG_BATCHES}...")
        batch = _invoke_with_retry(
            client, CATALOG_SYSTEM, user_prompt, CatalogBatchResponse,
            max_tokens=20000, label=f"Catalog batch {batch_idx + 1}",
        )
        for product in batch.products:
            features_str = "\n".join(f"- {f}" for f in product.features)
            limitations_str = "\n".join(f"- {l}" for l in product.limitations)
            entry = (
                f"SKU: {product.sku}\n"
                f"Name: {product.name}\n"
                f"Status: {product.status}\n"
                f"Pricing: {product.pricing}\n\n"
                f"{product.description}\n\n"
                f"Features:\n{features_str}\n\n"
                f"Limitations:\n{limitations_str}"
            )
            all_products.append(entry)

    return "\n\n===\n\n".join(all_products)


def _generate_conversation_online(client: BedrockClient) -> list[str]:
    """Generate 40-turn conversation via Nova API."""
    user_prompt = (
        f"Generate a {CONV_TURNS}-turn customer support conversation for Acme Cloud Platform. "
        f"Conversation structure: Turns 1-{CONV_IRRELEVANT_TURNS}: Customer and agent resolving a "
        f"DIFFERENT past issue (e.g., password reset, feature request). Fully resolved by turn "
        f"{CONV_IRRELEVANT_TURNS}. Turns {CONV_RELEVANT_START}-{CONV_TURNS}: Current billing issue — "
        f"customer is disputing a double charge on their latest invoice. Still unresolved (this is the "
        f"active issue). Format each turn as role: 'customer' or 'agent' with realistic dialogue. "
        f"Make irrelevant turns completely orthogonal to the current billing issue. Make each message "
        f"detailed and verbose (customers explain at length, agents give thorough responses with "
        f"step-by-step instructions). "
        f'Return ONLY a valid JSON object. No markdown, no preamble. '
        f'Schema: {{"turns": [{{"turn_number": int, "role": "customer|agent", "content": "str"}}]}}'
    )

    print("  Generating conversation history...")
    conv = _invoke_with_retry(
        client, CONV_SYSTEM, user_prompt, ConvResponse,
        max_tokens=20000, label="Conversation",
    )
    return [f"{turn.role.capitalize()}: {turn.content}" for turn in conv.turns]


def _generate_few_shot_online(client: BedrockClient) -> list[str]:
    """Generate 15 few-shot examples via Nova API."""
    user_prompt = (
        f"Generate {NUM_FEW_SHOT} few-shot example interactions for a customer support agent. "
        f"Distribution: {FEW_SHOT_RELEVANT} examples: billing disputes, refund requests, invoice "
        f"questions (match eval query scenarios). {FEW_SHOT_IRRELEVANT} examples: completely different "
        f"scenarios (onboarding, feature questions, password resets). Each example: customer message + "
        f"ideal agent response (5-8 detailed sentences). "
        f'Return ONLY a valid JSON object. No markdown, no preamble. '
        f'Schema: {{"examples": [{{"scenario_type": "str", "customer_message": "str", '
        f'"ideal_response": "str"}}]}}'
    )

    print("  Generating few-shot examples...")
    batch = _invoke_with_retry(
        client, FEW_SHOT_SYSTEM, user_prompt, FewShotResponse,
        max_tokens=16000, label="Few-shot",
    )
    return [
        f"Scenario: {ex.scenario_type}\n"
        f"Customer: {ex.customer_message}\n"
        f"Agent: {ex.ideal_response}"
        for ex in batch.examples
    ]


def _generate_legal_online(client: BedrockClient) -> str:
    """Generate legal documents via Nova API."""
    user_prompt = (
        "Generate three legal document excerpts for Acme Cloud Platform: "
        "1. Terms of Service excerpt (~800 words) covering subscription terms, acceptable use, "
        "termination, liability. "
        "2. Privacy Policy excerpt (~800 words) covering data collection, retention, user rights, "
        "GDPR/CCPA compliance. "
        "3. Refund Policy excerpt (~400 words) covering eligibility, timelines, process. "
        'Return ONLY a valid JSON object. No markdown, no preamble. '
        'Schema: {"documents": [{"title": "str", "content": "str"}]}'
    )

    print("  Generating legal documents...")
    legal = _invoke_with_retry(
        client, LEGAL_SYSTEM, user_prompt, LegalResponse,
        max_tokens=16000, label="Legal",
    )
    return ("\n\n" + "=" * 80 + "\n\n").join(doc.content for doc in legal.documents)


# ---------------------------------------------------------------------------
# Token Adjustment
# ---------------------------------------------------------------------------

def _pad_text_to_tokens(text: str, target: int, section_name: str) -> str:
    """Append filler content if text is under target token count."""
    current = estimate_tokens(text)
    if current >= target:
        return text

    deficit = target - current
    print(f"  Padding {section_name}: {current} -> {target} tokens (+{deficit})")

    filler_paragraphs = [
        f"\n\n[Additional {section_name} Notes — Entry {{i}}]\n"
        f"This supplementary section provides extended documentation and context for "
        f"Acme Cloud Platform's {section_name.lower()} area. Our platform is designed with "
        f"enterprise-grade reliability and security, backed by SOC 2 Type II certification and "
        f"comprehensive SLA guarantees. Customers benefit from our dedicated support infrastructure, "
        f"which includes 24/7 monitoring, automated incident response, and proactive capacity planning. "
        f"Our engineering team continuously optimizes system performance, with bi-weekly releases that "
        f"include bug fixes, security patches, and feature enhancements. All changes undergo rigorous "
        f"testing including unit tests, integration tests, load tests, and staged rollout with automatic "
        f"rollback capabilities. Customer data protection remains our top priority, with encryption at "
        f"rest (AES-256), in transit (TLS 1.3), and comprehensive audit logging for all data access events. "
        f"For detailed technical specifications, architecture diagrams, and API documentation, please "
        f"refer to our comprehensive documentation portal at docs.acmecloud.io."
    ]

    filler_text = filler_paragraphs[0]
    tokens_per_filler = estimate_tokens(filler_text)
    reps_needed = max(1, deficit // max(tokens_per_filler, 1)) + 1

    padding = ""
    for i in range(reps_needed):
        padding += filler_text.replace("{i}", str(i + 1))
        if estimate_tokens(text + padding) >= target:
            break

    return text + padding


def _adjust_catalog_for_total(
    catalog_text: str,
    current_total: int,
    catalog_tokens: int,
) -> str:
    """Adjust catalog content to bring total within 200K-220K range."""
    if TOKEN_TARGET_MIN <= current_total <= TOKEN_TARGET_MAX:
        return catalog_text

    if current_total < TOKEN_TARGET_MIN:
        # Need more tokens — pad catalog
        needed = TOKEN_TARGET_TOTAL - current_total
        new_target = catalog_tokens + needed
        print(f"  Total {current_total} < {TOKEN_TARGET_MIN}. Padding catalog by {needed} tokens.")
        return _pad_text_to_tokens(catalog_text, new_target, "Product Catalog")

    # current_total > TOKEN_TARGET_MAX — trim catalog
    excess = current_total - TOKEN_TARGET_TOTAL
    print(f"  Total {current_total} > {TOKEN_TARGET_MAX}. Trimming catalog by ~{excess} tokens.")
    # Find a boundary to trim at
    separator = "\n\n===\n\n"
    parts = catalog_text.split(separator)
    trimmed_parts = []
    running = 0
    target_catalog = catalog_tokens - excess
    for part in parts:
        part_tokens = estimate_tokens(part)
        if running + part_tokens > target_catalog and trimmed_parts:
            break
        trimmed_parts.append(part)
        running += part_tokens

    return separator.join(trimmed_parts)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _assemble_payload(
    system_prompt: str,
    faq_text: str,
    catalog_text: str,
    conversation_turns: list[str],
    tool_defs: list[dict],
    few_shot_examples: list[str],
    legal_text: str,
    eval_queries: list[EvalQuery],
) -> ContextPayload:
    """Build ContextPayload from all generated content."""
    sections: list[ContextSection] = []

    # System prompt
    sections.append(ContextSection(
        id="sys_001",
        label="System Prompt — Agent Alex",
        section_type=SectionType.SYSTEM_PROMPT,
        content=system_prompt,
        token_count=estimate_tokens(system_prompt),
        metadata={"agent_name": "Alex", "company": "Acme Cloud Platform"},
    ))

    # FAQ
    sections.append(ContextSection(
        id="faq_001",
        label="Company FAQ — Acme Cloud Platform",
        section_type=SectionType.RAG_DOCUMENT,
        content=faq_text,
        token_count=estimate_tokens(faq_text),
        metadata={"entry_count": faq_text.count("Q:"), "redundancy_rate": 0.4},
    ))

    # Catalog
    sections.append(ContextSection(
        id="catalog_001",
        label="Product Catalog — Acme Cloud Platform",
        section_type=SectionType.RAG_DOCUMENT,
        content=catalog_text,
        token_count=estimate_tokens(catalog_text),
        metadata={"product_count": catalog_text.count("SKU:"), "discontinued_count": catalog_text.count("discontinued")},
    ))

    # Conversation turns
    for i, turn in enumerate(conversation_turns, 1):
        sections.append(ContextSection(
            id=f"conv_{i:03d}",
            label=f"Conversation Turn {i}",
            section_type=SectionType.CONVERSATION_TURN,
            content=turn,
            token_count=estimate_tokens(turn),
            metadata={"turn_number": i, "relevant": i >= CONV_RELEVANT_START},
        ))

    # Tool definitions
    for tool in tool_defs:
        sections.append(ContextSection(
            id=tool["id"],
            label=tool["label"],
            section_type=SectionType.TOOL_DEFINITION,
            content=tool["content"],
            token_count=estimate_tokens(tool["content"]),
            metadata={"tool_name": json.loads(tool["content"]).get("name", "")},
        ))

    # Few-shot examples
    for i, example in enumerate(few_shot_examples, 1):
        sections.append(ContextSection(
            id=f"shot_{i:03d}",
            label=f"Few-Shot Example {i}",
            section_type=SectionType.FEW_SHOT_EXAMPLE,
            content=example,
            token_count=estimate_tokens(example),
            metadata={"example_number": i, "relevant": i <= FEW_SHOT_RELEVANT},
        ))

    # Legal
    sections.append(ContextSection(
        id="legal_001",
        label="Legal Disclaimers — ToS, Privacy, Refund Policy",
        section_type=SectionType.CUSTOM,
        content=legal_text,
        token_count=estimate_tokens(legal_text),
        metadata={"document_count": 3},
    ))

    total_tokens = sum(s.token_count for s in sections)

    return ContextPayload(
        sections=sections,
        evaluation_queries=eval_queries,
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# Cost Logging
# ---------------------------------------------------------------------------

def _log_cost(mode: str, client: BedrockClient | None) -> None:
    """Append cost entry to data/cost_log.json."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "generate_demo_payload.py",
        "mode": mode,
        "api_calls": client.total_api_calls if client else 0,
        "input_tokens": client.total_input_tokens if client else 0,
        "output_tokens": client.total_output_tokens if client else 0,
        "cost_usd": client.total_cost if client else 0.0,
    }

    COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if COST_LOG_PATH.exists():
        try:
            existing = json.loads(COST_LOG_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(entry)
    COST_LOG_PATH.write_text(json.dumps(existing, indent=2))
    print(f"\nCost log written to {COST_LOG_PATH}")
    if client:
        print(f"  API calls: {client.total_api_calls}")
        print(f"  Input tokens: {client.total_input_tokens:,}")
        print(f"  Output tokens: {client.total_output_tokens:,}")
        print(f"  Estimated cost: ${client.total_cost:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the demo payload."""
    parser = argparse.ArgumentParser(description="Generate ContextForge demo payload")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use template content without API calls",
    )
    args = parser.parse_args()

    mode = "offline" if args.offline else "online"
    print(f"=== ContextForge Demo Payload Generator ({mode} mode) ===\n")

    # Ensure output directories exist
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    # Hand-written content
    print("Building hand-written content...")
    system_prompt = _build_system_prompt()
    tool_defs = _build_tool_definitions()
    eval_queries = _build_eval_queries()

    client: BedrockClient | None = None

    if args.offline:
        print("\nGenerating offline content (templates)...")
        faq_text = _generate_faq_offline()
        catalog_text = _generate_catalog_offline()
        conversation_turns = _generate_conversation_offline()
        few_shot_examples = _generate_few_shot_offline()
        legal_text = _generate_legal_offline()
    else:
        print("\nGenerating online content (Nova API)...")
        client = BedrockClient()
        faq_text = _generate_faq_online(client)
        catalog_text = _generate_catalog_online(client)
        conversation_turns = _generate_conversation_online(client)
        few_shot_examples = _generate_few_shot_online(client)
        legal_text = _generate_legal_online(client)

    # Pad sections to meet token targets
    print("\nAdjusting section sizes...")
    faq_text = _pad_text_to_tokens(faq_text, TOKEN_TARGET_FAQ, "FAQ")
    catalog_text = _pad_text_to_tokens(catalog_text, TOKEN_TARGET_CATALOG, "Product Catalog")
    legal_text = _pad_text_to_tokens(legal_text, TOKEN_TARGET_LEGAL, "Legal Disclaimers")

    # Assemble payload (before catalog flex adjustment)
    payload = _assemble_payload(
        system_prompt=system_prompt,
        faq_text=faq_text,
        catalog_text=catalog_text,
        conversation_turns=conversation_turns,
        tool_defs=tool_defs,
        few_shot_examples=few_shot_examples,
        legal_text=legal_text,
        eval_queries=eval_queries,
    )

    # Flex adjustment on catalog
    non_catalog_tokens = payload.total_tokens - estimate_tokens(catalog_text)
    catalog_text = _adjust_catalog_for_total(
        catalog_text, payload.total_tokens, estimate_tokens(catalog_text)
    )

    # Reassemble if catalog was adjusted
    if estimate_tokens(catalog_text) != payload.sections[2].token_count:
        payload = _assemble_payload(
            system_prompt=system_prompt,
            faq_text=faq_text,
            catalog_text=catalog_text,
            conversation_turns=conversation_turns,
            tool_defs=tool_defs,
            few_shot_examples=few_shot_examples,
            legal_text=legal_text,
            eval_queries=eval_queries,
        )

    # Write payload
    PAYLOAD_PATH.write_text(json.dumps(payload.model_dump(mode="json"), indent=2))
    print(f"\nPayload written to {PAYLOAD_PATH}")

    # Write eval queries
    queries_data = [q.model_dump(mode="json") for q in eval_queries]
    QUERIES_PATH.write_text(json.dumps(queries_data, indent=2))
    print(f"Eval queries written to {QUERIES_PATH}")

    # Log cost
    _log_cost(mode, client)

    # Print summary
    print("\n=== Payload Summary ===")
    section_groups: dict[str, int] = {}
    for section in payload.sections:
        group = section.section_type.value
        section_groups[group] = section_groups.get(group, 0) + section.token_count

    for group, tokens in sorted(section_groups.items(), key=lambda x: -x[1]):
        print(f"  {group:25s}: {tokens:>8,} tokens")

    print(f"  {'─' * 36}")
    print(f"  {'TOTAL':25s}: {payload.total_tokens:>8,} tokens")
    print(f"  Sections: {len(payload.sections)}")
    print(f"  Eval queries: {len(payload.evaluation_queries)}")

    in_range = TOKEN_TARGET_MIN <= payload.total_tokens <= TOKEN_TARGET_MAX
    status = "OK" if in_range else "WARNING: outside target range"
    print(f"  Target range: {TOKEN_TARGET_MIN:,}-{TOKEN_TARGET_MAX:,} ({status})")


if __name__ == "__main__":
    main()
