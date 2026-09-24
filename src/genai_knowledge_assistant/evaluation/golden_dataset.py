"""Hand-labeled test questions for router correctness evaluation.
Each example states which of the five routes (rag, graph, memory,
web_search, general) the router should pick — this is the ground truth
against which the eval runner checks actual routing decisions."""

GOLDEN_DATASET = [
    # --- rag: answerable from ingested documents ---
    {
        "id": "g1",
        "query": "What does my uploaded contract say about the termination clause?",
        "expected_route": "rag",
    },
    {
        "id": "g2",
        "query": "Summarize the key findings from the research paper I uploaded.",
        "expected_route": "rag",
    },
    {
        "id": "g3",
        "query": "According to the document I shared, what is the refund policy?",
        "expected_route": "rag",
    },
    {
        "id": "g4",
        "query": "What deadlines are mentioned in the meeting notes I gave you?",
        "expected_route": "rag",
    },
    # --- graph: relationship / multi-hop questions ---
    {
        "id": "g5",
        "query": "How does Anthropic relate to the founder of OpenAI?",
        "expected_route": "graph",
    },
    {
        "id": "g6",
        "query": "Who works at the company that acquired the startup mentioned in my documents?",
        "expected_route": "graph",
    },
    {
        "id": "g7",
        "query": "What connects the two people mentioned in the report?",
        "expected_route": "graph",
    },
    {
        "id": "g8",
        "query": "Which organizations are linked to the technology described in my notes?",
        "expected_route": "graph",
    },
    # --- memory: recalling something from a past conversation ---
    {
        "id": "g9",
        "query": "What did I tell you my favorite programming language was?",
        "expected_route": "memory",
    },
    {
        "id": "g10",
        "query": "Do you remember what my job title is?",
        "expected_route": "memory",
    },
    {
        "id": "g11",
        "query": "What was the name of the project I mentioned last week?",
        "expected_route": "memory",
    },
    {
        "id": "g12",
        "query": "What preferences did I share with you earlier about my writing style?",
        "expected_route": "memory",
    },
    # --- web_search: current / time-sensitive information ---
    {
        "id": "g13",
        "query": "What's the latest news on the Fed's interest rate decision?",
        "expected_route": "web_search",
    },
    {
        "id": "g14",
        "query": "Who won the game last night?",
        "expected_route": "web_search",
    },
    {
        "id": "g15",
        "query": "What's the current price of Bitcoin?",
        "expected_route": "web_search",
    },
    {
        "id": "g16",
        "query": "Are there any new AI model releases this week?",
        "expected_route": "web_search",
    },
    # --- general: plain knowledge, no special routing needed ---
    {
        "id": "g17",
        "query": "What is the capital of France?",
        "expected_route": "general",
    },
    {
        "id": "g18",
        "query": "Explain how photosynthesis works.",
        "expected_route": "general",
    },
    {
        "id": "g19",
        "query": "What's the difference between a list and a tuple in Python?",
        "expected_route": "general",
    },
    {
        "id": "g20",
        "query": "Give me a fun fact about octopuses.",
        "expected_route": "general",
    },
    # --- ambiguous: could plausibly go to more than one route ---
    {
        "id": "g21",
        "query": "Tell me about the connection between the company in my report and its main competitor.",
        "expected_route": "graph",  # relationship framing, but "in my report" could pull it toward rag
        "note": "rag vs graph — relationship question, but grounded in an uploaded doc",
    },
    {
        "id": "g22",
        "query": "What's the most recent update on the project I uploaded notes about last month?",
        "expected_route": "rag",  # "recent" suggests web_search, but it's scoped to an already-uploaded doc
        "note": "rag vs web_search — 'recent' is a red herring since the source is a static upload",
    },
    {
        "id": "g23",
        "query": "Based on what I told you before, should I use PostgreSQL or MongoDB for this project?",
        "expected_route": "memory",  # recalls prior context, but sounds like a general technical question
        "note": "memory vs general — explicitly references past conversation, but the actual ask is generic advice",
    },
    {
        "id": "g24",
        "query": "How is the founder of the startup mentioned in my documents connected to any current news?",
        "expected_route": "graph",  # relationship + doc, but "current news" pulls toward web_search
        "note": "graph vs web_search — relationship framing but wants up-to-date info",
    },
    {
        "id": "g25",
        "query": "What did the article I saved say about who acquired whom?",
        "expected_route": "rag",  # asks about a document, but "who acquired whom" is a classic rag signal
        "note": "rag vs graph — ownership/relationship question but framed as document recall",
    },
    {
        "id": "g26",
        "query": "Is the pricing information in my uploaded contract still accurate today?",
        "expected_route": "rag",  # asks about a document, but "still accurate today" implies needing current info
        "note": "rag vs web_search — document lookup with an implicit freshness check",
    },
    {
        "id": "g27",
        "query": "Remind me what tools you recommended and whether anything better has come out since.",
        "expected_route": "memory",  # first half is memory, second half implies web_search
        "note": "memory vs web_search — compound question split across two different needs",
    },
    {
        "id": "g28",
        "query": "What's the relationship between the two authors of the paper I uploaded?",
        "expected_route": "graph",  # relationship question, but "paper I uploaded" is a strong rag signal
        "note": "rag vs graph — classic case, relationship question fully scoped to one document",
    },
    {
        "id": "g29",
        "query": "Based on my notes, what should I expect from the upcoming earnings call?",
        "expected_route": "rag",  # "upcoming" suggests web_search, but explicitly grounded in "my notes"
        "note": "rag vs web_search — future-tense framing over a static document",
    },
    {
        "id": "g30",
        "query": "You mentioned a company earlier — is it still operating, or has anything changed?",
        "expected_route": "memory",  # starts with recall, but "is it still operating" needs current info
        "note": "memory vs web_search — recall the entity from memory, then needs a live status check",
    },
]
