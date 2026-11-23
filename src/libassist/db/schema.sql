-- LibAssist schema (SQLite). Invariants live in the database too, not only in Python:
-- CHECK constraints, foreign keys, and a partial unique index for "one active hold per member and title".

CREATE TABLE IF NOT EXISTS members (
    id          TEXT PRIMARY KEY,                       -- e.g. M-1001
    first_name  TEXT NOT NULL,
    last_name   TEXT NOT NULL,
    email       TEXT NOT NULL,
    member_type TEXT NOT NULL CHECK (member_type IN ('student', 'staff')),
    version     INTEGER NOT NULL DEFAULT 1              -- bumped on every update; exposed as the ETag
);

CREATE TABLE IF NOT EXISTS books (
    id               TEXT PRIMARY KEY,                  -- e.g. B-001
    title            TEXT NOT NULL,
    author           TEXT NOT NULL,
    year             INTEGER,
    subjects         TEXT NOT NULL DEFAULT '',          -- comma-separated
    description      TEXT NOT NULL DEFAULT '',
    total_copies     INTEGER NOT NULL CHECK (total_copies >= 0),
    available_copies INTEGER NOT NULL CHECK (available_copies >= 0),
    CHECK (available_copies <= total_copies)
);

CREATE TABLE IF NOT EXISTS loans (
    id            TEXT PRIMARY KEY,                     -- e.g. L-5001
    member_id     TEXT NOT NULL REFERENCES members (id),
    book_id       TEXT NOT NULL REFERENCES books (id),
    borrowed_on   TEXT NOT NULL,                        -- ISO date
    due_on        TEXT NOT NULL,                        -- ISO date
    renewal_count INTEGER NOT NULL DEFAULT 0 CHECK (renewal_count >= 0),
    fine_carried_cents INTEGER NOT NULL DEFAULT 0 CHECK (fine_carried_cents >= 0),
                                                        -- fine frozen when an overdue loan is renewed
    returned_on   TEXT                                  -- NULL while the book is out
);
CREATE INDEX IF NOT EXISTS idx_active_loans_by_member ON loans (member_id) WHERE returned_on IS NULL;

CREATE TABLE IF NOT EXISTS holds (
    id          TEXT PRIMARY KEY,                       -- e.g. H-7001
    member_id   TEXT NOT NULL REFERENCES members (id),
    book_id     TEXT NOT NULL REFERENCES books (id),
    status      TEXT NOT NULL CHECK (status IN ('QUEUED', 'READY_FOR_PICKUP', 'CANCELLED', 'FULFILLED', 'EXPIRED')),
    created_at  TEXT NOT NULL,                          -- ISO timestamp; defines queue order
    ready_until TEXT                                    -- ISO date, only while READY_FOR_PICKUP
);
CREATE INDEX IF NOT EXISTS idx_holds_queue ON holds (book_id, status, created_at);
-- Defence in depth: even if application code had a bug, the database refuses a second active hold.
CREATE UNIQUE INDEX IF NOT EXISTS uq_one_active_hold_per_title
    ON holds (member_id, book_id) WHERE status IN ('QUEUED', 'READY_FOR_PICKUP');

-- Stored responses for POST requests carrying an Idempotency-Key header.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    scoped_key    TEXT PRIMARY KEY,                     -- "<principal>:<Idempotency-Key>"
    request_hash  TEXT NOT NULL,                        -- fingerprint of method, path and body
    status_code   INTEGER NOT NULL,
    response_body TEXT NOT NULL,                        -- JSON, replayed verbatim on retries
    created_at    TEXT NOT NULL
);
