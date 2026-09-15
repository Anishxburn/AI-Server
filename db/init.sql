CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'manual',
    standard_name TEXT,
    file_name TEXT,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    page_number INTEGER,
    section_reference TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY,
    user_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS qa_logs (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    is_ems_related BOOLEAN NOT NULL,
    sources_used JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS qa_embeddings (
    id UUID PRIMARY KEY,
    qa_log_id UUID NOT NULL REFERENCES qa_logs(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qa_embeddings_embedding_idx
ON qa_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY,
    qa_log_id UUID NOT NULL REFERENCES qa_logs(id) ON DELETE CASCADE,
    rating INTEGER,
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS daxview_conversations (
    id UUID PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS daxview_conversations_owner_idx
ON daxview_conversations (deployment_id, company_id, user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS daxview_turns (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES daxview_conversations(id) ON DELETE CASCADE,
    deployment_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    request_id TEXT,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS daxview_turns_unique_turn_idx
ON daxview_turns (deployment_id, conversation_id, id);

CREATE TABLE IF NOT EXISTS daxview_jobs (
    id TEXT PRIMARY KEY,
    turn_id UUID NOT NULL REFERENCES daxview_turns(id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES daxview_conversations(id) ON DELETE CASCADE,
    deployment_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    cancelled_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS daxview_jobs_unique_turn_idx
ON daxview_jobs (deployment_id, conversation_id, turn_id);

CREATE TABLE IF NOT EXISTS daxview_job_events (
    job_id TEXT NOT NULL REFERENCES daxview_jobs(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, event_id)
);
