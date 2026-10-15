CREATE TABLE IF NOT EXISTS ai_debug_traces (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    event TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ai_debug_traces_created_idx ON ai_debug_traces (created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS ai_debug_traces_request_idx ON ai_debug_traces (request_id, id DESC);

ALTER TABLE daxview_turns ADD COLUMN IF NOT EXISTS parent_turn_id UUID REFERENCES daxview_turns(id) ON DELETE SET NULL;
ALTER TABLE daxview_turns ADD COLUMN IF NOT EXISTS clarification_state JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS daxview_turns_parent_idx ON daxview_turns (parent_turn_id) WHERE parent_turn_id IS NOT NULL;
