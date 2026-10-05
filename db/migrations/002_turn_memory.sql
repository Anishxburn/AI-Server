ALTER TABLE daxview_turns ADD COLUMN IF NOT EXISTS assistant_reply TEXT;
ALTER TABLE daxview_turns ADD COLUMN IF NOT EXISTS resolved_plan JSONB NOT NULL DEFAULT '{}'::jsonb;

