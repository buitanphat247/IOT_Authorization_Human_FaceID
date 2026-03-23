-- ============================================
-- SUPABASE SQL SCHEMA
-- Face Recognition System
-- Run this in Supabase SQL Editor
-- ============================================

-- 1. Face Embeddings Table
CREATE TABLE IF NOT EXISTS face_embeddings (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    embedding   FLOAT8[] NOT NULL,                -- 512-dim ArcFace vector
    quality_score FLOAT8 DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast name lookup
CREATE INDEX IF NOT EXISTS idx_face_embeddings_name 
    ON face_embeddings(name);

-- 2. Attendance Logs Table
CREATE TABLE IF NOT EXISTS attendance_logs (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    score       FLOAT8 NOT NULL,
    status      TEXT DEFAULT 'PRESENT',           -- PRESENT, ABSENT, LATE
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_attendance_logs_time 
    ON attendance_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_attendance_logs_name 
    ON attendance_logs(name);

-- 3. Enable Row Level Security (Optional but recommended)
ALTER TABLE face_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE attendance_logs ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (adjust as needed)
CREATE POLICY "Allow all for anon" ON face_embeddings
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all for anon" ON attendance_logs
    FOR ALL USING (true) WITH CHECK (true);

-- ============================================
-- DONE! Tables are ready.
-- ============================================
