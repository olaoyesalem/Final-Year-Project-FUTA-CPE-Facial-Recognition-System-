-- ============================================================
-- FRAMS Supabase Schema
-- Run this in the Supabase SQL Editor (Project → SQL Editor)
-- ============================================================

-- Students registered in the system
CREATE TABLE IF NOT EXISTS students (
    id          BIGSERIAL PRIMARY KEY,
    face_label  INTEGER   NOT NULL UNIQUE,
    name        TEXT      NOT NULL,
    matric_no   TEXT      NOT NULL UNIQUE,
    department  TEXT      NOT NULL DEFAULT 'Computer Engineering',
    is_active   BOOLEAN   NOT NULL DEFAULT TRUE,
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Face images uploaded via web enrollment
-- The Pi downloader polls this table for is_downloaded = FALSE rows
CREATE TABLE IF NOT EXISTS face_images (
    id            BIGSERIAL PRIMARY KEY,
    student_id    BIGINT    NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    face_label    INTEGER   NOT NULL,
    storage_path  TEXT      NOT NULL,   -- path inside the "face-images" bucket
    is_downloaded BOOLEAN   NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_images_downloaded
    ON face_images (is_downloaded)
    WHERE is_downloaded = FALSE;

-- Attendance logs (for cloud-side reporting — Pi syncs via sync_manager)
CREATE TABLE IF NOT EXISTS attendance_logs (
    id          BIGSERIAL PRIMARY KEY,
    student_id  BIGINT    NOT NULL REFERENCES students(id),
    course_code TEXT,
    session     TEXT,
    confidence  NUMERIC(6,4),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Storage bucket
-- Create this in the Supabase dashboard:
--   Storage → New Bucket → Name: face-images → Public: OFF
-- Then add this RLS policy so the service-role key can read/write:
-- ============================================================
-- (Run in SQL Editor after creating the bucket)
-- CREATE POLICY "service role full access"
-- ON storage.objects FOR ALL
-- USING (bucket_id = 'face-images')
-- WITH CHECK (bucket_id = 'face-images');
