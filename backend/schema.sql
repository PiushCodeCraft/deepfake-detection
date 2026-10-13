-- DeepTrace Database Schema
CREATE TABLE IF NOT EXISTS scan_history (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255),
    prediction VARCHAR(50) NOT NULL,
    confidence NUMERIC(5, 2) NOT NULL,
    fake_probability NUMERIC(5, 2),
    real_probability NUMERIC(5, 2),
    landmarks_count INT DEFAULT 478,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
