// === Range indexes (PRD §5.3) ===
CREATE INDEX feat_layer IF NOT EXISTS FOR (f:SAEFeature) ON (f.sae_id);
CREATE INDEX feat_index IF NOT EXISTS FOR (f:SAEFeature) ON (f.index);
CREATE INDEX feat_community IF NOT EXISTS FOR (f:SAEFeature) ON (f.communityId);
CREATE INDEX prompt_source IF NOT EXISTS FOR (p:Prompt) ON (p.source);
CREATE INDEX manifold_layer IF NOT EXISTS FOR (m:Manifold) ON (m.layer_id);

// === Vector indexes (Neo4j 5.x / 2026.x syntax) ===
// Dimension 2304 = Gemma 2 2B residual stream
CREATE VECTOR INDEX feat_decoder IF NOT EXISTS
  FOR (f:SAEFeature) ON (f.decoder_vec)
  OPTIONS {indexConfig: { `vector.dimensions`: 2304, `vector.similarity_function`: 'cosine' }};

CREATE VECTOR INDEX feat_encoder IF NOT EXISTS
  FOR (f:SAEFeature) ON (f.encoder_vec)
  OPTIONS {indexConfig: { `vector.dimensions`: 2304, `vector.similarity_function`: 'cosine' }};

// Dimension 384 = sentence-transformers/all-MiniLM-L6-v2
CREATE VECTOR INDEX label_emb IF NOT EXISTS
  FOR (a:AutoInterpLabel) ON (a.embedding)
  OPTIONS {indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' }};

CREATE VECTOR INDEX waypoint_centroid IF NOT EXISTS
  FOR (w:Waypoint) ON (w.centroid)
  OPTIONS {indexConfig: { `vector.dimensions`: 2304, `vector.similarity_function`: 'cosine' }};
