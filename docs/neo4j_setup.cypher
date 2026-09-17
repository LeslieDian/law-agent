// ============================================================
// Neo4j 法律知识图谱初始化
// 用法：cat docs/neo4j_setup.cypher | cypher-shell -u neo4j -p $NEO4J_PASSWORD
// ============================================================

// ---------- 1. 唯一性约束 ----------
CREATE CONSTRAINT law_id_unique IF NOT EXISTS
FOR (n:Law) REQUIRE n.law_id IS UNIQUE;

CREATE CONSTRAINT law_version_id_unique IF NOT EXISTS
FOR (n:LawVersion) REQUIRE n.version_id IS UNIQUE;

CREATE CONSTRAINT provision_id_unique IF NOT EXISTS
FOR (p:Provision) REQUIRE p.provision_id IS UNIQUE;

CREATE CONSTRAINT case_id_unique IF NOT EXISTS
FOR (c:Case) REQUIRE c.case_id IS UNIQUE;

CREATE CONSTRAINT legal_concept_name_unique IF NOT EXISTS
FOR (n:LegalConcept) REQUIRE n.name IS UNIQUE;

CREATE CONSTRAINT cause_name_unique IF NOT EXISTS
FOR (n:Cause) REQUIRE n.name IS UNIQUE;

CREATE CONSTRAINT court_name_unique IF NOT EXISTS
FOR (n:Court) REQUIRE n.name IS UNIQUE;

// ---------- 2. 普通索引（时间版本过滤走这条） ----------
CREATE INDEX provision_effective_range IF NOT EXISTS
FOR (p:Provision) ON (p.effective_from, p.effective_to);

CREATE INDEX provision_article_no IF NOT EXISTS
FOR (p:Provision) ON (p.law_name, p.article, p.paragraph);

// ---------- 3. 向量索引（四路消融，各模型维度不同，必须分开建） ----------
// 主索引：Qwen3-Embedding-0.6B (1024d) 与 BAAI/bge-m3 (1024d) 共用同一维度，
// 两者用各自的字段区分，切换向量模型时对该字段全量重建（换模型 = 全量重建索引）
CREATE VECTOR INDEX provision_embedding IF NOT EXISTS
FOR (p:Provision) ON (p.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE VECTOR INDEX provision_embedding_bge_m3 IF NOT EXISTS
FOR (p:Provision) ON (p.embedding_bge_m3)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

// 退化对照：hfl/chinese-bert-wwm-ext (768d)
// 注意这是预训练语言模型而非检索嵌入模型，用作「非检索专用模型」的基线
CREATE VECTOR INDEX provision_embedding_bert768 IF NOT EXISTS
FOR (p:Provision) ON (p.embedding_bert768)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};

// 消融上限：Qwen3-Embedding-4B (2560d)
CREATE VECTOR INDEX provision_embedding_4b IF NOT EXISTS
FOR (p:Provision) ON (p.embedding_qwen3_4b)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 2560,
    `vector.similarity_function`: 'cosine'
  }
};

// ---------- 4. 全文索引（关键词 / BM25 路召回） ----------
CREATE FULLTEXT INDEX provision_fulltext IF NOT EXISTS
FOR (p:Provision) ON EACH [p.law_name, p.text];

CREATE FULLTEXT INDEX case_fulltext IF NOT EXISTS
FOR (c:Case) ON EACH [c.facts, c.question];

// ---------- 5. 校验 ----------
SHOW VECTOR INDEXES;
SHOW FULLTEXT INDEXES;
SHOW CONSTRAINTS;

// ============================================================
// Provision 节点最小字段（导入时保证齐全）
//   provision_id     唯一 ID，建议 f"{law_id}-v{version_date}-a{article}-p{paragraph}-s{sub}"
//   law_id / law_name
//   version_id / version_date
//   article / paragraph / subparagraph
//   text
//   embedding         bge-m3 1024d
//   embedding_bert768 消融用
//   effective_from / effective_to   ★ 时间版本过滤依赖这两个字段
//   source_url / text_hash
//
// 关系模型
//   (Law)-[:HAS_VERSION]->(LawVersion)
//   (LawVersion)-[:HAS_PROVISION]->(Provision)
//   (Provision)-[:NEXT]->(Provision)
//   (LawVersion)-[:AMENDS]->(LawVersion)
//   (Case)-[:CITES]->(Provision)
//   (Case)-[:INVOLVES]->(LegalConcept)
//   (Case)-[:HAS_CAUSE]->(Cause)
//   (Case)-[:DECIDED_BY]->(Court)
// ============================================================
