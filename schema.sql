--
-- PostgreSQL database dump
--

\restrict 5LAaSyh0sS9Cvv84OFXqAuvogo96CMX1DF5SVOrATVap4EkpLrrHzkoAem6YCIK

-- Dumped from database version 16.13 (Debian 16.13-1.pgdg12+1)
-- Dumped by pg_dump version 17.7 (Ubuntu 17.7-0ubuntu0.25.10.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: usuario
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO usuario;

--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: usuario
--

COMMENT ON SCHEMA public IS '';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: accession; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.accession (
    code character varying NOT NULL,
    "primary" boolean,
    tag character varying,
    protein_id character varying,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.accession OWNER TO usuario;

--
-- Name: chain; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.chain (
    id integer NOT NULL,
    name character varying NOT NULL,
    structure_id character varying NOT NULL,
    sequence_id integer,
    accession_code character varying
);


ALTER TABLE public.chain OWNER TO usuario;

--
-- Name: chain_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.chain_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.chain_id_seq OWNER TO usuario;

--
-- Name: chain_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.chain_id_seq OWNED BY public.chain.id;


--
-- Name: go_terms; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.go_terms (
    go_id character varying NOT NULL,
    category character varying NOT NULL,
    description character varying
);


ALTER TABLE public.go_terms OWNER TO usuario;

--
-- Name: protein; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.protein (
    id character varying NOT NULL,
    sequence_id integer,
    data_class character varying,
    molecule_type character varying,
    created_date date,
    sequence_update_date date,
    annotation_update_date date,
    description character varying,
    gene_name character varying,
    organism character varying,
    organelle character varying,
    taxonomy_id character varying,
    comments character varying,
    protein_existence integer,
    seqinfo character varying,
    disappeared boolean,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.protein OWNER TO usuario;

--
-- Name: protein_go_term_annotation; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.protein_go_term_annotation (
    id integer NOT NULL,
    protein_id character varying NOT NULL,
    go_id character varying NOT NULL,
    evidence_code character varying NOT NULL
);


ALTER TABLE public.protein_go_term_annotation OWNER TO usuario;

--
-- Name: protein_go_term_annotation_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.protein_go_term_annotation_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.protein_go_term_annotation_id_seq OWNER TO usuario;

--
-- Name: protein_go_term_annotation_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.protein_go_term_annotation_id_seq OWNED BY public.protein_go_term_annotation.id;


--
-- Name: sequence; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.sequence (
    id integer NOT NULL,
    sequence character varying NOT NULL,
    sequence_hash character varying
);


ALTER TABLE public.sequence OWNER TO usuario;

--
-- Name: sequence_embedding_type; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.sequence_embedding_type (
    id integer NOT NULL,
    name character varying NOT NULL,
    description character varying,
    task_name character varying,
    model_name character varying
);


ALTER TABLE public.sequence_embedding_type OWNER TO usuario;

--
-- Name: sequence_embedding_type_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.sequence_embedding_type_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sequence_embedding_type_id_seq OWNER TO usuario;

--
-- Name: sequence_embedding_type_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.sequence_embedding_type_id_seq OWNED BY public.sequence_embedding_type.id;


--
-- Name: sequence_embeddings; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.sequence_embeddings (
    id integer NOT NULL,
    sequence_id integer NOT NULL,
    embedding_type_id integer NOT NULL,
    layer_index integer NOT NULL,
    embedding public.halfvec NOT NULL,
    shape integer[],
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.sequence_embeddings OWNER TO usuario;

--
-- Name: sequence_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.sequence_embeddings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sequence_embeddings_id_seq OWNER TO usuario;

--
-- Name: sequence_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.sequence_embeddings_id_seq OWNED BY public.sequence_embeddings.id;


--
-- Name: sequence_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.sequence_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sequence_id_seq OWNER TO usuario;

--
-- Name: sequence_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.sequence_id_seq OWNED BY public.sequence.id;


--
-- Name: state; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.state (
    id integer NOT NULL,
    model_id character varying NOT NULL,
    file_path character varying NOT NULL,
    chain_id integer NOT NULL,
    structure_id character varying NOT NULL
);


ALTER TABLE public.state OWNER TO usuario;

--
-- Name: state_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.state_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.state_id_seq OWNER TO usuario;

--
-- Name: state_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.state_id_seq OWNED BY public.state.id;


--
-- Name: structural_alignment_types; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.structural_alignment_types (
    id integer NOT NULL,
    name character varying NOT NULL,
    description character varying,
    task_name character varying
);


ALTER TABLE public.structural_alignment_types OWNER TO usuario;

--
-- Name: structural_alignment_types_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.structural_alignment_types_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.structural_alignment_types_id_seq OWNER TO usuario;

--
-- Name: structural_alignment_types_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.structural_alignment_types_id_seq OWNED BY public.structural_alignment_types.id;


--
-- Name: structure; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.structure (
    id character varying NOT NULL,
    protein_id character varying NOT NULL,
    method character varying,
    resolution double precision,
    file_path character varying NOT NULL,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.structure OWNER TO usuario;

--
-- Name: structure_3di; Type: TABLE; Schema: public; Owner: usuario
--

CREATE TABLE public.structure_3di (
    id integer NOT NULL,
    state_id integer NOT NULL,
    embedding character varying,
    created_at timestamp without time zone,
    updated_at timestamp without time zone
);


ALTER TABLE public.structure_3di OWNER TO usuario;

--
-- Name: structure_3di_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario
--

CREATE SEQUENCE public.structure_3di_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.structure_3di_id_seq OWNER TO usuario;

--
-- Name: structure_3di_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario
--

ALTER SEQUENCE public.structure_3di_id_seq OWNED BY public.structure_3di.id;


--
-- Name: chain id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.chain ALTER COLUMN id SET DEFAULT nextval('public.chain_id_seq'::regclass);


--
-- Name: protein_go_term_annotation id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.protein_go_term_annotation ALTER COLUMN id SET DEFAULT nextval('public.protein_go_term_annotation_id_seq'::regclass);


--
-- Name: sequence id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence ALTER COLUMN id SET DEFAULT nextval('public.sequence_id_seq'::regclass);


--
-- Name: sequence_embedding_type id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embedding_type ALTER COLUMN id SET DEFAULT nextval('public.sequence_embedding_type_id_seq'::regclass);


--
-- Name: sequence_embeddings id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embeddings ALTER COLUMN id SET DEFAULT nextval('public.sequence_embeddings_id_seq'::regclass);


--
-- Name: state id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.state ALTER COLUMN id SET DEFAULT nextval('public.state_id_seq'::regclass);


--
-- Name: structural_alignment_types id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structural_alignment_types ALTER COLUMN id SET DEFAULT nextval('public.structural_alignment_types_id_seq'::regclass);


--
-- Name: structure_3di id; Type: DEFAULT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structure_3di ALTER COLUMN id SET DEFAULT nextval('public.structure_3di_id_seq'::regclass);


--
-- Name: accession accession_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.accession
    ADD CONSTRAINT accession_pkey PRIMARY KEY (code);


--
-- Name: chain chain_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.chain
    ADD CONSTRAINT chain_pkey PRIMARY KEY (id);


--
-- Name: go_terms go_terms_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.go_terms
    ADD CONSTRAINT go_terms_pkey PRIMARY KEY (go_id);


--
-- Name: protein_go_term_annotation protein_go_term_annotation_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.protein_go_term_annotation
    ADD CONSTRAINT protein_go_term_annotation_pkey PRIMARY KEY (id, protein_id, go_id);


--
-- Name: protein protein_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.protein
    ADD CONSTRAINT protein_pkey PRIMARY KEY (id);


--
-- Name: sequence_embedding_type sequence_embedding_type_name_key; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embedding_type
    ADD CONSTRAINT sequence_embedding_type_name_key UNIQUE (name);


--
-- Name: sequence_embedding_type sequence_embedding_type_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embedding_type
    ADD CONSTRAINT sequence_embedding_type_pkey PRIMARY KEY (id);


--
-- Name: sequence_embeddings sequence_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embeddings
    ADD CONSTRAINT sequence_embeddings_pkey PRIMARY KEY (id);


--
-- Name: sequence sequence_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence
    ADD CONSTRAINT sequence_pkey PRIMARY KEY (id);


--
-- Name: state state_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.state
    ADD CONSTRAINT state_pkey PRIMARY KEY (id);


--
-- Name: structural_alignment_types structural_alignment_types_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structural_alignment_types
    ADD CONSTRAINT structural_alignment_types_pkey PRIMARY KEY (id);


--
-- Name: structure_3di structure_3di_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structure_3di
    ADD CONSTRAINT structure_3di_pkey PRIMARY KEY (id);


--
-- Name: structure structure_file_path_key; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structure
    ADD CONSTRAINT structure_file_path_key UNIQUE (file_path);


--
-- Name: structure structure_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structure
    ADD CONSTRAINT structure_pkey PRIMARY KEY (id);


--
-- Name: protein_go_term_annotation uq_pga_protein_go; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.protein_go_term_annotation
    ADD CONSTRAINT uq_pga_protein_go UNIQUE (protein_id, go_id);


--
-- Name: sequence_embeddings uq_seqemb_seq_type_layer; Type: CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embeddings
    ADD CONSTRAINT uq_seqemb_seq_type_layer UNIQUE (sequence_id, embedding_type_id, layer_index);


--
-- Name: idx_sequence_hash; Type: INDEX; Schema: public; Owner: usuario
--

CREATE INDEX idx_sequence_hash ON public.sequence USING btree (sequence_hash);


--
-- Name: ix_seqemb_seq; Type: INDEX; Schema: public; Owner: usuario
--

CREATE INDEX ix_seqemb_seq ON public.sequence_embeddings USING btree (sequence_id);


--
-- Name: ix_seqemb_type; Type: INDEX; Schema: public; Owner: usuario
--

CREATE INDEX ix_seqemb_type ON public.sequence_embeddings USING btree (embedding_type_id);


--
-- Name: ix_sequence_sequence_hash; Type: INDEX; Schema: public; Owner: usuario
--

CREATE UNIQUE INDEX ix_sequence_sequence_hash ON public.sequence USING btree (sequence_hash);


--
-- Name: accession accession_protein_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.accession
    ADD CONSTRAINT accession_protein_id_fkey FOREIGN KEY (protein_id) REFERENCES public.protein(id);


--
-- Name: chain chain_accession_code_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.chain
    ADD CONSTRAINT chain_accession_code_fkey FOREIGN KEY (accession_code) REFERENCES public.accession(code);


--
-- Name: chain chain_sequence_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.chain
    ADD CONSTRAINT chain_sequence_id_fkey FOREIGN KEY (sequence_id) REFERENCES public.sequence(id);


--
-- Name: chain chain_structure_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.chain
    ADD CONSTRAINT chain_structure_id_fkey FOREIGN KEY (structure_id) REFERENCES public.structure(id);


--
-- Name: protein_go_term_annotation protein_go_term_annotation_go_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.protein_go_term_annotation
    ADD CONSTRAINT protein_go_term_annotation_go_id_fkey FOREIGN KEY (go_id) REFERENCES public.go_terms(go_id);


--
-- Name: protein_go_term_annotation protein_go_term_annotation_protein_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.protein_go_term_annotation
    ADD CONSTRAINT protein_go_term_annotation_protein_id_fkey FOREIGN KEY (protein_id) REFERENCES public.protein(id);


--
-- Name: protein protein_sequence_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.protein
    ADD CONSTRAINT protein_sequence_id_fkey FOREIGN KEY (sequence_id) REFERENCES public.sequence(id);


--
-- Name: sequence_embeddings sequence_embeddings_embedding_type_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embeddings
    ADD CONSTRAINT sequence_embeddings_embedding_type_id_fkey FOREIGN KEY (embedding_type_id) REFERENCES public.sequence_embedding_type(id);


--
-- Name: sequence_embeddings sequence_embeddings_sequence_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.sequence_embeddings
    ADD CONSTRAINT sequence_embeddings_sequence_id_fkey FOREIGN KEY (sequence_id) REFERENCES public.sequence(id);


--
-- Name: state state_chain_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.state
    ADD CONSTRAINT state_chain_id_fkey FOREIGN KEY (chain_id) REFERENCES public.chain(id);


--
-- Name: structure_3di structure_3di_state_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structure_3di
    ADD CONSTRAINT structure_3di_state_id_fkey FOREIGN KEY (state_id) REFERENCES public.state(id);


--
-- Name: structure structure_protein_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario
--

ALTER TABLE ONLY public.structure
    ADD CONSTRAINT structure_protein_id_fkey FOREIGN KEY (protein_id) REFERENCES public.protein(id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: usuario
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;


--
-- PostgreSQL database dump complete
--

\unrestrict 5LAaSyh0sS9Cvv84OFXqAuvogo96CMX1DF5SVOrATVap4EkpLrrHzkoAem6YCIK

