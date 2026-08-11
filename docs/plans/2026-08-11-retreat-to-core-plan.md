# Retreat to Core — Plano de Execução

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Contexto:** Decisões tomadas na sessão de grilling de 2026-08-11, registradas em
`CONTEXT.md`, `docs/adr/0001-remover-correcao-dual-asr.md` e
`docs/adr/0002-abandonar-ocr-de-video.md`. O produto recua ao núcleo bom e funcional:
**transcrição → notas estruturadas → Obsidian**. Sai a correção dual-ASR ("GEC"),
sai o OCR de vídeo, o CLI volta ao endpoint síncrono, e entra verificação mínima
(comando de teste canônico + smoke test real).

**Como rodar os testes durante a execução** (até a Task 6 codificar isto):
`PYTHONPATH=tests/stubs:.:src uv run pytest tests/ -v` a partir da raiz, com
`TEMP_DIR`/`OUTPUT_DIR` graváveis e um shim de `yt-dlp` no `PATH` (ver `CLAUDE.md`).

---

## Task 1: Remover a correção dual-ASR (ADR 0001)

**Files:** `src/notes_service.py`, `src/run_llm_api.py`, `src/audio_downloader.py`,
`tests/test_notes_service.py`, `tests/test_audio_downloader.py`, `README.md`

- Remover de `notes_service.py`: `CORRECTION_SYSTEM_PROMPT`, `correct_transcript()`,
  `_correct_chunk()`. Manter `generate()` e `_build_context_block()` — o Video
  Context continua alimentando o prompt de notas como autoridade ortográfica.
- Remover de `run_llm_api.py`: campo `use_yt_captions` do request, campo
  `corrected_transcript` da resposta, e o bloco GEC do handler
  (`corrected_transcript = ...` / `transcript_for_notes = corrected_transcript or ...`
  vira só `llm_result.get("text")`).
- Remover de `audio_downloader.py`: download e parse de legendas (`_parse_vtt` e o
  fetch de `en-orig`) — as legendas só existiam como referência do GEC. Manter
  título/canal/tags/capítulos no `get_video_context()`.
- Atualizar os 2 testes de `test_notes_service.py` (hoje testam o prompt de correção)
  e o teste de `test_audio_downloader.py` que cobre captions.
- Remover a seção "GEC Transcript Correction" do README.
- **Verify:** suíte verde; `grep -ri "gec\|caption\|corrected" src/` sem sobras.
- **Commit:** `refactor: remove dual-ASR (GEC) correction per ADR 0001`

## Task 2: Remover o OCR do runtime (ADR 0002)

**Files:** `src/run_llm_api.py`, `cli.py`, `src/ocr_client.py`, `src/ocr_dedup.py`,
`src/frame_extractor.py`, `src/config.py`, `docker-compose.yml`, `ocr-service/`,
`ocr-poc/`, `poc/`, testes correspondentes, `README.md`

- Remover endpoints `/ocr-youtube` e `/ocr-file` e o singleton `OCRClient` de
  `run_llm_api.py`.
- Remover comando `ocr` e `render_ocr_output` de `cli.py`.
- Apagar `src/ocr_client.py`, `src/ocr_dedup.py`, `src/frame_extractor.py`,
  `tests/test_ocr_client.py`, `tests/test_ocr_dedup.py`, `tests/test_frame_extractor.py`.
- Remover `OCR_SERVICE_URL`, `OCR_SERVICE_TIMEOUT`, `OCR_SCENE_THRESHOLD`,
  `OCR_MAX_FRAMES` de `config.py` e `.env.example`.
- Remover o serviço `ocr-service` do `docker-compose.yml` (reconciliar com o diff
  já presente no working tree) e apagar os diretórios `ocr-service/`, `ocr-poc/` e
  `poc/` (código nunca commitado; a pesquisa que importa já está em `docs/research/`).
- Atualizar README: tabela de arquitetura vira 2 serviços, remover seções de OCR.
- **Verify:** suíte verde; `docker compose config` válido; `grep -ri "ocr" src/ cli.py`
  sem sobras.
- **Commit:** `refactor: remove video OCR pipeline per ADR 0002`

## Task 3: Reverter o CLI ao endpoint síncrono

**Files:** `cli.py`, `tests/test_cli.py`

- Remover `_stream_youtube()`; o caminho YouTube passa a chamar
  `POST /transcribe-youtube-llm` (mesmo padrão do caminho de upload de arquivo).
- Em `tests/test_cli.py`, remover os 2 testes de `_stream_youtube` (modificação hoje
  não-commitada) e adicionar teste do caminho síncrono com httpx mockado.
- **Verify:** suíte verde; `grep -n "stream" cli.py` sem sobras.
- **Commit:** `fix: CLI YouTube path calls the sync endpoint that actually exists`

## Task 4: Proveniência na nota Obsidian

**Files:** `src/notes_service.py`, `src/obsidian_service.py`, `src/run_llm_api.py`,
`tests/test_obsidian_service.py`, `tests/test_notes_service.py`

- `notes_service.generate()` passa a retornar também `truncated: bool` (verdadeiro
  quando `TRUNCATION_NOTICE` foi aplicado ao transcript antes do prompt).
- `obsidian_service.save_note()` ganha dois campos de frontmatter:
  `source_transcript:` (caminho do markdown salvo pela transcrição) e
  `truncated: true/false`.
- `run_llm_api.py` fia o caminho do arquivo salvo e a flag até o export.
- Testes: frontmatter contém os dois campos; flag reflete truncamento real.
- **Verify:** suíte verde.
- **Commit:** `feat: Obsidian note carries source-transcript path and truncation flag`

## Task 5: Limpar dependências mortas

**Files:** `requirements.txt`

- Remover `openai`, `supabase`, `fastmcp` (nenhum import em `src/`). Confirmar
  `requests` com `grep -rn "import requests" src/ cli.py` antes de remover.
  Reconciliar com o diff não-commitado do working tree.
- **Verify:** `docker compose build yt-llm-service` conclui; suíte verde.
- **Commit:** `chore: drop unused deps (openai, supabase, fastmcp)`

## Task 6: Codificar o comando de teste

**Files:** `Makefile` (novo), `tests/stubs/` (commitar), `conftest.py`, `README.md`

- `make test` executa a receita canônica: `PYTHONPATH` com `tests/stubs`, raiz e
  `src`; `TEMP_DIR`/`OUTPUT_DIR` apontando para diretório temporário; shim de
  `yt-dlp` no `PATH`. Preferir mover a lógica de path para `conftest.py` (inserir
  `tests/stubs` em `sys.path`) para o comando encolher.
- Commitar `tests/stubs/torch.py` com comentário explicando o porquê.
- README: seção "Running tests" com o comando único.
- **Verify:** `make test` verde a partir de um clone limpo de variáveis de ambiente.
- **Commit:** `chore: add make test as the canonical test entrypoint`

## Task 7: Smoke test real ponta a ponta

**Files:** `scripts/smoke.sh` (novo), `README.md`

- Script que: sobe `docker compose up -d --wait`, faz `POST /transcribe-youtube-llm`
  com um vídeo curto (~1 min, URL parametrizável, default estável), espera a resposta,
  e verifica: HTTP 200, transcript não-vazio, arquivo de nota criado em `OUTPUT_DIR`
  e não-vazio. Sai com código ≠ 0 em qualquer falha.
- Este é o teste que exercita WhisperX + llama-cpp de verdade — rodar sob demanda,
  não a cada commit.
- README: seção "Smoke test".
- **Verify:** rodar o script uma vez com os serviços de pé; anexar a duração observada
  ao README.
- **Commit:** `feat: add end-to-end smoke test script`

## Task 8: Branches órfãos

- `feature/ocr-migration` (Tesseract, superado — registrado no ADR 0002) e
  `feature/performance-boost` (6 commits antigos, cultura de docs anterior):
  **perguntar ao usuário** antes de apagar; se autorizado, `git branch -D` + prune
  dos worktrees obsoletos.
- **Commit:** nenhum (operação de branch).

---

**Ordem:** 1 → 2 → 3 são independentes entre si mas mexem em `run_llm_api.py`/`cli.py`
— executar em sequência para evitar conflito. 4–7 dependem de 1–3 concluídas.
8 a qualquer momento, com autorização.
