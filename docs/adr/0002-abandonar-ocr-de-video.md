# 0002 — Remover o OCR de vídeo do runtime

**Status:** Aceito — 2026-08-11

## Contexto

O objetivo era capturar texto exibido nos quadros do vídeo (código, slides) para
enriquecer a extração de conhecimento. Quatro abordagens foram tentadas em sequência,
nenhuma com registro de abandono:

1. **Tesseract** (branch `feature/ocr-migration`, 2026-03-04) — pacote completo com
   33 testes; abandonado no dia seguinte, sem merge nem registro.
2. **PaddleOCR via sidecar** (master, 2026-03-05) — a única que chegou ao runtime:
   scene detection (ffmpeg) → PaddleOCR full-frame → dedup Jaccard entre quadros.
3. **CLIP region cropping** (`poc/`, não commitado) — recortar a região de conteúdo
   antes do OCR.
4. **ROI temporal** (plano `2026-03-05-video-ocr-temporal-roi-poc-plan.md`) — planejado,
   nunca implementado.

## O que foi medido (vs. projetado)

- **Sinal-ruído ~1:150** no pipeline shipped, sobre screencast real (`result-crap.txt`,
  1092 linhas): o OCR lê com 79–99% de confiança — mas lê chrome de UI (`node_modules`,
  números de gutter, `Problems Output Debug Console`, overlay de screen-share).
  A falha é de **seleção semântica**, não de reconhecimento; threshold de confiança
  não resolve.
- **A deduplicação nunca dispara**: quadros vêm de *scene detection* (selecionados por
  serem dissimilares) e a fusão exige Jaccard ≥ 0.6 entre consecutivos — as premissas
  se cancelam. Na medição real: 20 quadros → 20 spans, zero fusões. (Os 15 testes
  unitários do módulo passam; testam uma condição que a produção não produz.)
- **CLIP**: um único screenshot medido; 73% de redução em threshold 0.53, colapso para
  full-frame em 0.54. Janela útil de ~0.01 — frágil demais.
- Todos os demais números dos planos (75–98% de redução) eram **projeções de
  literatura**, explicitamente marcadas como estimativas, nunca reproduzidas localmente.
- O benchmark de engines (`ocr-poc/benchmark.py`) foi construído mas **nunca executado**.

## Decisão

Remover do runtime: endpoints `/ocr-youtube` e `/ocr-file`, comando `ocr` do CLI,
serviço `ocr-service` do docker-compose, módulos `ocr_client`/`ocr_dedup`/
`frame_extractor` e a configuração associada. O histórico git preserva o código;
`docs/research/` fica commitado como registro do resultado negativo.

## Critério de retomada

Só reintegrar OCR ao produto depois que um POC demonstrar, **medido em vídeos reais do
usuário** (não projetado de literatura), saída majoritariamente útil — por exemplo:
≤30 linhas por vídeo com >80% de conteúdo genuíno. Integração vem depois da medição,
nunca antes. As direções promissoras documentadas em `docs/research/` (filtro temporal
+ ROI espacial + pós-filtro por LLM) continuam válidas como hipóteses.

## Consequências

- A API encolhe: quem quiser texto de tela precisa esperar uma versão que funcione.
- GPU 1 deixa de hospedar o sidecar PaddleOCR; só o llama-cpp permanece.
- Some a única razão de o compose ter três serviços; sobra transcrição + notas.
