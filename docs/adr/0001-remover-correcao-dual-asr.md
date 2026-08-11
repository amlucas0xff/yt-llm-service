# 0001 — Remover a correção dual-ASR ("GEC") do pipeline

**Status:** Aceito — 2026-08-11

## Contexto

O pipeline de YouTube tinha um passo opcional (ligado por padrão, `use_yt_captions=True`)
em que o gpt-oss-20b comparava a transcrição do WhisperX com as legendas automáticas do
YouTube (`en-orig`) e produzia uma "transcrição corrigida", que substituía a primária
como entrada da geração de notas.

A ideia é boa: duas ASRs independentes erram em lugares diferentes, e a discordância
marca onde provavelmente há erro ("entropy" → "Anthropic" é um ganho real e observado).

Mas a implementação corrompia a fonte silenciosamente, por dois defeitos estruturais:

1. **Alinhamento proporcional, não temporal.** Transcrições longas eram fatiadas em
   blocos de 2000 palavras, e a "referência" de cada bloco era recortada das legendas
   por fração de contagem de palavras. Como as duas fontes têm densidades diferentes
   (filler words, cadência), o desalinhamento acumula e blocos tardios recebem como
   referência trechos de outra parte do vídeo.
2. **Truncamento aceito como correção.** `max_tokens=4000` por bloco, com os tokens de
   raciocínio contando no limite; resposta cortada no meio era aceita desde que
   não-vazia. O fallback ao texto original só disparava em exceção.

Ambos degradam **sem sinal** — e o resultado degradado alimentava notas e Obsidian.
Nenhum teste verificava preservação de conteúdo (só montagem de prompt).

## Decisão

Remover o passo do pipeline: `correct_transcript`/`_correct_chunk`, o campo
`corrected_transcript` e o consumo de legendas como referência de correção.
A transcrição primária do WhisperX volta a ser a única fonte das notas.

## Alternativas consideradas

- **Desligar por padrão e manter o código** — rejeitado: código morto não exercitado
  apodrece, e os dois defeitos ficariam intactos atrás de uma flag.
- **Consertar e manter** — é o caminho de retomada legítimo, mas é um projeto próprio,
  não um ajuste. Condições para retomar:
  1. alinhamento **temporal** entre as fontes (ambas têm timestamps; usar);
  2. detecção de truncamento (comparar contagem de palavras entrada/saída; rejeitar
     respostas incompletas);
  3. saída em forma de **diff auditável** (o que mudou, onde), não texto reescrito
     opaco;
  4. teste de preservação de conteúdo, não só de prompt.

## Consequências

- Notas podem voltar a conter erros fonéticos de nomes próprios que a correção pegava.
  Mitigação parcial: o Video Context (título/canal/tags) continua disponível ao prompt
  de notas como autoridade ortográfica.
- O pipeline perde sua única etapa que *reescrevia* a fonte; todo o resto é derivação.
  A auditoria por amostragem (nota vs. transcrição) volta a ser suficiente.
