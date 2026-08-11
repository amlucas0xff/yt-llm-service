# CONTEXT

Glossário do domínio. Só vocabulário — sem detalhes de implementação.

## Termos

- **Transcrição (primária)** — o texto produzido pelo ASR local (WhisperX) a partir do áudio do vídeo. É a fonte da verdade do pipeline; tudo o mais deriva dela.
- **Legendas de referência** — as legendas automáticas geradas pelo próprio YouTube (`en-orig`). Uma segunda ASR, independente, usada apenas como sinal de comparação — nunca como fonte primária.
- **Correção dual-ASR** (nome histórico: "GEC") — ideia de usar um LLM para comparar a transcrição primária com as legendas de referência e corrigir palavras mal-ouvidas. O nome "GEC" (Grammatical Error Correction) é um empréstimo impreciso de NLP: o que se corrige são erros de *reconhecimento de fala*, não de gramática. **Removida do produto** (ver ADR 0001); condições de retomada registradas lá.
- **Video Context** — metadados do vídeo (título, canal, tags, capítulos) obtidos junto do download; serve como autoridade ortográfica para nomes próprios nos prompts.
- **Notas estruturadas** — documento markdown (título, overview, seções, takeaways) gerado por LLM a partir da transcrição. É *derivada e auditável*: sempre pode ser conferida contra a transcrição.
- **Export Obsidian** — gravação da nota no vault do usuário com frontmatter.
- **OCR de vídeo** — extração de texto de quadros do vídeo (scene detection + OCR). *Experimento falho*: em screencasts, sinal-ruído medido de ~1:150; o texto capturado é chrome de UI, não conteúdo. **Removido do produto** (ver ADR 0002); critério de retomada registrado lá.

## Distinções que importam

- **Primária vs. corrigida**: só a correção dual-ASR *reescreve* a fonte; notas e export apenas *derivam* dela. Risco de corrupção silenciosa vive exclusivamente na correção.
- **Medido vs. projetado**: números vindos de execução real sobre vídeos do usuário vs. estimativas extraídas de literatura. Os planos de OCR falharam por tratar projeções como medições.
