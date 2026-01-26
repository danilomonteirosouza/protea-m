# PROTEA-M — Código-fonte da Tese  
Framework Preditivo-Generativo Multimodal para Comunicação em Situações de Crise

Este repositório contém o **código-fonte oficial da tese de doutorado** que propõe o **PROTEA-M**, um *framework* **preditivo-generativo multimodal** voltado ao eixo de **disseminação e comunicação** em Sistemas de Alerta Precoce (*Early Warning Systems – EWS*), com foco em **pessoas com Complexas Necessidades de Comunicação**, como indivíduos com **Transtorno do Espectro Autista (TEA)**.

O framework integra:
- **Modelagem preditiva**, para inferir o **perfil comunicativo individual** com base na Matriz de Comunicação;
- **Geração multimodal**, para produzir **Histórias Sociais personalizadas**, incluindo texto estruturado e quadrinhos ilustrados.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Framework](#arquitetura-do-framework)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Dados de Entrada](#dados-de-entrada)
- [Execução do Pipeline](#execução-do-pipeline)
- [Bloco Preditivo](#bloco-preditivo)
- [Bloco Generativo](#bloco-generativo)
- [Saídas Geradas](#saídas-geradas)
- [Reprodutibilidade](#reprodutibilidade)
- [Aspectos Éticos e Privacidade](#aspectos-éticos-e-privacidade)
- [Troubleshooting](#troubleshooting)
- [Como Citar](#como-citar)
- [Licença](#licença)

---

## Visão Geral

O PROTEA-M foi concebido para **preparar indivíduos com TEA para situações de emergência**, especialmente inundações, por meio de conteúdos comunicativos personalizados. Diferentemente de abordagens puramente generativas, o framework introduz uma **camada preditiva anterior**, responsável por inferir o perfil comunicativo do usuário antes da geração do conteúdo.

---

## Arquitetura do Framework

O framework é dividido em dois macroblocos:

### 1. Bloco Preditivo
Responsável por inferir:
- Nível comunicativo
- Estratégias de comunicação adequadas
- Parâmetros de personalização para geração

### 2. Bloco Generativo
Responsável por:
- Gerar roteiros estruturados de Histórias Sociais
- Criar painéis ilustrados (quadrinhos)
- Organizar as saídas multimodais por indivíduo

---


## Estrutura do Repositório (esperada para execução sem falhas)

> **Importante:** os scripts atuais utilizam caminhos relativos por padrão. Para evitar erros de “pasta/arquivo não encontrado”, mantenha a estrutura abaixo **ou** passe caminhos explícitos via CLI.

```
.
├── README.md
├── requirements.txt
├── api_key.env                 # opcional (ou .env). Pode conter OPENAI_API_KEY e OPENAI_* (ver seção Bloco Generativo)
├── modulo_preditivo.py         # módulo preditivo (treino/avaliação + geração de embeddings SBERT)
├── modulo_generativo.py        # módulo generativo (História Social + HQ)
├── dados/
│   ├── matriz_caa_ajustada.csv       # ENTRADA (Modelo 1 / base original, sem embeddings)  ← default do preditivo
│   ├── base_embeddings_sbert.csv     # GERADO automaticamente (se não existir) pelo preditivo
│   └── resultado_v98_combo_blend/    # SAÍDA do preditivo (criado na mesma pasta do CSV SBERT)
│       ├── fold_1/
│       ├── fold_2/
│       ├── ...
│       └── relatorios_gerais/
├── examples/
│   └── matriz_caa_ajustada.csv       # opcional: ENTRADA default do generativo (pode ser cópia/symlink do arquivo em dados/)
└── workspace_social_story/      # SAÍDA do generativo (default)
    ├── pessoa_<PessoaID>/
    │   ├── textos/
    │   └── imagens/
    └── resultados/
        └── logs/
            └── social_story_offline_reason.txt   # aparece quando o modo offline/fallback é acionado
```
---

## Pré-requisitos

- Python 3.10 ou superior
- Conda ou virtualenv
- Dependências listadas em `requirements.txt` ou `environment.yml`

---

## Configuração do Ambiente

```bash
conda env create -f environment.yml
conda activate protea-m
```

ou

```bash
python -m venv .venv
pip install -r requirements.txt
```

---



## Dados de Entrada

### Arquivo do Modelo 1 (`matriz_caa_ajustada.csv`)

Tanto o bloco preditivo quanto o generativo partem de um CSV do **Modelo 1** com, no mínimo, as colunas:

- `PessoaID`
- `PlanoIntervencao`
- `PlanoGuia`
- `NivelComunicacao`
- `FuncaoComunicativa`
- `Ambiente`
- `ParceiroComunicacional`
- `SuporteNecessario`
- `FormaPreferida`
- `DescricaoComportamento`

No generativo, caso alguma coluna esteja ausente, ela é criada vazia para evitar falha na execução. fileciteturn2file0L73-L143

---
## Execução do Pipeline

### 1) Bloco Preditivo (avaliação + geração de embeddings)

Executa o pipeline com os padrões (lê `dados/matriz_caa_ajustada.csv`; cria `dados/base_embeddings_sbert.csv` se necessário; salva resultados em `dados/resultado_v98_combo_blend/`):

```bash
python modulo_preditivo.py
```

Exemplo com caminhos explícitos (recomendado se seu CSV estiver em outra pasta):

```bash
python modulo_preditivo.py \
  --csv_original dados/matriz_caa_ajustada.csv \
  --csv_sbert dados/base_embeddings_sbert.csv
```

O script procura os arquivos por candidatos comuns (ex.: raiz do projeto, `dados/`, e algumas variações relativas) antes de falhar. fileciteturn2file2L100-L118

### 2) Bloco Generativo (História Social + HQ)

Executa usando o CSV do Modelo 1 e grava em `workspace_social_story/`:

```bash
python modulo_generativo.py --matriz_csv dados/matriz_caa_ajustada.csv
```

Se você mantiver a cópia em `examples/` (padrão do script), o comando mínimo é:

```bash
python modulo_generativo.py
```

Para forçar modo offline (sem chamadas online):

```bash
python modulo_generativo.py --matriz_csv dados/matriz_caa_ajustada.csv --gen_mode offline
```


---


## Saídas Geradas

### Saídas do Bloco Preditivo

- `dados/base_embeddings_sbert.csv`: base pré-computada com embeddings SBERT + colunas `*_freq` (gerada automaticamente quando ausente). fileciteturn3file9L64-L83  
- `dados/resultado_v98_combo_blend/`: diretório de resultados com subpastas por *fold*, gráficos (`.png`) e relatórios (`.csv/.json`). (A pasta é criada no mesmo diretório onde o CSV SBERT é resolvido.)

### Saídas do Bloco Generativo

- `workspace_social_story/pessoa_<PessoaID>/textos/`: roteiro e textos estruturados da História Social. fileciteturn2file0L152-L161  
- `workspace_social_story/pessoa_<PessoaID>/imagens/`: imagens geradas (ou placeholders em modo offline). fileciteturn2file0L152-L161  
- `workspace_social_story/resultados/logs/`: logs auxiliares; quando o modo offline/fallback é acionado, o motivo pode ser registrado em `social_story_offline_reason.txt`. fileciteturn2file0L221-L229  


---

## Reprodutibilidade

- Seed fixa
- Configurações versionadas
- Logs de execução

---

## Aspectos Éticos e Privacidade

- Dados anonimizados
- Uso acadêmico
- Conformidade ética

---

## Como Citar

```
Souza, D. (2026). PROTEA-M: Framework preditivo-generativo multimodal para apoio à comunicação em situações de crise. Código-fonte da tese de doutorado.
```

---

## Licença

Uso acadêmico restrito.
