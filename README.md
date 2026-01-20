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

## Estrutura do Repositório

```
.
├── README.md
├── requirements.txt / environment.yml
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── reports/
│   └── generation/
├── scripts/
│   ├── run_training.py
│   ├── run_inference.py
│   ├── run_story_generation.py
│   └── run_end_to_end.py
└── docs/
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

## Execução do Pipeline

```bash
python scripts/run_end_to_end.py
```

---

## Saídas Geradas

```
outputs/
└── stories/
    └── person_id/
        ├── texts/
        └── images/
```

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
