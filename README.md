# PROTEA-M  
Framework Preditivo-Generativo Multimodal para Comunicação em Situações de Crise

Este repositório contém o **código-fonte oficial** que propõe o **PROTEA-M**, um *framework* **preditivo-generativo multimodal** voltado ao eixo de **disseminação e comunicação** em Sistemas de Alerta Precoce (*Early Warning Systems – EWS*), com foco em **pessoas com Complexas Necessidades de Comunicação**, como indivíduos com **Transtorno do Espectro Autista (TEA)**.

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
├── api_key.env
├── modulo_preditivo.py
├── modulo_generativo.py
├── dados/
│   ├── matriz_caa_ajustada.csv
│   ├── base_embeddings_sbert.csv
│   └── resultado_v98_combo_blend/
├── examples/
│   └── matriz_caa_ajustada.csv
└── workspace_social_story/
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

A matriz CAA utilizada como entrada do pipeline corresponde à base **MARSOCH (Modelo 1)**, disponibilizada publicamente no seguinte repositório:

https://github.com/danilomonteirosouza/marsoch

A matriz de embeddings SBERT (`base_embeddings_sbert.csv`) **não é fornecida previamente** e é **gerada automaticamente após a execução do bloco preditivo**.

---

## Execução do Pipeline

```bash
python modulo_preditivo.py
python modulo_generativo.py
```

---

## Licença

Uso acadêmico restrito.
