# -*- coding: utf-8 -*-
"""
PROTEA-M — Geração de História Social em Quadrinhos
Versão: 0.1.0

Objetivo
--------
Fluxo fim a fim para:
1) Ler a planilha do Modelo 1 (M1) contendo perfil da pessoa com TEA;
2) Para cada pessoa, gerar APENAS uma História Social em quadrinhos
   (texto estruturado + descrição dos painéis);
3) Gerar UMA imagem de página em quadrinhos (com todos os painéis) usando OpenAI (ou fallback offline);
4) Salvar os resultados em subdiretórios por pessoa (textos/ e imagens/).

Destaques
---------
- Mantém configuração de comunicação com a OpenAI (API key, org, project, base_url);
- Usa apenas:
    - generate_image (para as imagens dos painéis);
    - complete_json (para gerar o roteiro da história social);
- NÃO gera PTI, PDF, métricas, áudio, vídeo ou outros materiais.
- Tipo de desastre parametrizado (default: "inundacao").
"""

from __future__ import annotations

__version__ = "0.1.0-social-story-only"

import os
import sys
import json
import argparse
import textwrap
import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# ==============================
# Imports opcionais
# ==============================
try:
    import pandas as pd
except Exception as e:
    print("ERRO: pandas não instalado. Instale com `pip install pandas`", file=sys.stderr)
    raise

try:
    import requests
    _REQ_OK = True
except Exception:
    _REQ_OK = False

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # segue sem dotenv

# openai (SDK moderno)
_OPENAI_OK = False
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


# ==============================
# Utils gerais (IO, strings, etc.)
# ==============================
_PTI_LAST_WORKSPACE: Optional[Path] = None

REQ_COLS = [
    "PessoaID", "PlanoIntervencao", "PlanoGuia", "NivelComunicacao", "FuncaoComunicativa",
    "Ambiente", "ParceiroComunicacional", "SuporteNecessario", "FormaPreferida",
    "DescricaoComportamento"
]


def _strip_bom(name: str) -> str:
    return str(name).replace("\ufeff", "").replace("\u200b", "").strip().rstrip(",;")


def read_csv_auto(path: str | Path) -> "pd.DataFrame":
    path = str(path)
    encodings = ["utf-8-sig", "utf-8", "latin1"]

    def _clean(df):
        df.columns = [_strip_bom(c) for c in df.columns]
        return df

    def _looks_misparsed(df: "pd.DataFrame") -> bool:
        # Caso clássico: header veio "colado" com ';' e aparece Unnamed
        if any(";" in c for c in df.columns):
            return True
        if df.shape[1] == 1:
            # 1 coluna só quase sempre é separador errado
            return True
        return False

    # 1) tentativa com auto-detecção de separador (MAS validando)
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc, low_memory=False)
            df = _clean(df)
            if not _looks_misparsed(df):
                return df
        except Exception:
            pass

    # 2) tenta ; explicitamente
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=";", engine="c", encoding=enc, low_memory=False)
            df = _clean(df)
            if not _looks_misparsed(df):
                return df
        except Exception:
            pass

    # 3) tenta , explicitamente
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=",", engine="c", encoding=enc, low_memory=False)
            df = _clean(df)
            if not _looks_misparsed(df):
                return df
        except Exception:
            pass

    # 4) fallback final (bem conservador)
    df = pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig", low_memory=False)
    df = _clean(df)
    return df

def ensure_required_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    """Garante colunas obrigatórias; cria vazias se ausentes."""
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = ""
    if "PessoaID" in df.columns:
        df["PessoaID"] = df["PessoaID"].astype(str)
    return df


def slugify(text: str, maxlen: int = 40) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    out = "_".join([t for t in out.split("_") if t])  # remove múltiplos _
    return out[:maxlen] if out else "item"


def ensure_dirs_person(workspace: Path, pessoa_id: str) -> Dict[str, Path]:
    root = workspace / f"pessoa_{slugify(pessoa_id, 80)}"
    dirs = {
        "root": root,
        "textos": root / "textos",
        "imagens": root / "imagens",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_write_text(path: Path, content: str, encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def safe_write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _mask_key(k: str) -> str:
    k = (k or "").strip()
    if not k:
        return "(vazia)"
    if len(k) <= 10:
        return k[0] + "***" + k[-1]
    return f"{k[:6]}...{k[-4:]}"


def load_env_file(explicit_path: Optional[str] = None):
    """Carrega variáveis de ambiente de um arquivo .env/api_key.env, se existir."""
    if load_dotenv is None:
        print("[PROTEA-M] python-dotenv não instalado — seguindo sem carregar arquivo .env")
        return

    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_var_path = os.getenv("PTI_ENV_FILE", "").strip()
    if env_var_path:
        candidates.append(Path(env_var_path))

    here = Path(__file__).resolve().parent
    candidates += [
        here / ".env",
        here / "api_key.env",
        Path.cwd() / ".env",
        Path.cwd() / "api_key.env",
    ]

    loaded_from = None
    for p in candidates:
        try:
            if p and p.exists():
                load_dotenv(dotenv_path=str(p), override=False)
                loaded_from = str(p)
                break
        except Exception:
            pass

    if loaded_from:
        print(f"[PROTEA-M] Variáveis de ambiente carregadas de: {loaded_from}")
    else:
        print("[PROTEA-M] Nenhum arquivo .env/api_key.env encontrado — usando ambiente do sistema (se houver).")


def _write_offline_reason(reason: str):
    """Escreve motivo de modo offline em resultados/logs, se possível."""
    try:
        base = _PTI_LAST_WORKSPACE if _PTI_LAST_WORKSPACE else Path("workspace_out")
        logs_dir = Path(base) / "resultados" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "social_story_offline_reason.txt").write_text(reason, encoding="utf-8")
    except Exception:
        pass


def sanity_check_openai(client, model_name: str) -> Optional[str]:
    """
    Faz uma chamada mínima para validar credenciais/projeto/modelo.
    Retorna None se OK; caso contrário, retorna string detalhando o motivo.
    """
    try:
        _ = client.models.list()
        _ = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=16,
        )
        return None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        cls = e.__class__.__name__
        msg = getattr(e, "message", None) or str(e)
        status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
        text = None
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                text = resp.text
            except Exception:
                text = None
        parts = [f"Classe do erro: {cls}", f"Mensagem: {msg}"]
        if status:
            parts.append(f"HTTP status: {status}")
        if text:
            parts.append(f"Resposta da API: {text[:800]}")
        parts.append("Stacktrace:\n" + tb)
        return "\n".join(parts)


# ==============================
# IA Generativa — Base + Providers
# ==============================
class GenerativeProviderBase:
    def generate_image(self, prompt: str, out_path: Path) -> Dict[str, Any]:
        raise NotImplementedError

    def complete_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class OfflineProvider(GenerativeProviderBase):
    """Placeholders locais (sem IA online)."""

    def __init__(self, seed: int = 123):
        self.rng = random.Random(seed)

    def generate_image(self, prompt: str, out_path: Path) -> Dict[str, Any]:
        # Apenas escreve um arquivo de texto com o prompt
        out_txt = out_path.with_suffix(".txt")
        content = f"[IMG PLACEHOLDER OFFLINE]\nPROMPT ORIGINAL:\n{prompt}\n"
        safe_write_text(out_txt, content)
        return {"provider": "offline", "path": str(out_txt)}

    def complete_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        # Sem IA online: retorna None para acionar fallback heurístico
        return None


class OpenAIProvider(GenerativeProviderBase):
    """
    Usa OpenAI para:
    - complete_json (história social);
    - generate_image (imagens dos painéis).
    Se der erro crítico, desativa e cai para OfflineProvider.
    """

    def __init__(
        self,
        model_text: str = "gpt-5",
        model_image: str = "gpt-image-1",
        seed: int = 123,
    ):
        self.model_text = model_text
        self.model_image = model_image
        self.seed = seed
        self.offline = OfflineProvider(seed=seed)
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.ok = bool(self.api_key and _OPENAI_OK)
        self._client = None
        self._disabled_reason = None

        if self.ok:
            try:
                base_url_env = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
                base_url = base_url_env.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url = base_url + "/v1"

                org_id = os.getenv("OPENAI_ORGANIZATION", "")
                proj_id = os.getenv("OPENAI_PROJECT", "")

                self.org_name = os.getenv("OPENAI_ORG_NAME", "")
                self.organization = org_id
                self.project = proj_id

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=base_url,
                    organization=org_id if org_id else None,
                    project=proj_id if proj_id else None,
                )

                reason = sanity_check_openai(self._client, self.model_text)
                if reason is not None:
                    self._disable("Sanity check falhou (OpenAI indisponível/negado/modelo inválido)")
                    _write_offline_reason(reason)
                    print("[PROTEA-M] Motivo OFFLINE (resumo):")
                    for line in reason.splitlines()[:6]:
                        print("  ", line)
            except Exception as e:
                self._disable(f"Falha ao iniciar cliente OpenAI: {e}")

    def _disable(self, reason: str):
        if not self._disabled_reason:
            print(f"[PROTEA-M] OpenAI desativado nesta execução → {reason}")
        self.ok = False
        self._disabled_reason = reason
        self._client = None
        try:
            _write_offline_reason(reason)
        except Exception:
            pass

    # ---------- IMAGEM ----------
    def generate_image(self, prompt: str, out_path: Path) -> Dict[str, Any]:
        if not self.ok or self._client is None:
            return self.offline.generate_image(prompt, out_path)
        try:
            resp = self._client.images.generate(
                model=self.model_image,
                prompt=prompt,
                size="1536x1024",  # <-- TROCA AQUI
                # quality="high",     # <-- opcional (se quiser mais nitidez)
            )
            b64 = resp.data[0].b64_json
            img_bytes = base64.b64decode(b64)
            out_png = out_path.with_suffix(".png")
            safe_write_bytes(out_png, img_bytes)
            return {"provider": "openai", "path": str(out_png), "model": self.model_image}
        except Exception as e:
            self._disable(f"Falha imagem OpenAI ({e})")
            return self.offline.generate_image(prompt, out_path)

    # ---------- JSON ----------
    def complete_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.ok or self._client is None:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=self.model_text,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            self._disable(f"Falha JSON OpenAI ({e})")
            return None


def get_provider(gen_mode: str = "auto", gen_provider: str = "openai", seed: int = 123) -> GenerativeProviderBase:
    """
    gen_mode: 'auto' | 'offline'
    gen_provider: 'openai' (por enquanto, apenas OpenAI ou Offline)
    """
    gm = (gen_mode or "auto").strip().lower()
    gp = (gen_provider or "openai").strip().lower()

    if gm in ("offline", "off"):
        return OfflineProvider(seed=seed)

    if gp == "openai":
        return OpenAIProvider(
            model_text="gpt-5",
            model_image="gpt-image-1",
            seed=seed,
        )

    # fallback final
    return OfflineProvider(seed=seed)


def _log_generation_mode(gen_mode: str, gen_provider: str, provider_obj: GenerativeProviderBase):
    print(f"[PROTEA-M] Geração configurada: gen_mode={gen_mode}, gen_provider={gen_provider}")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    print(f"[PROTEA-M] OPENAI_API_KEY = {_mask_key(key)}")
    if isinstance(provider_obj, OpenAIProvider) and getattr(provider_obj, "ok", False):
        print(f"[PROTEA-M] Modo IA: OpenAI ✅  (texto={provider_obj.model_text}, imagem={provider_obj.model_image})")
        print(f"  - OPENAI_ORG = {getattr(provider_obj, 'org_name', '(não definido)')}")
        print(f"  - OPENAI_ORGANIZATION = {getattr(provider_obj, 'organization', '(não definido)')}")
        print(f"  - OPENAI_PROJECT = {getattr(provider_obj, 'project', '(não definido)')}")
    else:
        print("[PROTEA-M] Modo OFFLINE (placeholders) ⚠️")


# ==============================
# Modelo de dados simplificado
# ==============================
def _row_to_person(row: "pd.Series") -> Dict[str, Any]:
    return {k: row.get(k, "NA") for k in REQ_COLS}


# ==============================
# Lógica de História Social
# ==============================
def _desaster_profile(tipo_desastre: str) -> Dict[str, str]:
    """Retorna descrição curta do cenário, conforme o tipo de desastre."""
    td = (tipo_desastre or "").strip().lower()

    if td == "inundacao":
        return {
            "tipo": "inundação",
            "descricao": (
                "Cenário de enchentes e alagamentos urbanos ou próximos a rios. "
                "Foco em reconhecer sinais de subida da água, buscar locais altos "
                "e seguros, seguir orientações de responsáveis e defesas civis."
            ),
        }
    elif td == "deslizamento":
        return {
            "tipo": "deslizamento de terra",
            "descricao": (
                "Cenário de encostas com risco de deslizamento após chuvas fortes. "
                "Foco em identificar trincas, ruídos incomuns, dificuldade de abrir "
                "portas e janelas, e evacuação rápida para local seguro."
            ),
        }
    elif td == "incendio":
        return {
            "tipo": "incêndio",
            "descricao": (
                "Cenário de incêndio em casa ou escola. Foco em reconhecer cheiro de "
                "fumaça, alarmes, sair agachado, não usar elevadores e encontrar ponto "
                "de encontro seguro."
            ),
        }
    else:
        return {
            "tipo": td if td else "desastre",
            "descricao": (
                "Cenário de emergência relacionado a desastres. "
                "Foco em reconhecer sinais de perigo, seguir o plano de segurança "
                "e pedir ajuda a adultos de confiança."
            ),
        }

def _profile_prompt_snippet(pessoa: Dict[str, Any]) -> str:
    """Texto curto que descreve o perfil da criança para usar nos prompts de imagem."""
    nivel = (pessoa.get("NivelComunicacao") or "").strip().lower()
    forma = (pessoa.get("FormaPreferida") or "").strip().lower()
    suporte = (pessoa.get("SuporteNecessario") or "").strip().lower()
    ambiente = (pessoa.get("Ambiente") or "").strip().lower()
    parceiro = (pessoa.get("ParceiroComunicacional") or "").strip().lower()
    plano = (pessoa.get("PlanoIntervencao") or "").strip().lower()
    guia = (pessoa.get("PlanoGuia") or "").strip().lower()

    map_nivel = {
        "nao_verbal": "criança autista não verbal ou com comunicação muito limitada",
        "pre_verbal": "criança autista em fase pré-verbal, usando gestos ou apoios",
        "emergente": "criança autista com comunicação emergente",
        "frases_simples": "criança autista que fala em frases simples",
        "conversacional": "criança autista com fala conversacional",
    }
    map_forma = {
        "pecs": "usa principalmente cartões visuais / PECS",
        "aac": "usa recursos de comunicação alternativa (AAC)",
        "visual": "prefere apoios visuais bem claros",
        "verbal": "usa mais fala para se comunicar",
        "gestual": "usa muitos gestos para se comunicar",
        "escrita": "responde bem a palavras escritas simples",
    }
    map_suporte = {
        "alto": "necessita ALTO suporte, poucas informações por vez e cenas muito simples",
        "moderado": "necessita suporte MODERADO e cenas organizadas",
        "baixo": "necessita BAIXO suporte, mas ainda se beneficia de estrutura visual clara",
    }
    map_amb = {
        "casa": "ambiente doméstico (casa)",
        "escola": "ambiente escolar estruturado",
        "clinica": "ambiente clínico/terapêutico",
        "comunidade": "espaços da comunidade (rua, praça, abrigo)",
    }
    map_parc = {
        "pais": "pais ou responsáveis",
        "profissionais": "profissionais de saúde/educação",
        "pares": "colegas / outras crianças",
        "irmaos": "irmãos/irmãs",
        "desconhecidos": "adultos pouco familiares",
    }

    desc_nivel = map_nivel.get(nivel, "criança autista")
    desc_forma = map_forma.get(forma, "")
    desc_suporte = map_suporte.get(suporte, "")
    desc_amb = map_amb.get(ambiente, "")
    desc_parc = map_parc.get(parceiro, "")

    partes = [desc_nivel]
    if desc_forma:
        partes.append(desc_forma)
    if desc_suporte:
        partes.append(desc_suporte)
    if plano:
        partes.append(f"plano de intervenção focado em {plano}")
    if guia:
        partes.append(f"abordagem {guia}")
    if desc_amb:
        partes.append(f"cenário principal: {desc_amb}")
    if desc_parc:
        partes.append(f"interagindo com {desc_parc}")

    base = "; ".join(partes)
    return (
        base
        + ". Estilo visual: história em quadrinhos simples, alto contraste, poucos detalhes, "
          "sem sobrecarga sensorial, cores suaves, fundo limpo, expressões faciais simples."
    )


def _heuristic_social_story(
    pessoa: Dict[str, Any],
    tipo_desastre: str,
    media_lang: str = "pt-BR",
) -> Dict[str, Any]:
    """Geração heurística de história social (fallback quando IA não retorna JSON)."""

    prof = _desaster_profile(tipo_desastre)
    td_label = prof["tipo"]
    td_desc = prof["descricao"]

    nome_pessoa = pessoa.get("PessoaID", "a criança")
    perfil_snippet = _profile_prompt_snippet(pessoa)

    titulo = f"Minha história sobre segurança em caso de {td_label}"

    paineis = []

    # Painel 1 — Situação
    paineis.append({
        "id": "P1",
        "titulo_painel": "O que pode acontecer",
        "descricao_narrativa": (
            f"{nome_pessoa} mora em um lugar onde às vezes pode acontecer {td_label}. "
            f"É importante saber o que fazer para ficar mais seguro."
        ),
        "fala_personagem": "Eu posso aprender o que fazer quando algo perigoso acontece.",
        "texto_apoio_cuidador": (
            f"Explique em palavras simples o que é {td_label} e use imagens ou objetos concretos "
            f"para apoiar a compreensão, respeitando o ritmo da criança e seu perfil comunicacional."
        ),
        "foco_habilidade": "Psicoeducação sobre o desastre e segurança.",
        "prompt_imagem": (
            "primeiro quadro de uma página de história em quadrinhos; "
            "casa e arredores em situação tranquila, sugerindo risco de "
            f"{td_label} de forma suave, sem cenas de pânico. "
            + perfil_snippet
        ),
    })

    # Painel 2 — Sinais de alerta
    paineis.append({
        "id": "P2",
        "titulo_painel": "Como sei que algo está acontecendo",
        "descricao_narrativa": (
            f"{nome_pessoa} pode perceber alguns sinais de que {td_label} está acontecendo, como "
            f"mensagens no celular, barulho de sirenes ou avisos de adultos."
        ),
        "fala_personagem": "Quando alguém me avisa ou vejo sinais, eu presto atenção.",
        "texto_apoio_cuidador": (
            "Mostre exemplos de sinais de alerta (sirene, mensagem de texto, anúncio em alto-falante). "
            "Se necessário, use cartões visuais para cada sinal, organizados em sequência."
        ),
        "foco_habilidade": "Reconhecimento de sinais de alerta.",
        "prompt_imagem": (
            "segundo quadro em estilo de história em quadrinhos; celular com ícone de alerta, "
            "símbolo de sirene e um adulto apontando para um aviso, sem texto escrito na imagem. "
            + perfil_snippet
        ),
    })

    # Painel 3 — O que eu devo fazer
    paineis.append({
        "id": "P3",
        "titulo_painel": "Plano de ação",
        "descricao_narrativa": (
            f"Quando há risco de {td_label}, {nome_pessoa} segue o plano combinado com os adultos: "
            "ouvir o que eles dizem, ir para o lugar seguro e ficar junto da família ou cuidador."
        ),
        "fala_personagem": "Eu sigo as instruções do adulto e vou para o lugar combinado.",
        "texto_apoio_cuidador": (
            "Repita sempre o mesmo plano de ação, mostrando no mapa da casa/escola o caminho até o ponto seguro. "
            "Use setas visuais grandes e claras e pratique em forma de ensaio comportamental."
        ),
        "foco_habilidade": "Seguir rotas e instruções de segurança.",
        "prompt_imagem": (
            "terceiro quadro da página; adulto guiando a criança por um caminho simples com setas no chão "
            "até um local claramente marcado como seguro. "
            + perfil_snippet
        ),
    })

    # Painel 4 — Como posso me sentir
    paineis.append({
        "id": "P4",
        "titulo_painel": "Sentimentos e autocuidado",
        "descricao_narrativa": (
            f"Em situações de {td_label}, {nome_pessoa} pode sentir medo ou desconforto. "
            "Tudo bem sentir isso. A criança pode respirar fundo, apertar um objeto de conforto "
            "e lembrar que os adultos estão ajudando."
        ),
        "fala_personagem": "Eu posso respirar fundo e usar meu objeto de conforto. Os adultos estão comigo.",
        "texto_apoio_cuidador": (
            "Ensine estratégias de autorregulação compatíveis com o perfil sensorial da criança, "
            "como respiração guiada, pressão profunda ou uso de fones abafadores, se forem adequados."
        ),
        "foco_habilidade": "Regulação emocional e uso de estratégias de enfrentamento.",
        "prompt_imagem": (
            "quarto quadro da página; criança em local seguro segurando objeto de conforto, "
            "adulto próximo em postura acolhedora, ícones discretos de respiração profunda. "
            + perfil_snippet
        ),
    })

    # Painel 5 — Finalização
    paineis.append({
        "id": "P5",
        "titulo_painel": "Depois que tudo passa",
        "descricao_narrativa": (
            f"Depois que o perigo de {td_label} termina, {nome_pessoa} pode conversar com os adultos "
            "sobre o que aconteceu e como se sentiu. Assim, todos podem melhorar o plano para a próxima vez."
        ),
        "fala_personagem": "Quando tudo termina, eu posso falar sobre como me senti.",
        "texto_apoio_cuidador": (
            "Reserve um tempo para revisar o que funcionou bem e o que pode ser ajustado. "
            "Use imagens da própria história em quadrinhos para apoiar a conversa."
        ),
        "foco_habilidade": "Revisão da experiência e aprendizagem.",
        "prompt_imagem": (
            "quinto quadro da página; adulto e criança sentados à mesa com figuras representando os passos "
            f"da situação de {td_label}, ambos em postura calma. "
            + perfil_snippet
        ),
    })

    story = {
        "versao": __version__,
        "idioma": media_lang,
        "tipo_desastre": td_label,
        "descricao_desastre": td_desc,
        "pessoa": {
            "PessoaID": pessoa.get("PessoaID"),
            "PlanoIntervencao": pessoa.get("PlanoIntervencao"),
            "PlanoGuia": pessoa.get("PlanoGuia"),
            "NivelComunicacao": pessoa.get("NivelComunicacao"),
            "FuncaoComunicativa": pessoa.get("FuncaoComunicativa"),
            "Ambiente": pessoa.get("Ambiente"),
            "ParceiroComunicacional": pessoa.get("ParceiroComunicacional"),
            "SuporteNecessario": pessoa.get("SuporteNecessario"),
            "FormaPreferida": pessoa.get("FormaPreferida"),
            "DescricaoComportamento": pessoa.get("DescricaoComportamento"),
        },
        "titulo_historia": titulo,
        "objetivo_geral": (
            "Apoiar a compreensão da criança sobre o desastre e treinar, de forma visual e estruturada, "
            "as ações de segurança combinadas com os cuidadores, respeitando seu perfil comunicacional e sensorial."
        ),
        "paineis": paineis,
    }
    return story

def build_social_story_json(
    pessoa: Dict[str, Any],
    tipo_desastre: str,
    media_lang: str,
    provider: GenerativeProviderBase,
) -> Dict[str, Any]:
    """
    Tenta gerar a história social via IA (JSON); em caso de falha, usa heurística.
    """

    prof = _desaster_profile(tipo_desastre)
    td_label = prof["tipo"]
    td_desc = prof["descricao"]

    ctx = textwrap.dedent(f"""
    Você é um especialista em TEA e em elaboração de HISTÓRIAS SOCIAIS EM QUADRINHOS
    para situações de emergência e desastres.

    Gere uma história social EM {media_lang} para uma criança com TEA, relacionada ao tipo de desastre:
    - Tipo de desastre: {td_label}
    - Descrição do cenário: {td_desc}

    Perfil da criança (dados do Modelo 1):
      - PessoaID: {pessoa.get("PessoaID","")}
      - PlanoIntervencao: {pessoa.get("PlanoIntervencao","")}
      - PlanoGuia: {pessoa.get("PlanoGuia","")}
      - NivelComunicacao: {pessoa.get("NivelComunicacao","")}
      - FuncaoComunicativa: {pessoa.get("FuncaoComunicativa","")}
      - Ambiente: {pessoa.get("Ambiente","")}
      - ParceiroComunicacional: {pessoa.get("ParceiroComunicacional","")}
      - SuporteNecessario: {pessoa.get("SuporteNecessario","")}
      - FormaPreferida: {pessoa.get("FormaPreferida","")}
      - DescricaoComportamento: {pessoa.get("DescricaoComportamento","")}
    """).strip()

    system_prompt = (
        "Você deve responder ESTRITAMENTE em JSON válido. "
        "Nada de comentários, explicações fora do JSON ou blocos de código."
    )

    user_prompt = textwrap.dedent(f"""
       Usando o contexto abaixo, gere uma HISTÓRIA SOCIAL EM QUADRINHOS, estruturada, para
       preparar a criança para situações de {td_label}.

       Regras:
       - Linguagem simples, direta e respeitosa, adequada a crianças com TEA.
       - Use SEMPRE as informações de perfil (PlanoIntervencao, PlanoGuia, NivelComunicacao,
         FuncaoComunicativa, Ambiente, ParceiroComunicacional, SuporteNecessario, FormaPreferida,
         DescricaoComportamento) para adaptar conteúdo e foco de cada painel.
       - Os campos 'prompt_imagem' DEVEM citar explicitamente que a ilustração é adequada a
         uma criança autista com esse perfil, mencionando nível de comunicação, forma preferida,
         suporte necessário, ambiente e parceiro comunicacional. Também devem mencionar:
         baixo ruído visual, alto contraste, poucos detalhes, cores suaves, ausência de texto
         escrito na imagem.
       - Foque em:
           * psicoeducação sobre o desastre;
           * reconhecimento de sinais de alerta;
           * plano de ação (o que fazer);
           * regulação emocional;
           * revisão da experiência.
       - A história deve ser dividida em 4 a 6 PAINÉIS.
       - Cada painel deve conter:
           * id (ex.: "P1", "P2"...)
           * titulo_painel (frase curta)
           * descricao_narrativa (2–4 frases objetivas)
           * fala_personagem (uma fala curta em primeira pessoa)
           * texto_apoio_cuidador (orientações para o adulto que media a leitura)
           * foco_habilidade (o que esse painel trabalha)
           * prompt_imagem (descrição textual da imagem, sem texto escrito na imagem)

       Formato EXATO de saída:
    {{
      "versao": "...",
      "idioma": "{media_lang}",
      "tipo_desastre": "{td_label}",
      "descricao_desastre": "...",
      "pessoa": {{
        "PessoaID": "...",
        "PlanoIntervencao": "...",
        "PlanoGuia": "...",
        "NivelComunicacao": "...",
        "FuncaoComunicativa": "...",
        "Ambiente": "...",
        "ParceiroComunicacional": "...",
        "SuporteNecessario": "...",
        "FormaPreferida": "...",
        "DescricaoComportamento": "..."
      }},
      "titulo_historia": "...",
      "objetivo_geral": "...",
      "paineis": [
        {{
          "id": "P1",
          "titulo_painel": "...",
          "descricao_narrativa": "...",
          "fala_personagem": "...",
          "texto_apoio_cuidador": "...",
          "foco_habilidade": "...",
          "prompt_imagem": "..."
        }}
      ]
    }}

    CONTEXTO:
    {ctx}
    """).strip()

    data = None
    try:
        data = provider.complete_json(system_prompt, user_prompt)
    except Exception:
        data = None

    if not isinstance(data, dict) or "paineis" not in data or not data["paineis"]:
        # fallback heurístico
        return _heuristic_social_story(pessoa, tipo_desastre=tipo_desastre, media_lang=media_lang)

    # ajusta campos mínimos e garante consistência
    data.setdefault("versao", __version__)
    data.setdefault("idioma", media_lang)
    data.setdefault("tipo_desastre", td_label)
    data.setdefault("descricao_desastre", td_desc)

    data.setdefault("pessoa", {})
    for k in REQ_COLS:
        if k == "PessoaID":
            data["pessoa"].setdefault("PessoaID", pessoa.get("PessoaID"))
        else:
            data["pessoa"].setdefault(k, pessoa.get(k))

    data.setdefault(
        "titulo_historia",
        f"Minha história sobre segurança em caso de {td_label}",
    )
    data.setdefault(
        "objetivo_geral",
        "Apoiar a compreensão da criança sobre o desastre e treinar ações de segurança de forma visual e estruturada."
    )

    # garante que paineis é lista de dicts com campos básicos
    paineis_norm = []
    for i, p in enumerate(data.get("paineis", []), start=1):
        if not isinstance(p, dict):
            continue
        pn = dict(p)
        pn.setdefault("id", f"P{i}")
        pn.setdefault("titulo_painel", f"Painel {i}")
        pn.setdefault("descricao_narrativa", "")
        pn.setdefault("fala_personagem", "")
        pn.setdefault("texto_apoio_cuidador", "")
        pn.setdefault("foco_habilidade", "")
        pn.setdefault("prompt_imagem", "ilustração simples, alto contraste, estilo cartoon, sem texto escrito")
        paineis_norm.append(pn)

    if not paineis_norm:
        return _heuristic_social_story(pessoa, tipo_desastre=tipo_desastre, media_lang=media_lang)

    data["paineis"] = paineis_norm
    return data

def _grid_for(n: int) -> tuple[int, int]:
    # retorna (cols, rows)
    if n <= 4:
        return (2, 2)
    return (3, 2)  # 5 ou 6 painéis -> 3 colunas x 2 linhas (landscape)

def build_comic_page_prompt(story: Dict[str, Any]) -> str:
    """
    Monta um único prompt de imagem para gerar UMA página em formato de história
    em quadrinhos com todos os painéis.
    O prompt usa fortemente o perfil da pessoa + descrição de cada painel.
    """

    pessoa = story.get("pessoa", {}) or {}
    paineis = story.get("paineis", []) or []

    n = len(paineis)
    cols, rows = _grid_for(n)

    nivel = pessoa.get("NivelComunicacao", "")
    funcao = pessoa.get("FuncaoComunicativa", "")
    ambiente = pessoa.get("Ambiente", "")
    parceiro = pessoa.get("ParceiroComunicacional", "")
    suporte = pessoa.get("SuporteNecessario", "")
    forma = pessoa.get("FormaPreferida", "")
    comportamento = pessoa.get("DescricaoComportamento", "")
    plano_int = pessoa.get("PlanoIntervencao", "")
    plano_guia = pessoa.get("PlanoGuia", "")

    perfil_texto = textwrap.dedent(f"""
    Perfil detalhado da criança:
    - Criança com TEA.
    - Nível de comunicação: {nivel}.
    - Função comunicativa predominante: {funcao}.
    - Ambiente principal de treino: {ambiente}.
    - Parceiro comunicacional típico: {parceiro}.
    - Nível de suporte necessário: {suporte}.
    - Forma preferida de comunicação: {forma}.
    - Descrição do comportamento: {comportamento}.
    - Plano de intervenção: {plano_int}.
    - Abordagem/Plano guia: {plano_guia}.

    Respeitar esse perfil significa:
    - Estimular a função comunicativa principal ({funcao}) no contexto do ambiente ({ambiente}) com o parceiro ({parceiro}).
    - Ajustar a quantidade de detalhes visuais e previsibilidade de acordo com o nível de comunicação ({nivel})
      e o suporte necessário ({suporte}).
    - Priorizar recursos visuais compatíveis com a forma preferida ({forma}) e com o estilo sensorial da criança.
    """).strip()

    descr_paineis = []
    for i, p in enumerate(paineis, start=1):
        descr_paineis.append(
            f"Painel {i}: título '{p.get('titulo_painel','')}'. "
            f"Cena: {p.get('descricao_narrativa','')}"
        )

    descr_paineis_txt = "\n".join(descr_paineis)

    prompt = textwrap.dedent(f"""
    CRIE UMA ÚNICA IMAGEM em formato de página de HQ em orientação HORIZONTAL (landscape),
    com {n} quadros retangulares organizados em uma grade de {rows} linhas por {cols} colunas, 
    com bordas visíveis entre os quadros.

    Tema da história: {story.get('titulo_historia','História sobre segurança')}.

    {perfil_texto}

    Descrição de cada quadro (sem colocar texto escrito na imagem, apenas personagens,
    cenários, expressões e elementos visuais):

    {descr_paineis_txt}

    Instruções de estilo:
    - Estilo tipo história em quadrinhos para crianças, mas simples, limpo e com alto contraste.
    - Pensado para crianças autistas: poucos estímulos visuais, fundo limpo, sem excesso de detalhes,
      cores suaves porém contrastantes.
    - NÃO escrever nenhuma palavra, letra ou número dentro dos quadros (nem balões de fala nem caixas de texto).
    - Personagens com expressões faciais claras, porém estilizados de forma neutra (não realista).
    - A página inteira deve ser uma única imagem, com todos os quadros já organizados como em uma HQ.
    """).strip()

    return prompt

def generate_social_story_comic_for_person(
    pessoa: Dict[str, Any],
    dirs: Dict[str, Path],
    provider: GenerativeProviderBase,
    media_lang: str,
    tipo_desastre: str,
) -> Dict[str, Any]:
    """
    Gera JSON da história + UMA imagem de página em quadrinhos para uma pessoa.
    - story: contém todos os painéis com textos.
    - imagem_pagina_unica: caminho da imagem da HQ inteira.
    """

    # 1) Gera a história social estruturada (texto + meta)
    story = build_social_story_json(
        pessoa=pessoa,
        tipo_desastre=tipo_desastre,
        media_lang=media_lang,
        provider=provider,
    )

    pid_slug = slugify(pessoa.get("PessoaID", "sem_id"), 40)
    base_name = f"historia_social_{pid_slug}_{slugify(tipo_desastre, 20)}"

    json_path = dirs["textos"] / f"{base_name}.json"

    # 2) Monta o prompt da PÁGINA de HQ usando o perfil + descrição dos painéis
    comic_prompt = build_comic_page_prompt(story)

    # 3) Gera APENAS UMA IMAGEM: página completa em quadrinhos
    page_out = dirs["imagens"] / f"{base_name}_pagina_unica.png"
    meta = provider.generate_image(comic_prompt, page_out)
    page_path = meta.get("path", str(page_out))

    # 4) Salva no JSON o caminho da página única
    story["imagem_pagina_unica"] = page_path

    # (opcional) se quiser, faça cada painel apontar para a mesma imagem
    for p in story.get("paineis", []):
        p["imagem_pagina_unica"] = page_path

    # 5) Salva JSON final
    safe_write_text(json_path, json.dumps(story, ensure_ascii=False, indent=2))

    return {
        "PessoaID": pessoa.get("PessoaID"),
        "tipo_desastre": story.get("tipo_desastre"),
        "json_path": str(json_path),
        "imagem_pagina_unica": page_path,
        "n_paineis": len(story.get("paineis", [])),
    }

# ==============================
# Orquestração principal
# ==============================
def run_from_m1_social_story(
    matriz_csv: str,
    workspace: str,
    gen_mode: str = "auto",
    gen_provider: str = "openai",
    media_lang: str = "pt-BR",
    seed: int = 42,
    print_json: bool = False,
    max_pessoas: int = 1,
    tipo_desastre: str = "inundacao",
    random_choice: bool = True,
) -> Dict[str, Any]:
    random.seed(seed)

    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    global _PTI_LAST_WORKSPACE
    _PTI_LAST_WORKSPACE = ws

    df_m1_all = read_csv_auto(matriz_csv)
    df_m1_all = ensure_required_cols(df_m1_all)

    total_disponivel = len(df_m1_all)
    solicitado = int(max_pessoas or 1)
    if solicitado <= 0:
        solicitado = 1
    efetivo = min(solicitado, total_disponivel)

    if total_disponivel == 0:
        print("[PROTEA-M] Aviso: a matriz do Modelo 1 está vazia. Nada a processar.")
        return {"people": []}

    if solicitado > total_disponivel:
        print(
            f"[PROTEA-M] Solicitado processar {solicitado} pessoa(s), "
            f"mas a matriz possui apenas {total_disponivel}."
        )

    print(
        f"[PROTEA-M] Processando {efetivo} de {total_disponivel} pessoa(s) "
        f"(solicitado={solicitado})."
    )

    # Seleção aleatória ou sequencial das pessoas
    if random_choice:
        print("[PROTEA-M] Seleção de pessoas: MODO ALEATÓRIO (default).")

        df_m1 = df_m1_all.sample(n=efetivo).reset_index(drop=True)

        # Log das pessoas selecionadas
        pessoas_escolhidas = df_m1["PessoaID"].tolist()
        print(f"[PROTEA-M] Pessoas selecionadas aleatoriamente: {pessoas_escolhidas}")

    else:
        print("[PROTEA-M] Seleção de pessoas: MODO SEQUENCIAL (primeiras linhas da matriz).")

        df_m1 = df_m1_all.head(efetivo).reset_index(drop=True)

        # Log das pessoas selecionadas
        pessoas_escolhidas = df_m1["PessoaID"].tolist()
        print(f"[PROTEA-M] Pessoas selecionadas (sequencial): {pessoas_escolhidas}")

    provider = get_provider(gen_mode=gen_mode, gen_provider=gen_provider, seed=seed)
    _log_generation_mode(gen_mode, gen_provider, provider)

    results = {"people": [], "tipo_desastre": tipo_desastre}

    for _, row in df_m1.iterrows():
        pessoa = _row_to_person(row)
        dirs = ensure_dirs_person(ws, pessoa["PessoaID"])

        print(f"[PROTEA-M] Gerando história social para PessoaID={pessoa['PessoaID']}...")

        res_person = generate_social_story_comic_for_person(
            pessoa=pessoa,
            dirs=dirs,
            provider=provider,
            media_lang=media_lang,
            tipo_desastre=tipo_desastre,
        )

        results["people"].append(res_person)

    if print_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return results


# ==============================
# CLI
# ==============================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PROTEA-M — Geração de História Social em Quadrinhos (somente HQ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--matriz_csv",
        required=False,
        default="examples/matriz_caa_ajustada.csv",
        help="Planilha do Modelo 1.",
    )
    p.add_argument(
        "--workspace",
        required=False,
        default="workspace_social_story",
        help="Diretório de saída.",
    )
    p.add_argument(
        "--gen_mode",
        choices=["auto", "offline"],
        default="auto",
        help="Geração: IA online (auto) com fallback ou offline (placeholders).",
    )
    p.add_argument(
        "--gen_provider",
        choices=["openai"],
        default="openai",
        help="Provedor de IA ('openai', por enquanto).",
    )
    p.add_argument(
        "--media_lang",
        default="pt-BR",
        help="Idioma do texto da história social.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória.",
    )
    p.add_argument(
        "--print_json",
        action="store_true",
        help="Imprime o JSON-resumo ao final.",
    )
    p.add_argument(
        "--max_pessoas",
        "--n_pessoas",
        dest="max_pessoas",
        type=int,
        default=1,
        help="Quantidade de pessoas a processar (padrão=1; máximo = nº de linhas da matriz_caa_ajustada).",
    )
    p.add_argument(
        "--env_file",
        required=False,
        help="Caminho para .env ou api_key.env contendo variáveis (OPENAI_*).",
    )
    p.add_argument(
        "--tipo_desastre",
        required=False,
        default="inundacao",
        help="Tipo de desastre para contextualizar a história social (ex.: inundacao, deslizamento, incendio).",
    )
    p.add_argument(
        "--no_random_choice",
        dest="random_choice",
        action="store_false",
        help=(
            "Desativa a escolha aleatória das pessoas. "
            "Quando usado, processa sempre as primeiras linhas (com base em --max_pessoas)."
        ),
    )
    p.set_defaults(random_choice=True)

    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    load_env_file(args.env_file)
    print(f"[PROTEA-M] OPENAI_BASE_URL   = {os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"[PROTEA-M] OPENAI_PROJECT    = {os.getenv('OPENAI_PROJECT', '(não definido)')}")
    print(f"[PROTEA-M] OPENAI_API_KEY    = {_mask_key(os.getenv('OPENAI_API_KEY', ''))}")
    print(f"[PROTEA-M] OPENAI_ORG_NAME   = {os.getenv('OPENAI_ORG_NAME', '(não definido)')}")
    print(f"[PROTEA-M] OPENAI_ORG_ID     = {os.getenv('OPENAI_ORGANIZATION', '(não definido)')}")

    res = run_from_m1_social_story(
        matriz_csv=args.matriz_csv,
        workspace=args.workspace,
        gen_mode=args.gen_mode,
        gen_provider=args.gen_provider,
        media_lang=args.media_lang,
        seed=args.seed,
        print_json=args.print_json,
        max_pessoas=args.max_pessoas,
        tipo_desastre=args.tipo_desastre,
        random_choice=args.random_choice,
    )
    print("Processamento concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
