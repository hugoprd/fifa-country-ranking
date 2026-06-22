# Security Policy

- [PT-BR](#pt-br)
- [EN-US](#en-us)

----------------------------------------------------------------------

# [PT-BR]

## Versões Suportadas

Este é um projeto de pesquisa acadêmica em desenvolvimento ativo e, por enquanto, não segue um versionamento semântico formal (releases via tags). Por isso, apenas a branch `main` recebe atualizações de segurança no momento.

| Versão                          | Suportada           |
| -------------------------------- | -------------------- |
| `main` (desenvolvimento ativo)   | :white_check_mark:   |
| Commits/branches anteriores      | :x:                  |

> Esta tabela será atualizada com versões reais assim que o projeto atingir uma primeira release estável (`v1.0.0`).

## Reportando uma Vulnerabilidade

Se você encontrar uma vulnerabilidade de segurança neste projeto, por favor **não abra uma issue pública**. Use um dos canais abaixo:

1. **(Recomendado) GitHub Security Advisories** — acesse a aba [Security do repositório](https://github.com/hugoprd/fifa-country-ranking/security/advisories/new) e clique em "Report a vulnerability". Isso abre um canal de comunicação privado diretamente comigo.
   *(Para este link funcionar, é preciso habilitar "Private vulnerability reporting" em Settings → Security → Code security do repositório.)*

O que esperar:
- Confirmação de recebimento em até 5 dias úteis.
- Uma avaliação inicial (aceite ou recusa, com justificativa) em até 14 dias.
- Se a vulnerabilidade for aceita, você será mantido informado sobre o progresso da correção, e receberá os créditos pelo reporte (a menos que prefira anonimato) quando a correção for publicada.

### Escopo

Como este é um projeto de pesquisa, sem usuários finais nem dados pessoais armazenados em produção, o escopo de vulnerabilidades relevantes inclui principalmente:

- Vulnerabilidades em dependências do projeto (`torch`, `torchvision`, `keras`, `pandas`, `beautifulsoup4`, `soccerdata`, `loguru`, `tqdm`, `uv`) que afetem a execução do pipeline.
- Execução de código arbitrário via deserialização de modelos (ex.: checkpoints do PyTorch/Keras, `pickle`) ou via dados obtidos por scraping.
- Exposição acidental de credenciais ou chaves de API usadas na coleta de dados.

Fora de escopo: indisponibilidade (DoS) das fontes de dados de terceiros (FootyStats, Wikipedia, Kaggle/Transfermarkt) e vulnerabilidades nessas próprias plataformas externas.

----------------------------------------------------------------------

# [EN-US]

## Supported Versions

This is an academic research project under active development, and it does not yet follow formal semantic versioning (tagged releases). For this reason, only the `main` branch currently receives security updates.

| Version                       | Supported           |
| ------------------------------ | -------------------- |
| `main` (active development)   | :white_check_mark:   |
| Previous commits/branches      | :x:                  |

> This table will be updated with real version numbers once the project reaches a first stable release (`v1.0.0`).

## Reporting a Vulnerability

If you find a security vulnerability in this project, please **do not open a public issue**. Use one of the channels below instead:

1. **(Recommended) GitHub Security Advisories** — go to the [repository's Security tab](https://github.com/hugoprd/fifa-country-ranking/security/advisories/new) and click "Report a vulnerability". This opens a private communication channel directly with me.
   *(For this link to work, "Private vulnerability reporting" must be enabled under Settings → Security → Code security in the repository.)*

What to expect:
- Acknowledgment of receipt within 5 business days.
- An initial assessment (accepted or declined, with justification) within 14 days.
- If accepted, you'll be kept updated on the progress of the fix, and credited for the report (unless you prefer to remain anonymous) once the fix is published.

### Scope

Since this is a research project with no end users or personal data stored in production, relevant vulnerabilities mainly include:

- Vulnerabilities in project dependencies (`torch`, `torchvision`, `keras`, `pandas`, `beautifulsoup4`, `soccerdata`, `loguru`, `tqdm`, `uv`) that affect pipeline execution.
- Arbitrary code execution via model deserialization (e.g., PyTorch/Keras checkpoints, `pickle`) or via scraped data.
- Accidental exposure of credentials or API keys used for data collection.

Out of scope: denial of service against third-party data sources (FootyStats, Wikipedia, Kaggle/Transfermarkt) and vulnerabilities in those external platforms themselves.
