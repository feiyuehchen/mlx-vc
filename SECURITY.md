# Security Policy

## Supported versions

Only the latest minor release of mlx-vc receives security fixes.  We are pre-1.0; the API may change between minor versions.

| Version | Supported          |
|---------|--------------------|
| 0.x     | :white_check_mark: |

## Reporting a vulnerability

Please report security issues **privately** so they can be patched before public disclosure.

- **GitHub Security Advisory** (preferred): use the "Report a vulnerability" button in the [Security tab](https://github.com/feiyuehchen/mlx-vc/security/advisories).
- **Email**: as a fallback, contact the maintainer at `brian9436et@gmail.com` with `[mlx-vc security]` in the subject line.

Please include:

1. A description of the vulnerability and its impact (RCE, sandbox escape, weight integrity, etc.)
2. Step-by-step reproduction (a minimal proof-of-concept is ideal)
3. The affected version / commit hash
4. Any suggested fix or mitigation you have in mind

We aim to acknowledge reports within **3 business days** and to ship a fix or mitigation within **30 days** of confirmation, whichever the severity warrants.

## Scope

mlx-vc loads pre-trained weights from third-party HuggingFace repos and reference repositories (Plachtaa/seed-vc, OlaWod/FreeVC, ASLP-lab/MeanVC, Acelogic's RVC port, etc.).  The integrity / safety of those upstream artifacts is **out of scope** for this policy — please report problems with those weights to their respective maintainers.

In scope:

- Code-execution risks in our subprocess runner / FastAPI server
- Path traversal in the WS / batch endpoints
- Crash-on-malformed-input bugs that affect availability of the server
- Issues in the reference-resolution logic (`MLX_VC_REF_DIR`, upload paths)
- Data leakage in the demo backends (e.g. logs, tmp files)

Out of scope:

- Attacks that require an attacker who already has filesystem write access
- Theoretical issues with no impact on the running system
- Issues in upstream torch / mlx / huggingface_hub / etc. — please report to those projects directly
