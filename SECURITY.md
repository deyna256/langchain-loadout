# Security policy

## Supported versions

While the version is `0.x`, only the latest release receives security fixes.

## Reporting a vulnerability

Report it privately through
[GitHub's vulnerability reporting](https://github.com/deyna256/langchain-skill-router/security/advisories/new),
not in a public issue. Include the affected version, what an attacker can do and the steps to reproduce it.
Keep real credentials and user data out of the report.

Relevant areas include the handling of judge credentials such as `TYPESAFE_API_KEY`, and what the library
sends to a judge: the request, the recent conversation and the skills' text.
