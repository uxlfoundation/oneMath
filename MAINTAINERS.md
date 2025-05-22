# Introduction

This document defines roles in oneMath project.

# Roles and responsibilities

oneMath project defines four main roles:
 * [Contributor](#contributor)
 * [Domain maintainer](#domain-maintainer)
 * [Backend maintainer](#backend-maintainer)
 * [Architecture maintainer](#architecture-maintainer)

These roles are merit based. Refer to the corresponding section for specific
requirements and the nomination process.

## Contributor

A Contributor invests time and resources to improve oneMath project.
Anyone can become a Contributor by bringing value in one of the following ways:
  * Answer questions from community members.
  * Submit feedback to design proposals.
  * Review and/or test pull requests.
  * Test releases and report bugs.
  * Contribute code, including bug fixes, features implementations,
and performance optimizations.
  * Contribute design proposals.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow the project [contributing guidelines](CONTRIBUTING.md).

Privileges:
  * Eligible to join one of the maintainer groups.

## Domain Maintainer

Domain maintainer has responsibility for a specific domain in the project.
Domain maintainers are collectively responsible for developing and maintaining their domain,
including reviewing all changes to their domain and indicating
whether those changes are ready to merge. They have a track record of
contribution and review in the project.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow and enforce the project [contributing guidelines](CONTRIBUTING.md).
  * Co-own with other domain maintainers a specific domain, including contributing
    bug fixes, implementing features, and answering domain specific questions
    in [#onemath](https://uxlfoundation.slack.com/archives/onemath) Slack channel.
  * Review pull requests in their specific domain.
  * Monitor testing results and flag issues in their specific areas of
    responsibility.
  * Support and guide Contributors.

Requirements:
  * Experience as Contributor in the specific domain for at least 6 months.
  * Commit at least 25% of working time to the project.
  * Track record of accepted code contributions to a specific domain.
  * Track record of contributions to the code review process.
  * Demonstrated in-depth knowledge of the specific domain.
  * Commits to being responsible for that specific domain.

Privileges:
  * PR approval counts towards approval requirements for a specific domain.
  * Can promote fully approved Pull Requests to the `develop` branch.
  * Can recommend Contributors to become Domain maintainer.
  * Eligible to become an Architecture maintainer.

The process of becoming a Domain maintainer is:
1. A Contributor requests to join corresponding Domain maintainer GitHub team.
2. At least one specific Domain maintainer approves the request.

### List of GitHub teams for Domain maintainers

| GitHub team name | Domain maintainers |
:-----------|:------------|
| @uxlfoundation/onemath-blas-write | oneMath BLAS maintainers |
| @uxlfoundation/onemath-dft-write | oneMath DFT maintainers |
| @uxlfoundation/onemath-lapack-write | oneMath LAPACK maintainers |
| @uxlfoundation/onemath-rng-write | oneMath RNG maintainers |
| @uxlfoundation/onemath-sparse-write | oneMath Sparse Algebra maintainers |
| @uxlfoundation/onemath-vm-write | oneMath Vector Math maintainers |

## Backend Maintainer

Backend maintainer has responsibility for a specific backend in the project.
Backend maintainers are collectively responsible for developing and maintaining their backends,
including reviewing certain changes to their backend and indicating
whether those changes are ready to merge. They have a track record of
contribution and review in the project.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow and enforce the project [contributing guidelines](CONTRIBUTING.md).
  * Co-own with other backend maintainers a specific backend, including contributing
    bug fixes, implementing features, and answering backend specific questions
    in [#onemath](https://uxlfoundation.slack.com/archives/onemath) Slack channel.
  * Review pull requests pertaining to their specific backend.
  * Monitor testing results and flag issues in their specific areas of
    responsibility.
  * Support and guide Contributors.

Requirements:
  * Experience as Contributor in the specific backend for at least 6 months.
  * Track record of accepted code contributions to a specific backend.
  * Track record of contributions to the code review process.
  * Demonstrated in-depth knowledge of the specific backend.
  * Commits to being responsible for that specific backend.

Privileges:
  * PR approval counts towards approval requirements for a specific backend.
  * Can promote fully approved Pull Requests to the `develop` branch.
  * Can recommend Contributors to become Backend maintainers.
  * Eligible to become an Architecture maintainer.

The process of becoming a Backend maintainer is:
1. A Contributor requests to join corresponding Backend maintainer GitHub team.
2. At least one specific Backend maintainer approves the request.

### List of GitHub teams for Backend maintainers

| GitHub team name | Backend maintainers |
:-----------|:------------|
| @uxlfoundation/onemath-cpu-aarch64 | AArch64 CPU maintainers |
| @uxlfoundation/onemath-cpu-x64 | x86 CPU maintainers |
| @uxlfoundation/onemath-gpu-amd | AMD GPU maintainers |
| @uxlfoundation/onemath-gpu-intel | Intel GPU maintainers |
| @uxlfoundation/onemath-gpu-nvidia | NVIDIA GPU maintainers |

## Architecture Maintainer
Architecture maintainers are the most established contributors who are responsible for the
project technical direction and participate in making decisions about the
strategy and priorities of the project.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow and enforce the project [contributing guidelines](CONTRIBUTING.md)
  * Co-own with other Domain maintainers on the technical direction of a specific domain.
  * Co-own with other Backend maintainers on the support and implementation of a specific backend.
  * Co-own with other Architecture maintainers on the project as a whole, including
    determining strategy and policy for the project.
  * Support and guide Contributors, Domain maintainers, and Backend maintainers.

Requirements:
  * Experience as a Domain or Backend maintainer or Contributor with focus on the project
    architecture for at least 12 months.
  * Commit at least 25% of working time to the project.
  * Track record of major project contributions.
  * Demonstrated deep knowledge of the project architecture and build.
  * Demonstrated broad knowledge of the project across multiple domains and/or multiple backends.
  * Is able to exercise judgment for the good of the project, independent of
    their employer, friends, or team.

Privileges:
  * Can represent the project in public as a Maintainer.
  * Can recommend Contributor, Domain maintainer, or Backend maintainer to become Architecture maintainer.

Process of becoming a maintainer:
1. A Contributor, Domain maintainer, or Backend maintainer requests to join oneMath Architecture
   maintainers GitHub team (@uxlfoundation/onemath-arch-write).
2. At least one of the existing Architecture maintainers approves the request.

